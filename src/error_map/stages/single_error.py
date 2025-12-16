import asyncio
import csv
import os
from pathlib import Path
import random
import string
import sys
from typing import List, Dict
from ..utils.cache import cached
from ..core.config import Config
from ..inference import InferenceClient
import ast
from tqdm.asyncio import tqdm_asyncio


async def analyze_record(record: Dict, inference_client: InferenceClient, success_outputs: Dict, use_correct_predictions: bool) -> Dict:
    # Add correct outputs
    key = (record['dataset'], record['example_id'])
    record['correct_output_list'] = success_outputs.get(key, [])

    # Analyze with inference
    template_vars = {
        "input_text": record.get('input_text', ''),
        "output_text": record.get('output_text', ''),
        "correct_answer": record.get('correct_answer', ''),
    }
    if use_correct_predictions:
        template_vars['correct_outputs'] = random.sample(record['correct_output_list'], 1) if record['correct_output_list'] else []

    result = {
        **record,
        "prompt": "",
        "judge_model": "",
        "judge_response": "",
        "template_used": "",
        "inference_success": False,
        "full_response": ""
    }

    try:
        inference_result = await inference_client.infer(
            "single_error_analysis.j2",
            template_vars,
            schema_name="single_error_schema.json"
        )

        result.update({
            "prompt": inference_result.get("prompt", ""),
            "judge_model": inference_result.get("model", ""),
            "judge_response": inference_result.get("content", ""),
            "template_used": inference_result.get("template", ""),
            "inference_success": inference_result.get("success", False),
            "full_response": inference_result.get("full_response", "")
        })

    except Exception as e:
        result["full_response"] = str(e)

    return result


async def _process_record_for_filtering(record: Dict) -> Dict:
    is_error = record.get('error', False)
    if is_error:
        return {'type': 'error', 'record': record}
    else:
        key = (record['dataset'], record['example_id'])
        return {'type': 'success', 'key': key, 'output': record['output_text']}


async def _filter_and_build_lookup(records: List[Dict]) -> tuple[List[Dict], Dict]:
    """Filter error records and build success outputs lookup in parallel"""
    results = await asyncio.gather(*[_process_record_for_filtering(record) for record in records])
    
    # Separate error records and build success lookup
    error_records = []
    success_outputs = {}
    
    for result in results:
        if result['type'] == 'error':
            error_records.append(result['record'])
        else:
            key = result['key']
            if key not in success_outputs:
                success_outputs[key] = []
            success_outputs[key].append(result['output'])
    
    return error_records, success_outputs


@cached("single_error", None)
async def analyze_single_errors(
    records: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    use_correct_predictions: bool,
) -> List[Dict]:
    
    # Filter error records and build success lookup in parallel
    error_records, success_outputs = await _filter_and_build_lookup(records)
    
    if not error_records:
        print("No error records found")
        return []

    print(f"Analyzing {len(error_records)} error records in parallel...")

    # Analyze all errors in parallel
    return await tqdm_asyncio.gather(*[analyze_record(record, inference_client, success_outputs, use_correct_predictions) for record in error_records])
