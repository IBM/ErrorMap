import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional
from ..utils.cache import cached
from ..core.config import Config
from ..inference import InferenceClient
from collections import Counter
import random
from tqdm import tqdm

async def _extract_description(record: Dict, field: str) -> Optional[Dict[Any, str]]:
    judge_response_str = record.get('judge_response', '')
    try:
        judge_response = json.loads(judge_response_str)
    except (json.JSONDecodeError, TypeError):
        return None
    
    if isinstance(judge_response, dict):
        final_answer = judge_response.get('final_answer', {})
        if isinstance(final_answer, dict):
            response = final_answer.get(field, '')
        else:
            response = ''
    else:
        response = ''
        
    return response if response else None



async def construct_taxonomy(
    error_records: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    field: str = "error_title",
    taxonomy_params: Dict = None,
    repeat_samples: int = 0,
) -> List[Dict]:
    print(f"Constructing taxonomy from {len(error_records)} error records...")

    taxonomy_params = config.taxonomy_params if taxonomy_params is None else taxonomy_params

    # shufle before taxonomy construction, to make the batches more varied
    random.seed(config.seed)
    random.shuffle(error_records)

    description_results = await asyncio.gather(*[_extract_description(record, field) for record in error_records])
    
    counts = Counter(description_results)
    descriptions = [(item, count) for item, count in counts.items()]

    if not descriptions:
        print(f"No error descriptions found in field '{field}'")
        return [{"judge_model": "none", "judge_response": "No descriptions found", "field": field}]
    
    taxonomies = []
    batch_size = taxonomy_params["batch_size"]
    range_from = 0
    if repeat_samples:
        range_to = repeat_samples
        range_jump = 1
    else:
        range_to = len(descriptions)
        range_jump = batch_size

    for i in tqdm(range(range_from, range_to, range_jump)):
        if repeat_samples:
            curr_batch = random.sample(descriptions, min(len(descriptions), batch_size))
        else:
            curr_batch = descriptions[i:i + batch_size]
        template_vars = {
            **taxonomy_params,
            "data_type": field,
            "data": curr_batch,
        }

        if i == 0: # first run: taxonomy generation 
            result = await inference_client.infer("taxonomy_generation.j2", template_vars, schema_name="generate_taxonomy_schema.json")
        else: # the rest: update taxonomy runs
            if result and result["content"]:
                template_vars["cluster_list"] = result["content"]
                result = await inference_client.infer("taxonomy_update.j2", template_vars, schema_name="update_taxonomy_schema.json")

        taxonomy_result = {
                "num_errors": len(curr_batch),
                "error_batch": curr_batch,
                "judge_model": result["model"],
                "judge_response": result["content"],
                "field": field,
                "prompt": result["prompt"],
                "template_used": result["template"],
                "inference_success": result["success"],
                "taxonomy_params": taxonomy_params,
            }
        taxonomies.append(taxonomy_result)
    
    # taxonomy review
    template_vars = {
        **taxonomy_params,
        "data_type": field,
        "cluster_list": result["content"],
    }
    result = await inference_client.infer("taxonomy_review.j2", template_vars, schema_name="review_taxonomy_schema.json")

    taxonomy_result = {
            "judge_model": result["model"],
            "judge_response": result["content"],
            "field": field,
            "prompt": result["prompt"],
            "template_used": result["template"],
            "inference_success": result["success"],
            "taxonomy_params": taxonomy_params,
        }
    taxonomies.append(taxonomy_result)

    return taxonomies
