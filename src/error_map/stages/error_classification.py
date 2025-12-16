
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from error_map.stages.taxonomy_construction import _extract_description
from ..utils.cache import cached
from ..core.config import Config
from ..inference import InferenceClient
from collections import Counter
from tqdm.asyncio import tqdm_asyncio


def get_last_exsiting_taxonomy(error_taxonomy: List[Dict]) -> Dict:
    taxonomy_dict = None
    for i in range(len(error_taxonomy) - 1, -1, -1):
        taxonomy_json = error_taxonomy[i]["judge_response"]
        try:
            taxonomy_dict = json.loads(taxonomy_json)
            num_categories = len(taxonomy_dict["clusters"])
            if i == len(error_taxonomy) - 1:
                print(f"Using reviewed taxonomy, with {num_categories} categories..")
            else:
                print(f"Reviewed taxonomy wasn't found! Using taxonomy from iter: {i+1}/{len(error_taxonomy)}, with {num_categories} categories..")
            break
        except:
            continue
    return taxonomy_dict


async def classify_batch(description_batch: List, taxonomy: Dict, inference_client: InferenceClient, field: str) -> Dict:

    template_vars = {
        "data_type": field,
        "data": description_batch,
        "taxonomy": taxonomy,
    }

    try:
        result = await inference_client.infer("classify_errors.j2", template_vars, schema_name="classify_errors_schema.json")
        return {
            "prompt": result["prompt"],
            "judge_model": result["model"],
            "judge_response": result["content"],
            "template_used": result["template"],
            "inference_success": result["success"],
            "full_response": result["full_response"]
        }
    except Exception as e:
        return {"error": str(e), "batch": description_batch}



async def classify_errors(
    error_records: List[Dict],
    error_taxonomy: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    field: str = "error_title",
) -> List[Dict]:
    print(f"Classifying {len(error_records)} errors to taxonomy...")

    # get unique error list
    description_results = await asyncio.gather(*[_extract_description(record, field) for record in error_records])
    descriptions = list(set([i for i in description_results if i]))

    # use final existing taxonomy
    taxonomy = get_last_exsiting_taxonomy(error_taxonomy)

    # send error batches to be classified
    batch_size = config.taxonomy_params["classify_batch_size"]
    description_batches = [descriptions[i:i + batch_size] for i in range(0, len(descriptions), batch_size)]

    # classify batches in parallel
    return await tqdm_asyncio.gather(*[classify_batch(description_batch, taxonomy, inference_client, field) for description_batch in description_batches])

