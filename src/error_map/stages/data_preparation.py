import asyncio
from typing import List, Dict, Optional, Union
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ..utils.constants import REQUIRED_DATA_COLUMNS
from ..utils.cache import cached
from ..core.config import Config


async def _process_and_filter_record(record: Dict, config: Config, models: Optional[List[str]], thresholds: List[Dict]) -> Dict:
    dataset = record['dataset']
    threshold = config.dataset_params.get(dataset, {}).get('success_threshold', thresholds.get(dataset, 0.7))
    record['error'] = record['score'] < threshold
    
    # Mark for filtering if needed
    if models and record['model'] not in models and record['error']:
        record['_filtered'] = True
    else:
        record['_filtered'] = False
    
    return record


def _load_csv_sync(data_path: str, dataset: str) -> List[Dict]:
    try:
        df = pd.read_csv(f"{data_path}/{dataset}.csv")
        missing_cols = [col for col in REQUIRED_DATA_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset '{dataset}.csv' is missing required columns: {missing_cols}")
        df["dataset"] = dataset
        return df.to_dict('records')
    except FileNotFoundError:
        print(f"Dataset {dataset}.csv not found, skipping...")
        return []

async def _load_dataset_async(data_path: str, dataset: str) -> List[Dict]:
    """Async wrapper for CSV loading"""
    # Run CSV loading in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, _load_csv_sync, data_path, dataset)


def _sample_failures(failures: List[Dict], ratio: float, seed: int) -> List[Dict]:
    df = pd.DataFrame(failures)
    sampled_df = (
        df.groupby(['model', 'dataset'], group_keys=False)
        .sample(frac=ratio, random_state=seed)
        .reset_index(drop=True)   
    )
    return sampled_df.to_dict('records')


@cached("data_preparation", None)
async def prepare_data(
    exp_id: str,
    config: Config,
    models: Optional[List[str]] = None,
    ratio: float = 0.1,
) -> List[Dict]:
    print("Loading data...")
    
    tasks = [_load_dataset_async(config.data_path, dataset) for dataset in config.datasets]
    dataset_results = await asyncio.gather(*tasks)
    
    # Flatten all records - let exceptions propagate
    records = []
    for dataset_records in dataset_results:
        records.extend(dataset_records)

    # Calculate the default threshold fro each dataset
    if records:
        df = pd.DataFrame(records)
        thresholds_df = df.groupby("dataset")["score"].mean().reset_index()
        thresholds_dicts = thresholds_df.to_dict(orient="records")
        ds2threshold = {item["dataset"]: round(item["score"] * 0.7, 2) for item in thresholds_dicts}

    # Process error flags and filtering in parallel
    records = await asyncio.gather(*[_process_and_filter_record(record, config, models, ds2threshold) for record in records])
    
    # Filter out records marked for filtering
    records = [r for r in records if not r.get('_filtered', False)]
    
    # Split into failures and successes
    failures = [r for r in records if r['error']]
    successes = [r for r in records if not r['error']]

    if failures:
        sampled_failures = _sample_failures(failures, ratio, config.seed)
    else:
        sampled_failures = []

    print(f"Total Num. of Errors: {len(failures)}, "
          f"Sampled for Analysis: {len(sampled_failures)} "
          f"(ratio: {ratio*100:.1f}%)")

    return sampled_failures + successes