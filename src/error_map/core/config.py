import csv
import os
from pathlib import Path
import sys
from typing import Dict, List
from error_map.utils.constants import TaxonomyParams


class Config:
    def __init__(self, 
                 data_path: str = None,
                 output_dir: Path = None,
                 datasets: List[str] = None,
                 dataset_params: Dict = None,
                 taxonomy_params: TaxonomyParams = None,
                 seed: int = None):
        self.data_path = data_path or "data"
        self.output_dir = output_dir or Path("output")
        self.datasets = datasets or []
        self.dataset_params = dataset_params or {}
        self.taxonomy_params = taxonomy_params.get() if taxonomy_params else {}
        self.seed = seed

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # in case you are provided with a path but no datasets, run all of them
        if self.data_path and not self.datasets:
            self.datasets = [f.split(".")[0] for f in os.listdir(self.data_path) 
                             if os.path.isfile(os.path.join(self.data_path, f)) 
                             and f.endswith(".csv")]
