import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from error_map.utils.constants import TaxonomyParams, dataset2params
from .core.config import Config
from .stages import prepare_data, analyze_single_errors, construct_taxonomy_recursively
from .utils.cache import cached
from .inference import InferenceClient


class ErrorMap:
    
    def __init__(self, 
                 inference_type: str = "litellm-mock",
                 litellm_config: Optional[Dict] = None,
                 exp_id: str = None,
                 output_dir: Path = None,
                 data_path: str = "data",
                 datasets: List[str] = None,
                 dataset_params: Dict = None,
                 judge: Optional[str] = None,
                 provider: Optional[str] = None,
                 seed: Optional[int] = None,
                 models: List[str] = None, 
                 ratio: float = 0.1,
                 max_workers: int = 100,
                 use_correct_predictions: bool = True,
                 rare_freq: float = 0.0,
                 cols_to_keep: List[str] = None,
                 ):
        
        
        """
            inference_type (str): Specifies the inference backend type. Default is "litellm-mock" for testing without actual model calls.
            litellm_config (Optional[Dict]): Configuration dictionary for LiteLLM, used when inference_type involves real model inference.
            exp_id (str): Unique identifier for the experiment, useful for tracking and logging results.
            output_dir (Path): Directory to store output files.
            data_path (str): Path to the data directory. Default is "data".
            datasets (List[str]): List of dataset names to be used.
            dataset_params (Dict): Parameters specific to each dataset.
            judge (Optional[str]): Judge model or method used for evaluation.
            provider (Optional[str]): Provider name for inference.
            seed (Optional[int]): Random seed for reproducibility.
            models (List[str]): List of model names to be used.
            ratio (float): Sampling ratio for data.
            max_workers (int): Maximum number of parallel workers for processing.
            use_correct_predictions (bool): Utilize correct predictions from other models as references for the analyzer.
            rare_freq (float): Avoid long-tail categories (categories with a frequency below the specified threshold will be combined into an “Other” category).
            cols_to_keep (List[str]): Control the output file and include additional instance-level information from the input data file.
        """
        
        self.inference_type = inference_type
        self.exp_id = exp_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(output_dir or Path("output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_correct_predictions = use_correct_predictions
        self.models = models
        self.ratio = ratio
        self.litellm_config = litellm_config
        self.rare_freq = rare_freq
        self.cols_to_keep = cols_to_keep
        
        # save exp. config params
        params = {
            "inference_type": inference_type,
            "exp_id": str(exp_id),
            "output_dir": str(self.output_dir),
            "data_path": data_path,
            "datasets": datasets,
            "dataset_params": dataset_params,
            "judge": judge,
            "provider": provider,
            "seed": seed,
            "max_workers": max_workers,
            "use_correct_predictions": use_correct_predictions,
            "models": models,
            "ratio": ratio,
            "litellm_config": litellm_config,
            "rare_freq": rare_freq,
        }
        with open(os.path.join(self.output_dir, "config__exp_id=" + self.exp_id + ".json"), "w") as f:
            json.dump(params, f, indent=4)

        # Create config object
        self.config = Config(
            data_path=data_path,
            output_dir=self.output_dir,
            datasets=datasets or [],
            dataset_params=dataset_params or dataset2params,
            taxonomy_params=TaxonomyParams(),
            seed=seed,
        )

        # Setup inference client
        self.inference_client = InferenceClient(
            inference_type=inference_type,
            judge=judge,
            max_workers=max_workers,
            provider=provider,
            litellm_config=litellm_config,
        )
        
        # Apply output_dir to cached functions
        global prepare_data, analyze_single_errors, construct_taxonomy_recursively
        prepare_data = cached("data_preparation", self.output_dir)(prepare_data.__wrapped__)
        analyze_single_errors = cached("single_error", self.output_dir)(analyze_single_errors.__wrapped__)
        construct_taxonomy_recursively = cached("construct_taxonomy_recursively", self.output_dir)(construct_taxonomy_recursively.__wrapped__)
    
    async def run(self) -> Dict:
        print(f"🚀 Running error analysis: {self.exp_id}")
        
        data = await prepare_data(
            exp_id=self.exp_id,
            config=self.config,
            models=self.models,
            ratio=self.ratio,
        )
        print(f"📊 Prepared {len(data)} records")

        errors = [r for r in data if r.get('error', False)]
        if errors:
            analyzed = await analyze_single_errors(
                records=data,
                config=self.config,
                exp_id=self.exp_id,
                inference_client=self.inference_client,
                use_correct_predictions = self.use_correct_predictions,
            )
            print(f"🔍 Analyzed {len(analyzed)} errors")
        else:
            analyzed = []
            print("ℹ️ No errors to analyze")

        if analyzed:
            await construct_taxonomy_recursively(
                records=analyzed, 
                config=self.config, 
                exp_id=self.exp_id,
                inference_client=self.inference_client,
                rare_freq=self.rare_freq,
                cols_to_keep=self.cols_to_keep,
            )
        else:
            print("ℹ️ No errors to build taxonomy")

        return {
            "exp_id": self.exp_id,
            "total_records": len(data),
            "error_records": len(analyzed),
            "completed_at": datetime.now().isoformat()
        }



async def run(
            inference_type: str = "litellm-mock",
            litellm_config: Optional[Dict] = None,
            datasets: List[str] = None,
            dataset_params: Dict = None,
            seed: Optional[int] = None,
            models: List[str] = None, 
            ratio: float = 0.1,
            ) -> Dict:
    
    error_map = ErrorMap(
            inference_type=inference_type,
            datasets=datasets,
            dataset_params=dataset_params,
            seed=seed,
            models=models,
            ratio=ratio,
            litellm_config=litellm_config,
    )
    return await error_map.run()


__version__ = "0.1.0"
__all__ = ["ErrorMap", "run"]