from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class TaxonomyParams:

    batch_size: int = field(
        default=500,
        metadata={"description": "Size of minibatches for data processing."}
    )
    classify_batch_size: int = field(
        default=50,
        metadata={"description": "Size of minibatches for item classification."}
    )
    suggestion_length: int = field(
        default=30,
        metadata={"description": "Maximum length for taxonomy suggestions"}
    )
    cluster_name_length: int = field(
        default=5,
        metadata={"description": "Maximum length for cluster names"}
    )
    cluster_description_length: int = field(
        default=30,
        metadata={"description": "Maximum length for cluster descriptions"}
    )
    explanation_length: int = field(
        default=20,
        metadata={"description": "Maximum length for explanations"}
    )
    max_num_clusters: int = field(
        default=25,
        metadata={"description": "Maximum number of clusters allowed"}
    )
    taxonomy_update_repeat: int = field(
        default=10,
        metadata={"description": "Minimum number of repeating the taxonomy update stage."}
    )

    def get(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def get_modified_taxonomy_params(cls, param2val: Dict) -> Dict:
        params = asdict(cls())
        for k, v in param2val.items():
            params[k] = v
        return params



dataset2params: Dict[str, Any] = {

    "mmlu_pro_old":
        {
            "multiple_choice": True,
            "categories": True,
            "success_threshold": 1,
        },
    "gpqa":
        {
            "multiple_choice": True,
            "success_threshold": 1,
        },
    "omni_math": 
        {
            "success_threshold": 0.5,
        },

    "medhelm_v2_aci_bench": 
        {
            "success_threshold": 3.5,
        },
    "medhelm_v2_med_dialog_subset_healthcaremagic":
        {
            "success_threshold": 3.5,
        },
    "medhelm_v2_med_dialog_subset_icliniq":
        {
            "success_threshold": 3.5,
        },
    "medhelm_v2_medec": 
        {
            "success_threshold": 0.7,
        },
    "medhelm_v2_medi_qa": 
        {
            "success_threshold": 3.5,
        },
    "medhelm_v2_medication_qa":
        {
            "success_threshold": 3.5,
        },
    "medhelm_v2_mtsamples_procedures":
        {
            "success_threshold": 3.5,
        },
    "medhelm_v2_mtsamples_replicate":
        {
            "success_threshold": 3.5,
        },

}

REQUIRED_DATA_COLUMNS = ["example_id", "model", "input_text", "output_text", "score"]
