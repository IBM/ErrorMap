import json
import os
from typing import Any, List, Dict, Union, Set
from error_map.stages.error_classification import classify_errors
from error_map.stages.taxonomy_construction import construct_taxonomy
from error_map.stages.taxonomy_population import populate_taxonomy
from error_map.utils.constants import TaxonomyParams
from error_map.utils.taxonomy_tree import TaxonomyNode, TaxonomyTree
from ..utils.cache import cached
from ..core.config import Config
from ..inference import InferenceClient
import math
import asyncio


async def _run_taxonomy_stages(
    records: List[Dict], 
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    parent_category_name: str = None,
    rare_freq: float = None,
) -> List[Dict]:

    # define curr taxonomy size (max num of clusters)
    fixed_max_clusters = config.taxonomy_params["max_num_clusters"]
    curr_max_clusters = min(fixed_max_clusters, math.ceil(len(records)*0.1))
    curr_taxonomy_params = TaxonomyParams.get_modified_taxonomy_params({"max_num_clusters": curr_max_clusters})

    # in case there is already a parent category, specify it in the prompt
    curr_taxonomy_params["parent_category"] = parent_category_name

    # run stages: create categories, classify errors, and pupolate 
    taxonomy = await construct_taxonomy(
        error_records=records,
        config=config,
        exp_id=exp_id,
        inference_client=inference_client,
        taxonomy_params=curr_taxonomy_params,
    ) if records else []

    if not taxonomy:
        print("ℹ️ No errors to build taxonomy")
        return

    classification = await classify_errors(
        error_records=records,
        error_taxonomy=taxonomy,
        config=config,
        exp_id=exp_id,
        inference_client=inference_client,
    ) if taxonomy else []

    if not classification:
        print("ℹ️ No taxonomy to run classification")
        return
    populated = await populate_taxonomy(
        error_records=records,
        error_taxonomy=taxonomy,
        error_classify=classification,
        exp_id=exp_id,
        config=config,
        rare_freq=rare_freq,
    )

    return populated


def _calculate_max_clusters(config: Config, num_items: int) -> int:
    fixed_max = config.taxonomy_params["max_num_clusters"]
    return min(fixed_max, math.ceil(num_items * 0.1))


def _get_str_from_params(parent: str, name: str, depth: int) -> str:
    return f"parent={parent}__name={name}__depth={depth}"


def _add_children_to_node(
    items: Union[List[Dict], Dict[str, str]],
    parent_node: TaxonomyNode,
    taxonomy_tree: TaxonomyTree,
    depth: int,
):
    def create_node(name: str, info: Dict) -> TaxonomyNode:
        return TaxonomyNode(
            id=_get_str_from_params(parent=parent_node.name, name=name, depth=depth),
            name=name,
            info=info
        )

    if isinstance(items, list):  # Records case
        for item in items:
            if not item:
                continue
            name = item.get("error_title", "Unknown")
            child = create_node(name, item)
            taxonomy_tree.add_node(parent_node=parent_node, child=child)

    elif isinstance(items, dict):  # Categories case
        for name, description in items.items():
            if not name:
                continue
            info = {"description": description}
            child = create_node(name, info)
            taxonomy_tree.add_node(parent_node=parent_node, child=child)

    print(f"Added {len(parent_node.children)} items under '{parent_node.name}'")


async def _recurse_error_collection(
    records: List[Dict], 
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    parent_node: TaxonomyNode = None, 
    depth: int = 0, 
    max_depth: int = 2,
    taxonomy_tree: TaxonomyTree = None,
    rare_freq: float = None,
):
    print(f"in recurse, records: {len(records)}")

    parent_node_name = parent_node.name if depth > 0 and parent_node.name else None # avoid using the name of the root node or an empty string
    populated = await _run_taxonomy_stages(records, config, exp_id, inference_client, parent_category_name=parent_node_name, rare_freq=rare_freq)
    if not populated:
        return

    categories = {
        record["error_category"]: record["category_description"]
        for record in populated if record["error_category"]
    }

    if len(categories) <= 1:
        _add_children_to_node(records, parent_node, taxonomy_tree, depth)
        return

    _add_children_to_node(categories, parent_node, taxonomy_tree, depth)

    tasks = []
    for category in categories.keys():
        curr_records = [r for r in populated if r["error_category"] == category]
        unique_titles = set(r["error_title"] for r in curr_records)

        if not unique_titles:
            print(f"⚠️ Category '{category}' has no children!")
            continue

        curr_max_clusters = _calculate_max_clusters(config, len(unique_titles))
        category_node_id = _get_str_from_params(parent=parent_node.name, name=category, depth=depth)
        category_node = taxonomy_tree.get_node(id=category_node_id)

        if len(unique_titles) <= 5 or curr_max_clusters <= 1 or depth + 1 > max_depth:
            _add_children_to_node(curr_records, category_node, taxonomy_tree, depth)
        else:
            tasks.append(_recurse_error_collection(
                records=curr_records, 
                config=config,
                exp_id=exp_id,
                inference_client=inference_client,
                parent_node=category_node, 
                depth=depth + 1, 
                max_depth=max_depth,
                taxonomy_tree=taxonomy_tree,
                rare_freq=rare_freq,
            ))

    if tasks:
        await asyncio.gather(*tasks)

@cached("construct_taxonomy_recursively", None)
async def construct_taxonomy_recursively(
    records: List[Dict], 
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    depth: int = 0, 
    max_depth: int = 2,
    rare_freq: float = None,
    cols_to_keep: List[str] = None,
) -> List[Dict]:
    root_name = "LLM Errors"
    root = TaxonomyNode(
        id=_get_str_from_params(parent=None, name=root_name, depth=depth),
        name=root_name,
    )
    taxonomy_tree = TaxonomyTree(root)

    await _recurse_error_collection(
                records=records, 
                config=config,
                exp_id=exp_id,
                inference_client=inference_client,
                parent_node=root, 
                depth=depth, 
                max_depth=max_depth,
                taxonomy_tree=taxonomy_tree,
                rare_freq=rare_freq,
            )
    try:
        with open(os.path.join(config.output_dir, "exp_name=construct_taxonomy_recursively__exp_id=" + exp_id + ".json"), "w") as f:
            json.dump(taxonomy_tree.to_dict(), f, indent=2)
    except Exception as e:
        print(e)

    # taxonomy tree to records
    results = taxonomy_tree.get_leaf_node_dicts_with_ancestry(cols_to_keep=cols_to_keep)

    return results