from typing import Dict, List, Optional
import json

class TaxonomyNode:
    def __init__(self, id: str, name: str, info: Optional[Dict] = None, parent: "TaxonomyNode" = None):
        self.id = id
        self.name = name
        self.info = info or {}
        self.children: List["TaxonomyNode"] = []
        self.parent = parent

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "info": {k: v for k, v in self.info.items() if k in {
                    "description", "prompt", "model", "dataset", "example_id", "input_text",
                    "candidate_answers", "output_text", "score", "judge_model", "judge_response", "error_summary", "error_title"}},
            "children": [child.to_dict() for child in self.children],
            "parent": self.parent.id if self.parent else ""       
        }


class TaxonomyTree:
    def __init__(self, root: TaxonomyNode):
        self.root = root
        self._lookup: Dict[str, TaxonomyNode] = {root.id: root}

    def add_node(self, parent_node: TaxonomyNode, child: TaxonomyNode) -> bool:
        if self._lookup.get(child.id):
            return False
        parent_node.children.append(child)
        self._lookup[child.id] = child
        return True

    def get_node(self, id: str) -> Optional[TaxonomyNode]:
        return self._lookup.get(id)

    def to_dict(self) -> Dict:
        return self.root.to_dict()
    
    def get_leaf_node_dicts_with_ancestry(self, cols_to_keep: List[str] = None) -> List[Dict]:
        result = []

        def traverse(node: TaxonomyNode, path: List[TaxonomyNode]):
            path.append(node)

            if not node.children:  # Leaf node
                fields = ["dataset", "example_id", "model", "input_text", "output_text", "score", "judge_model", "judge_response", "error_title", "error_summary", "prompt"]
                if cols_to_keep is not None:
                    fields = fields + cols_to_keep
                leaf_info = {field: node.info[field] for field in fields if field in node.info}

                for i, ancestor in enumerate(path):
                    if i == len(path) - 1: # no need to add the leaf node
                        break 
                    cat_info = ancestor.info
                    cat_info.update({"name": ancestor.name})
                    leaf_info[f"category_depth_{i}"] = json.dumps(cat_info)
                result.append(leaf_info)
            else:
                for child in node.children:
                    traverse(child, path.copy())

        traverse(self.root, [])
        return result
