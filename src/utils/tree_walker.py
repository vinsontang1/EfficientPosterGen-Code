from typing import List, Dict, Any

class TreeWalker:
    @staticmethod
    def extract_sequence_from_section(
        section_node: Dict[str, Any], 
        content_map: Dict[str, str], 
        parent_path: List[str]
    ) -> List[Dict[str, Any]]:
        
        sequence = []
        
        def _dfs(node: Dict[str, Any], ancestor_path: List[str]):
            node_id = node.get("id")
            if not node_id:
                current_node_path = ancestor_path
            else:
                current_node_path = ancestor_path + [node_id]

            current_p_ids = node.get("paragraph_ids", [])
            current_depth = node.get("level", 1) 
            
            for pid in current_p_ids:
                if pid in content_map:
                    text = content_map[pid].strip()
                    if text: 
                        sequence.append({
                            "id": pid,
                            "text": text,
                            "depth": current_depth, 
                            "section_title": node.get("title", ""),
                            "path": list(current_node_path) 
                        })
            
            children = node.get("children", [])
            for child in children:
                _dfs(child, current_node_path)
        
        _dfs(section_node, parent_path)
        return sequence