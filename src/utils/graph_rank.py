import networkx as nx
from typing import List, Dict
import logging
from config import config
logger = logging.getLogger(__name__)

class GraphRankSelector:
    def __init__(self, alpha: float = 0.8, beta: float = 0.6, min_count: int = 10, k: int = 0.95):
        self.alpha = alpha
        self.beta = beta
        self.min_count = min_count
        self.k = k
        self.min_section_count = config['selection']['min_section_count']

    def _calculate_pagerank(self, edges: List[Dict], all_node_ids: List[str]) -> Dict[str, float]:
        G = nx.DiGraph()
        G.add_nodes_from(all_node_ids)

        for edge in edges:
            G.add_edge(edge['source'], edge['target'], weight=edge['score'])

        try:
            return nx.pagerank(G, weight='weight', alpha=self.k)
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            return {nid: 0.0 for nid in all_node_ids}

    def _calculate_lca_depth(self, path_a: List[str], path_b: List[str]) -> int:
        lca_depth = 0
        for a, b in zip(path_a, path_b):
            if a == b:
                lca_depth += 1
            else:
                break
        return len(path_a) + len(path_b) - 2 * lca_depth

    def select_nodes(self, graph_data: Dict, threshold: float = 0.0) -> List[Dict]:
        edges = graph_data.get("edges", [])
        id_map = graph_data.get("id_map", {})
        if not id_map:
            return []

        all_node_ids_ = set()
        for edge in edges:
            all_node_ids_.add(edge["source"])
            all_node_ids_.add(edge["target"])
        all_node_ids = [uid for uid in all_node_ids_ if uid in id_map]
        if not all_node_ids:
            return []

        if len(all_node_ids) <= self.min_count:
            pr_scores = self._calculate_pagerank(edges, all_node_ids)
            N = len(all_node_ids)
            if N > 0:
                pr_scores = {k: v * N for k, v in pr_scores.items()}

            results = []
            for uid in all_node_ids:
                node_info = id_map[uid]
                results.append({
                    "unique_id": uid,
                    "score": pr_scores.get(uid, 0.0),
                    "pagerank": pr_scores.get(uid, 0.0),
                    "section_title": node_info.get("section_title", "UnknownSection"),
                    "context": "\n".join([item.get("text", "") for item in node_info.get("group_content", [])])
                })
            print("+++++++++++++++++++++++++++++++++++++++++++")
            print(len(results))
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Total nodes ({len(results)}) <= target ({self.min_count}), returning all.")
            return results

        pr_scores = self._calculate_pagerank(edges, all_node_ids)
        N = len(all_node_ids)
        if N > 0:
            pr_scores = {k: v * N for k, v in pr_scores.items()}

        selected_nodes: List[Dict] = []
        candidate_ids = set(all_node_ids)
        current_max_penalty = {uid: 0.0 for uid in candidate_ids}

        section_to_nodes = {}
        for uid in all_node_ids:
            sec = id_map[uid].get("section_title", "UnknownSection")
            section_to_nodes.setdefault(sec, []).append(uid)

        required_per_section = {
            sec: (self.min_section_count if len(uids) >= self.min_section_count else 0)
            for sec, uids in section_to_nodes.items()
        }
        selected_count_by_section = {sec: 0 for sec in section_to_nodes.keys()}

        def _score(uid: str) -> float:
            r_c = pr_scores.get(uid, 0.0)
            penalty = current_max_penalty.get(uid, 0.0)
            return self.alpha * r_c - (1 - self.alpha) * penalty

        def _pick_best(eligible_ids: set):
            best_uid, best_s = None, -1e18
            for uid in eligible_ids:
                s = _score(uid)
                if s > best_s:
                    best_uid, best_s = uid, s
            return best_uid, best_s

        def _select_one(uid: str, s: float):
            node_info = id_map[uid]
            node_path = node_info.get("path", [])

            selected_nodes.append({
                "unique_id": uid,
                "score": s,
                "pagerank": pr_scores.get(uid, 0.0),
                "section_title": node_info.get("section_title", "UnknownSection"),
                "context": node_info.get("group_content", [{}])[0].get("text", "")
            })

            sec = node_info.get("section_title", "UnknownSection")
            selected_count_by_section[sec] = selected_count_by_section.get(sec, 0) + 1

            candidate_ids.remove(uid)

            for other_uid in candidate_ids:
                other_path = id_map[other_uid].get("path", [])
                lca = self._calculate_lca_depth(node_path, other_path)
                sim = self.beta ** lca
                if sim > current_max_penalty[other_uid]:
                    current_max_penalty[other_uid] = sim

        while candidate_ids:
            need_sections = {
                sec for sec, req in required_per_section.items()
                if req > 0 and selected_count_by_section.get(sec, 0) < req
            }
            if not need_sections:
                break

            eligible = {uid for uid in candidate_ids
                        if id_map[uid].get("section_title", "UnknownSection") in need_sections}

            if not eligible:
                break

            best_uid, best_s = _pick_best(eligible)
            if best_uid is None:
                break
            _select_one(best_uid, best_s)

        while candidate_ids:
            best_uid, best_s = _pick_best(candidate_ids)
            if best_uid is None:
                break

            if best_s >= threshold:
                _select_one(best_uid, best_s)
                continue

            if len(selected_nodes) < self.min_count:
                _select_one(best_uid, best_s)
                continue
            break

        logger.info(f"Selected {len(selected_nodes)} nodes.")
        return selected_nodes







        