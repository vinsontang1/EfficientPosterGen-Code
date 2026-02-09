import json
import uuid
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DependencyGraphBuilder:
    def __init__(self, ppl_engine, threshold, min_edge_count: int = 20):
        self.engine = ppl_engine
        self.threshold = threshold
        self.min_edge_count = min_edge_count

    def _flatten_paper_groups(self, paper_data):
        """
        Returns:
            nodes_for_calc: 用于计算的临时列表
            content_mapping: 最终的存储映射 (包含 path)
        """
        nodes_for_calc = []
        content_mapping = {}
        
        for section in paper_data:
            section_title = section.get("section_root_title", "")
            
            for group in section.get("groups", []):
                if not group: continue
                
                node_id = str(uuid.uuid4())
                
                first_para = group[0]
                group_path = first_para.get('path', []) 
                
                
                combined_text = "\n".join([item['text'] for item in group if item.get('text')])
                if not combined_text.strip(): continue

                node_meta = {
                    "id": node_id,
                    "text": combined_text,
                }
                nodes_for_calc.append(node_meta)
                
                content_mapping[node_id] = {
                    "path": group_path,          
                    "section_title": section_title, 
                    "group_content": group       
                }
                
        return nodes_for_calc, content_mapping

    def compute_graph(self, paper_id, paper_data):
        # 获取节点列表和映射
        nodes, content_mapping = self._flatten_paper_groups(paper_data)
        n = len(nodes)
        
        logger.info(f"Building graph for {paper_id} with {n} nodes...")
        
        # 计算 Self-PPL
        self_ppls = np.zeros(n)
        for i in range(n):
            self_ppls[i] = self.engine.compute_conditional_ppl(context="", target=nodes[i]['text'])

        all_candidates = [] 
        
        for i in tqdm(range(n), desc="Target Nodes"):     
            for j in range(n): 
                if i == j: 
                #     print((self_ppls[i] - self.engine.compute_conditional_ppl(
                #     context=nodes[j]['text'], 
                #     target=nodes[i]['text']))/self_ppls[i]
                # )
                    continue 
                
                cond_ppl = self.engine.compute_conditional_ppl(
                    context=nodes[j]['text'], 
                    target=nodes[i]['text']
                )
                
                if self_ppls[i] == 0:
                    score = 0.0
                else:
                    score = (self_ppls[i] - cond_ppl) / self_ppls[i]
                    # print(f"Score between node {j} -> {i}: {score}")
                
                # 将所有边都加入候选池
                all_candidates.append({
                    "source": nodes[j]['id'],  
                    "target": nodes[i]['id'],  
                    "score": round(float(score), 4) 
                })

        final_edges = [edge for edge in all_candidates if edge['score'] > self.threshold]
        
        current_count = len(final_edges)
        logger.info(f"Edges passing threshold {self.threshold}: {current_count}")

        # 检查是否达到最小数量要求
        if current_count < self.min_edge_count:
            needed = self.min_edge_count - current_count
            logger.info(f"Not enough edges ({current_count} < {self.min_edge_count}). Filling {needed} more from candidates.")
            
            # 找出那些未达到阈值的边
            remaining_candidates = [edge for edge in all_candidates if edge['score'] <= self.threshold]
            remaining_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 补齐空缺
            top_ups = remaining_candidates[:needed]
            final_edges.extend(top_ups)
            
        logger.info(f"Final graph has {len(final_edges)} edges.")

        return {
            "edges": final_edges, 
            "id_map": content_mapping 
        }

