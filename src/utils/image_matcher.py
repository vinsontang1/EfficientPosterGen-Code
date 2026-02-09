import os
import json
import re
import io
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class AssetMatcher:
    def __init__(self, input_root, paper_id):
        self.input_root = Path(input_root)
        self.paper_id = paper_id
        self.paper_dir = self.input_root / paper_id
        
        # 1. 自动寻找 MD 文件
        self.md_path = self.paper_dir / "full.md"
        if not self.md_path.exists():
            md_files = list(self.paper_dir.glob("*.md"))
            if md_files:
                self.md_path = md_files[0]
        
        self.img_dir = self.paper_dir / "images"
        self.img_dir.mkdir(exist_ok=True)
        
        # 2. 正则表达式
        self.md_img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
        self.html_table_pattern = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
        
        # 标签正则 (V8版，抗干扰)
        self.label_pattern = re.compile(
            r'(?:[*_]*)(Figure|Fig\.|Fig|Table|Tab\.|Tab)(?:[*_]*)[\s\xa0]+(\d+)(?:[*_]*)\s*([:：\.]?)', 
            re.IGNORECASE
        )

        # [新增] 忽略区域的标题正则
        # 匹配: 行首 + 任意个# + 可能的编号 + 关键词 (Appendix, Reference, Acknowledge...)
        self.ignore_section_pattern = re.compile(
            r'^#+\s*(?:[\d\.]*\s*)?(?:Appendices?|Appendix|References?|Bibliography|Acknowledgements?|Acknowledgments?)\s*$',
            re.IGNORECASE | re.MULTILINE
        )

        # 3. 加载 Mineru 元数据
        self.html_to_img_map, self.ordered_table_imgs = self._load_mineru_metadata()

    def _normalize_html(self, html_str):
        if not html_str: return ""
        return "".join(html_str.split())

    def _load_mineru_metadata(self):
        html_map = {}
        ordered_imgs = []
        candidates = list(self.paper_dir.glob("*content_list.json")) + \
                     list(self.paper_dir.glob("*middle.json")) + \
                     list(self.paper_dir.glob("*.json"))
        target_json = None
        for f in candidates:
            if "selection" in f.name or "summary" in f.name: continue
            if f.stat().st_size > 1000: 
                target_json = f; break
        
        if target_json:
            try:
                with open(target_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else data.get('content_list', [])
                    for item in items:
                        if item.get('type') == 'table' and item.get('img_path'):
                            rel_path = item['img_path']
                            abs_path = (self.paper_dir / rel_path).resolve()
                            if abs_path.exists():
                                img_path_str = str(abs_path)
                                ordered_imgs.append(img_path_str)
                                if item.get('table_body'):
                                    fingerprint = self._normalize_html(item['table_body'])
                                    html_map[fingerprint] = img_path_str
            except: pass
        return html_map, ordered_imgs

    def _render_html_table(self, html_content, table_id):
        try:
            dfs = pd.read_html(io.StringIO(html_content))
            if not dfs: return None
            df = dfs[0]
            fig, ax = plt.subplots(figsize=(8, len(df)*0.5 + 1))
            ax.axis('tight'); ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
            save_path = self.img_dir / f"rendered_{table_id}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return str(save_path)
        except: return None

    def _get_candidate(self, content, start_pos, end_pos, direction="after"):
        WINDOW_SIZE = 3000
        if direction == "before":
            search_start = max(0, start_pos - WINDOW_SIZE)
            search_end = start_pos
            context = content[search_start:search_end]
        else:
            search_start = end_pos
            search_end = min(len(content), end_pos + WINDOW_SIZE)
            context = content[search_start:search_end]
            
        if not context.strip(): return None

        matches = list(self.label_pattern.finditer(context))
        if not matches: return None
            
        if direction == "before":
            target_match = matches[-1] 
            distance = len(context) - target_match.end()
        else:
            target_match = matches[0]
            distance = target_match.start()
            
        type_str, num_str, suffix = target_match.groups()
        key = self._normalize_key(type_str, num_str)
        has_strong_indicator = bool(suffix and suffix.strip() in [':', '：', '.'])
        
        match_start = target_match.start()
        raw_tail = context[match_start:]
        first_block = raw_tail.split('\n\n')[0]
        
        sub_matches = list(self.label_pattern.finditer(first_block[10:]))
        if sub_matches:
            cut_pos = sub_matches[0].start() + 10
            first_block = first_block[:cut_pos]

        clean_caption = first_block.replace('\n', ' ').strip()
        
        return {
            "key": key,
            "num": int(num_str), 
            "caption": clean_caption,
            "is_strong": has_strong_indicator,
            "distance": distance
        }

    def _normalize_key(self, type_str, num_str):
        t = type_str.lower()
        prefix = 'table' if 'tab' in t else 'figure'
        return f"{prefix}_{num_str}"

    def _detect_table_caption(self, content, match):
        cand_before = self._get_candidate(content, match.start(), match.end(), "before")
        cand_after  = self._get_candidate(content, match.start(), match.end(), "after")
        
        winner = None
        if cand_before and not cand_after: winner = cand_before
        elif cand_after and not cand_before: winner = cand_after
        elif cand_before and cand_after:
            if cand_before['is_strong'] and not cand_after['is_strong']: winner = cand_before
            elif cand_after['is_strong'] and not cand_before['is_strong']: winner = cand_after
            else:
                if cand_after['distance'] < cand_before['distance']: winner = cand_after
                else: winner = cand_before
        return winner

    def _fix_missing_table_keys(self, found_tables):
        """插值补全漏识别的表格"""
        for i in range(1, len(found_tables) - 1):
            curr = found_tables[i]
            prev = found_tables[i-1]
            next_item = found_tables[i+1]

            if curr['key'] is None:
                if (prev['key'] and 'table' in prev['key']) and \
                   (next_item['key'] and 'table' in next_item['key']):
                    
                    diff = next_item['num'] - prev['num']
                    if diff == 2:
                        missing_num = prev['num'] + 1
                        new_key = f"table_{missing_num}"
                        curr['key'] = new_key
                        curr['num'] = missing_num
                        curr['caption'] = f"Table {missing_num}: (Auto-detected)"
                        print(f"    [Auto-Fix] Interpolated: {prev['key']} ... [Table {missing_num}] ... {next_item['key']}")

    def _truncate_content(self, content):
        """
        [新增] 截断 Appendix/Reference 之后的内容
        """
        min_pos = len(content)
        found_cutoff = False
        
        # 查找所有匹配的“结束章节”标题
        for match in self.ignore_section_pattern.finditer(content):
            # 我们取最靠前的一个位置截断
            if match.start() < min_pos:
                min_pos = match.start()
                found_cutoff = True
                print(f"  [Truncate] Cutoff detected at section: '{match.group(0).strip()}' (pos: {min_pos})")
        
        if found_cutoff:
            return content[:min_pos]
        return content

    def _build_asset_index(self):
        asset_index = {}
        if not self.md_path.exists():
            return {}

        print(f"  > Indexing assets from: {self.md_path.name}")
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = self._truncate_content(content)

        # === 全局 caption 扫描 ===
        all_table_captions = self._collect_all_table_captions(content)

        # === Figure（原逻辑）===
        for match in self.md_img_pattern.finditer(content):
            rel_path = match.group(2).strip()
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]

            winner = self._detect_table_caption(content, match)
            if winner:
                key = winner['key']
                p = (self.paper_dir / rel_path).resolve()
                if p.exists():
                    asset_index[key] = {
                        "path": str(p),
                        "caption": winner['caption']
                    }

        # === Table（核心修复）===
        found_tables = []
        table_iter = list(self.html_table_pattern.finditer(content))

        for idx, match in enumerate(table_iter):
            table_pos = match.start()

            cap = self._resolve_table_caption(table_pos, all_table_captions)

            found_tables.append({
                "idx": idx,
                "html": match.group(0),
                "key": cap["key"] if cap else None,
                "num": cap["num"] if cap else None,
                "caption": cap["caption"] if cap else None
            })

        # === 插值补全（保留你原逻辑）===
        self._fix_missing_table_keys(found_tables)

        # === 资源绑定 ===
        for item in found_tables:
            key = item['key']
            if not key:
                continue

            fingerprint = self._normalize_html(item['html'])
            final_img_path = None

            if fingerprint in self.html_to_img_map:
                final_img_path = self.html_to_img_map[fingerprint]
            elif item['idx'] < len(self.ordered_table_imgs):
                final_img_path = self.ordered_table_imgs[item['idx']]
            else:
                final_img_path = self._render_html_table(item['html'], key)

            if final_img_path:
                asset_index[key] = {
                    "path": final_img_path,
                    "caption": item['caption']
                }

        print(f"  > Indexed {len(asset_index)} assets (final).")
        # print(asset_index)
        return asset_index

    
    def _collect_all_table_captions(self, content):
        captions = []
        for m in self.label_pattern.finditer(content):
            type_str, num_str, suffix = m.groups()
            if 'tab' not in type_str.lower():
                continue

            start = m.start()
            tail = content[start:start + 600]
            block = tail.split('\n\n')[0].replace('\n', ' ').strip()

            captions.append({
                "key": self._normalize_key(type_str, num_str),
                "num": int(num_str),
                "caption": block,
                "pos": start
            })
        return captions
    
    def _resolve_table_caption(self, table_pos, all_captions):
        before = None
        after = None

        for cap in all_captions:
            if cap["pos"] < table_pos:
                before = cap
            elif cap["pos"] > table_pos and after is None:
                after = cap

        if before and not after:
            return before

        if after and not before:
            return after

        if before and after:
            dist_before = table_pos - before["pos"]
            dist_after = after["pos"] - table_pos

            # 距离明显更近
            if abs(dist_before - dist_after) > 300:
                return before if dist_before < dist_after else after

            # 距离接近：优先 post-caption（学术规范）
            return after

        return None

    def process(self, selection_json_path):
        asset_db = self._build_asset_index()
        if not asset_db:
            return {}, {}, {}

        try:
            with open(selection_json_path, 'r') as f:
                selected_nodes = json.load(f).get("selected_nodes", [])
        except:
            return {}, {}, {}

        # 分离：图 / 表
        poster_images = {}  # id -> {"image_path", "caption"}
        poster_tables = {}  # id -> {"table_path", "caption"}
        poster_figures = {} # section -> {"image", "table", "image_ids", "table_ids", ...}

        img_id_counter = 0
        table_id_counter = 0

        # 分离映射，避免 figure_1 和 table_1 共用同一个 id 空间（更清晰、也更好调试）
        figkey_to_imgid = {}
        tabkey_to_tabid = {}

        for node in selected_nodes:
            section_title = node.get("section_title", "Unknown")
            refs = self.label_pattern.findall(node.get("context", ""))

            section_image_ids = []
            section_table_ids = []

            for type_str, num_str, suffix in refs:
                key = self._normalize_key(type_str, num_str)
                if key not in asset_db:
                    continue

                # Figure
                if key.startswith("figure_"):
                    if key not in figkey_to_imgid:
                        curr_id = str(img_id_counter)
                        figkey_to_imgid[key] = curr_id
                        poster_images[curr_id] = {
                            "image_path": asset_db[key]["path"],
                            "caption": asset_db[key]["caption"]
                        }
                        img_id_counter += 1

                    mapped_id = figkey_to_imgid[key]
                    if mapped_id not in section_image_ids:
                        section_image_ids.append(mapped_id)

                # Table
                elif key.startswith("table_"):
                    if key not in tabkey_to_tabid:
                        curr_id = str(table_id_counter)
                        tabkey_to_tabid[key] = curr_id
                        poster_tables[curr_id] = {
                            "table_path": asset_db[key]["path"],
                            "caption": asset_db[key]["caption"]
                        }
                        table_id_counter += 1

                    mapped_id = tabkey_to_tabid[key]
                    if mapped_id not in section_table_ids:
                        section_table_ids.append(mapped_id)

            # 写回 section 映射
            if section_image_ids or section_table_ids:
                if section_title not in poster_figures:
                    poster_figures[section_title] = {
                        # 主图/主表：先用第一个（你后面可以替换成“最相关”的选择策略）
                        "image": section_image_ids[0] if section_image_ids else None,
                        "table": section_table_ids[0] if section_table_ids else None,
                        "image_ids": section_image_ids,
                        "table_ids": section_table_ids,

                        # 可选：给一个主 caption（优先图，其次表）
                        "caption": (
                            poster_images[section_image_ids[0]]["caption"]
                            if section_image_ids else
                            poster_tables[section_table_ids[0]]["caption"]
                        )
                    }
                else:
                    # 合并去重
                    exist_imgs = poster_figures[section_title].get("image_ids", [])
                    for i in section_image_ids:
                        if i not in exist_imgs:
                            exist_imgs.append(i)
                    poster_figures[section_title]["image_ids"] = exist_imgs

                    exist_tabs = poster_figures[section_title].get("table_ids", [])
                    for t in section_table_ids:
                        if t not in exist_tabs:
                            exist_tabs.append(t)
                    poster_figures[section_title]["table_ids"] = exist_tabs

                    # 如果之前没有主图/主表，补上
                    if poster_figures[section_title].get("image") is None and exist_imgs:
                        poster_figures[section_title]["image"] = exist_imgs[0]
                    if poster_figures[section_title].get("table") is None and exist_tabs:
                        poster_figures[section_title]["table"] = exist_tabs[0]

                    # caption：如果原来为空，补一个
                    if not poster_figures[section_title].get("caption"):
                        if poster_figures[section_title].get("image") is not None:
                            poster_figures[section_title]["caption"] = poster_images[poster_figures[section_title]["image"]]["caption"]
                        elif poster_figures[section_title].get("table") is not None:
                            poster_figures[section_title]["caption"] = poster_tables[poster_figures[section_title]["table"]]["caption"]

        # 返回三份：images / tables / mapping
        return poster_images, poster_tables, poster_figures


def get_matched_assets(input_root: str, output_root: str, paper_id: str):
    matcher = AssetMatcher(input_root, paper_id)

    selection_json = Path(output_root) / "04_selection" / f"{paper_id}.json"
    return matcher.process(selection_json)

if __name__ == "__main__":
    # 测试代码
    IN_DIR = "paper2poster/parsed_papers"
    OUT_DIR = "./temp1"
    PID = "23_CROP_Certifying_Robust_Policies_for_Reinforcement_Learning_through_Functional_Sm" # 替换为实际存在的 ID
    
    imgs_path = os.path.join(OUT_DIR, f'{PID}_images.json')
    tabs_path = os.path.join(OUT_DIR, f'{PID}_tables.json')
    mapping_path = os.path.join(OUT_DIR, f'{PID}_mapping.json')
    
    imgs, tabs, mapping = get_matched_assets(IN_DIR, OUT_DIR, PID)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(imgs_path, 'w', encoding='utf-8') as f:
        json.dump(imgs, f, indent=4, ensure_ascii=False)
    print(f"Saved: {imgs_path}")

    with open(tabs_path, 'w', encoding='utf-8') as f:
        json.dump(tabs, f, indent=4, ensure_ascii=False)
    print(f"Saved: {tabs_path}")

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    print(f"Saved: {mapping_path}")
    
    print("\n=== Images Dict (Assets) ===")
    print(json.dumps(imgs, indent=2))
    print("\n=== tables Dict (Layout Mapping) ===")
    print(json.dumps(tabs, indent=2))
    print("\n=== mapping Dict (Layout Mapping) ===")
    print(json.dumps(mapping, indent=2))