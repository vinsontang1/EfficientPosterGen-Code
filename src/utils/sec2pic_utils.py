import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_nonempty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def collect_text_from_node(node: Dict[str, Any], content_pool: Dict[str, str]) -> List[str]:
    """
    收集 node 自身 paragraph_ids 对应的文本（保持顺序）
    """
    texts: List[str] = []
    for pid in node.get("paragraph_ids", []) or []:
        t = content_pool.get(pid)
        if is_nonempty_text(t):
            texts.append(t.strip())
    return texts


def collect_texts_recursively(node: Dict[str, Any], content_pool: Dict[str, str]) -> List[str]:
    """
    收集 node + 所有后代节点的 paragraph_ids 
    """
    texts = collect_text_from_node(node, content_pool)
    for child in node.get("children", []) or []:
        texts.extend(collect_texts_recursively(child, content_pool))
    return texts


def unique_title(title: str, used: Dict[str, int]) -> str:
    """
    避免同名大章节覆盖：重复则追加 (2)、(3)...
    """
    title = title.strip() or "Untitled"
    if title not in used:
        used[title] = 1
        return title
    used[title] += 1
    return f"{title} ({used[title]})"


def build_big_section_map(
    data: Dict[str, Any],
    target_level: int = 1,
    joiner: str = "\n\n",
    include_empty: bool = False,
) -> Dict[str, str]:
    structure = data.get("structure")
    content = data.get("content")

    if not isinstance(structure, list):
        raise ValueError("Input JSON missing or invalid 'structure' (expected a list)")
    if not isinstance(content, dict):
        raise ValueError("Input JSON missing or invalid 'content' (expected a dict)")
    out: Dict[str, str] = {}
    used_titles: Dict[str, int] = {}

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes or []:
            level = node.get("level", None)
            if level == target_level:
                title = (node.get("title") or "").strip() or "Untitled"
                texts = collect_texts_recursively(node, content)
                merged = joiner.join([t for t in texts if t.strip()])

                if merged or include_empty:
                    key = unique_title(title, used_titles)
                    out[key] = merged
                walk(node.get("children", []) or [])
            else:
                walk(node.get("children", []) or [])

    walk(structure)
    return out


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

def to_abs_img_path(img_path: str, root_dir: Path) -> str:
    p = Path(img_path)
    if p.is_absolute():
        return str(p)
    return str((root_dir / p).resolve())


def is_nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) > 0
    return True


def alias_from_index(i: int) -> str:
    """A,B,...,Z,AA,AB...  i is 0-based."""
    s = ""
    n = i + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_int(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except Exception:
        return None


# -----------------------------
# Selection logic
# -----------------------------

@dataclass(frozen=True)
class SelectionConfig:
    max_page_idx: int = 8


@dataclass(frozen=True)
class PickedCore:
    img_path_abs: str
    payload_key: str       # "table_body" or "caption"
    payload: Any
    item_type: str         # "table" or "image"


def choose_payload(item: Dict[str, Any]) -> Optional[Tuple[str, Any]]:

    tbl = item.get("table_body")

    cap_candidates = [
        item.get("img_caption"),
        item.get("image_caption"),
    ]
    cap = None
    for c in cap_candidates:
        if is_nonempty(c):
            cap = c
            break

    tbl_ok = is_nonempty(tbl)
    cap_ok = is_nonempty(cap)

    if not tbl_ok and not cap_ok:
        return None
    if tbl_ok:
        return "table_body", tbl
    return "caption", cap


def infer_item_type(raw_item: Dict[str, Any]) -> str:
    
    return "table" if is_nonempty(raw_item.get("table_body")) else "image"


def should_keep(item: Dict[str, Any], cfg: SelectionConfig, root_dir: Path) -> Optional[PickedCore]:
    # Step 1: img_path 非空
    img_path = item.get("img_path")
    if not is_nonempty(img_path):
        return None

    img_abs = to_abs_img_path(str(img_path), root_dir)

    if "page_idx" not in item:
        return None
    page_idx = parse_int(item.get("page_idx"))
    if page_idx is None or page_idx > cfg.max_page_idx:
        return None

    chosen = choose_payload(item)
    if chosen is None:
        return None
    payload_key, payload = chosen

    item_type = infer_item_type(item)

    return PickedCore(
        img_path_abs=img_abs,
        payload_key=payload_key,
        payload=payload,
        item_type=item_type,
    )


# -----------------------------
# Scanning & processing
# -----------------------------

def find_target_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*content_list.json"))


def pick_from_file(path: Path, root_dir: Path, cfg: SelectionConfig) -> List[PickedCore]:
    data = load_json(path)
    if not isinstance(data, list):
        return []

    out: List[PickedCore] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        picked = should_keep(raw, cfg, root_dir)
        if picked:
            out.append(picked)
    return out


def assign_aliases(items: List[PickedCore]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    返回：
    - alias_map: alias -> img_path_abs (绝对路径)
    - items_min: [{"alias","type","payload_key","payload"}, ...]
    """
    # 保持首次出现顺序去重（按绝对路径去重）
    path_order: List[str] = []
    seen = set()
    for it in items:
        if it.img_path_abs not in seen:
            seen.add(it.img_path_abs)
            path_order.append(it.img_path_abs)

    alias_map: Dict[str, str] = {}
    path_to_alias: Dict[str, str] = {}
    for i, p in enumerate(path_order):
        a = alias_from_index(i)
        alias_map[a] = p
        path_to_alias[p] = a

    items_min = [
        {
            "alias": path_to_alias[it.img_path_abs],
            "type": it.item_type,            # "image" or "table"
            "payload_key": it.payload_key,   # "caption" or "table_body"
            "payload": it.payload,
        }
        for it in items
    ]
    return alias_map, items_min


def run(root_dir: Path, cfg: SelectionConfig) -> Dict[str, Any]:
    files = find_target_files(root_dir)

    all_items: List[PickedCore] = []
    for fp in files:
        all_items.extend(pick_from_file(fp, root_dir, cfg))

    alias_map, items_min = assign_aliases(all_items)

    return {
        "stats": {
            "file_count": len(files),
            "picked_total": len(items_min),
        },
        "alias_map": alias_map,  
        "items": items_min,      
    }
