import json
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional
from src.utils.sec2pic_utils import build_big_section_map, run
from dataclasses import dataclass
from src.utils.bullet_points import Agent
from src.utils.agent_utils import ModelFactory
import json
from typing import Any, Dict, List, Set, Tuple


def normalize_section_keys_spaces_to_underscore(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    将 section 标题 key 中的空格替换为下划线：
    "1. Introduction" -> "1._Introduction"
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in results.items():
        new_k = k.replace(" ", "_")
        out[new_k] = v
    return out

@dataclass(frozen=True)
class SelectionConfig:
    max_page_idx: int = 8

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output not JSON object: {text[:200]}")
    return json.loads(text[start:end + 1])


def _validate_and_normalize_mapping(
    raw: Dict[str, Any],
    section_titles: Set[str],
    valid_aliases: Set[str],
) -> Dict[str, Dict[str, str]]:
    """
    输出格式固定为:
    {
      section_title: {"type": "image"|"table", "alias": "<A>", "reason": "..."}
    }
    """
    if not isinstance(raw, dict):
        raise ValueError("LLM output root must be a JSON object")

    used_aliases: Set[str] = set()
    out: Dict[str, Dict[str, str]] = {}

    for sec_title, v in raw.items():
        # section title 必须来自输入
        if sec_title not in section_titles:
            raise ValueError(f"Invalid section title from LLM: {sec_title}")

        if not isinstance(v, dict):
            raise ValueError(f"Section '{sec_title}' value must be an object")

        has_image = "image" in v
        has_table = "table" in v
        if has_image == has_table:  
            raise ValueError(f"Section '{sec_title}' must contain exactly one of 'image' or 'table'")

        reason = v.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Section '{sec_title}' must contain non-empty 'reason'")

        if has_image:
            alias = v.get("image")
            typ = "image"
        else:
            alias = v.get("table")
            typ = "table"

        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"Section '{sec_title}' alias must be a non-empty string")

        alias = alias.strip()

        if alias not in valid_aliases:
            raise ValueError(f"Section '{sec_title}' uses invalid alias '{alias}' not in candidates")

        if alias in used_aliases:
            raise ValueError(f"Alias '{alias}' reused (not allowed)")

        used_aliases.add(alias)

        out[sec_title] = {"type": typ, "alias": alias, "reason": reason.strip()}

    return out

def _parse_best_alias(llm_text: str) -> str:
    """
    解析模型输出中的 {"best_alias":"A"}，允许前后有杂文本。
    """
    llm_text = (llm_text or "").strip()
    start = llm_text.find("{")
    end = llm_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output not JSON: {llm_text[:200]}")
    obj = json.loads(llm_text[start:end + 1])
    best = obj.get("best_alias")
    if not best or not isinstance(best, str):
        raise ValueError(f"Missing best_alias in LLM output: {obj}")
    return best.strip()


def _default_prompt_builder(section_map: Dict[str, str], items: List[Dict[str, Any]]) -> str:
    """
    关键：必须把 section_map/items 真正注入 prompt（json.dumps），
    不要用 {{section_map}} 这种占位符。
    """
    json_content = [{"title": k, "description": v} for k, v in section_map.items()]
    allowed_titles = list(section_map.keys())
    return f"""
You are an expert assistant tasked with matching visual assets (images/tables) to the most relevant TOP-LEVEL sections.

You will be given:
1) json_content: a JSON array of top-level sections, each with:
   - "title"
   - "description" (merged text)
2) items: a list of candidate assets. Each item has:
   - "alias": a short ID like "A", "B", ...
   - "type": either "image" or "table"
   - "payload_key": "caption" or "table_body"
   - "payload": the caption text or table body content

Task:
Produce ONE JSON object mapping from section title to exactly ONE assigned asset OR no assignment.
- If assigned, value MUST be either:
  - {{"image":"<alias>","reason":"<one short sentence>"}}
  - {{"table":"<alias>","reason":"<one short sentence>"}}
- STRICT: Do NOT include both "image" and "table" for the same section.
- If no asset fits a section, omit that section from the output JSON.
- STRICT: Each alias can be assigned to AT MOST one section (no reuse).

Output rules:
- Only use section titles that appear in json_content.
- Only use aliases that appear in items.
- Output MUST be valid JSON and contain nothing except the JSON.

Input:
json_content = {json.dumps(json_content, ensure_ascii=False)}
items = {json.dumps(items, ensure_ascii=False)}

ALLOWED_SECTION_TITLES = {json.dumps(allowed_titles, ensure_ascii=False)} (STRICT)

Output schema:
{{
  "<Section Title>": {{
    "image": "<alias>" OR "table": "<alias>",
    "reason": "<one short sentence>"
  }},
  ...
}}
""".strip()

def add_path_by_alias(
    results: Dict[str, Dict[str, str]],
    alias_map: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """
    results: {"Methods": {"type":"image","alias":"A","reason":"..."}}
    alias_map: {"A": "/abs/...jpg", ...}
    返回：每项增加 path
    """
    out = {}
    for sec, v in results.items():
        alias = v.get("alias")
        path = alias_map.get(alias, "")
        out[sec] = {**v, "path": path}
    return out

def match_sections_to_pics(
    agent: Agent,
    content_dir: str,
    structure_json_path: str,
    out_path: str = "section2alias.json",
    target_level: int = 1,
    max_page_idx: int = 8,
    joiner: str = "\n\n",
    prompt_builder: Optional[Callable[[Dict[str, str], List[Dict[str, Any]]], str]] = None,
    max_retries: int = 3,   # 新增：最多重试次数
) -> Dict[str, Dict[str, Any]]:

    content_dir_p = Path(content_dir).expanduser().resolve()
    struct_p = Path(structure_json_path).expanduser().resolve()
    out_p = Path(out_path).expanduser().resolve()

    if not content_dir_p.exists() or not content_dir_p.is_dir():
        raise RuntimeError(f"content_dir does not exist or is not a directory: {content_dir_p}")
    if not struct_p.exists():
        raise RuntimeError(f"structure_json_path does not exist: {struct_p}")

    structure_data = _load_json(struct_p)
    section_map: Dict[str, str] = build_big_section_map(
        structure_data,
        target_level=target_level,
        joiner=joiner,
        include_empty=False,
    )

    cfg = SelectionConfig(max_page_idx=max_page_idx)
    run_out = run(content_dir_p, cfg)
    items: List[Dict[str, Any]] = run_out.get("items", [])
    alias_map: Dict[str, str] = run_out.get("alias_map", {})
    
    token_usage_total = {
        "input_text": 0,
        "input_image": 0,
        "input_total": 0,
        "output": 0,
    }

    if not items:
        _save_json({}, out_p)
        return {"__token_usage__": token_usage_total}

    section_titles = set(section_map.keys())
    valid_aliases = {
        it.get("alias")
        for it in items
        if isinstance(it, dict) and isinstance(it.get("alias"), str) and it.get("alias").strip()
    }

    # 3) prompt + one-shot LLM call
    pb = prompt_builder or _default_prompt_builder
    base_prompt = pb(section_map, items)

    last_err: Optional[Exception] = None
    last_raw_text: str = ""

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            prompt = base_prompt
        else:
            prompt = (
                base_prompt
                + "\n\n"
                + "FORMAT_ERROR_DETECTED:\n"
                + f"{str(last_err)}\n\n"
                + "You MUST fix the output. Return ONLY a valid JSON object that satisfies ALL constraints. "
                  "Do NOT add any commentary. Do NOT wrap in markdown.\n"
            )

        resp = agent.step(prompt, images_base64=None)
        
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if isinstance(usage, dict):
            token_usage_total["input_text"] += int(usage.get("input_text", 0) or 0)
            token_usage_total["input_image"] += int(usage.get("input_image", 0) or 0)
            token_usage_total["input_total"] += int(usage.get("input_total", 0) or 0)
            token_usage_total["output"] += int(usage.get("output", 0) or 0)
        
        last_raw_text = resp.get("content", "") or ""

        try:
            raw_obj = _extract_json_object(last_raw_text)
            results = _validate_and_normalize_mapping(raw_obj, section_titles, valid_aliases)
            results_with_path = add_path_by_alias(results, alias_map)

            results_with_path_norm = normalize_section_keys_spaces_to_underscore(results_with_path)

            _save_json(results_with_path_norm, out_p)
            ret = dict(results_with_path_norm)
            ret["__token_usage__"] = token_usage_total
            return ret
        
        except Exception as e:
            last_err = e
            print(f"[Attempt {attempt}] validation failed: {e}")
    _save_json({}, out_p)
    return {"__token_usage__": token_usage_total} 



    