import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

_ABSTRACT_HEADING_RE = re.compile(r"^\s*#+\s*abstract\b", re.IGNORECASE)

def make_poster_title_from_id(paper_id: str) -> str:
    """
    Example:
      '23_CROP_Certifying_Robust_Policies_for_Reinforcement_Learning' ->
      'CROP Certifying Robust Policies for Reinforcement Learning'
    """
    s = re.sub(r"^\d+_+", "", paper_id)   # drop leading digits + underscores
    s = s.replace("_", " ").strip()
    return s

def load_md_header_block(md_path: Path, fallback_lines: int = 10) -> str:
    """
    Read md, return content strictly before '# Abstract' (any # level).
    If cannot find abstract heading, return first fallback_lines lines (raw order).
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # find first abstract heading line index
    abs_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if _ABSTRACT_HEADING_RE.match(ln):
            abs_idx = i
            break

    if abs_idx is not None and abs_idx > 0:
        block = "\n".join(lines[:abs_idx]).strip()
        if block:
            return block

    # fallback: first N lines (keep as-is, but strip trailing whitespace)
    block = "\n".join([ln.rstrip() for ln in lines[:fallback_lines]]).strip()
    return block
    
def _build_metadata_prompt(header_block: str) -> str:
    return f"""
    You are extracting paper front-matter metadata for a scientific poster.

    Given the text snippet (typically authors/affiliations) BEFORE the Abstract section, extract:
    1) "authors": a single string listing authors in order.
    - If the snippet uses markers like 1,2,*,†, use Unicode superscript characters when helpful (e.g., ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁰, ⁎, †).
    - If there are too many authors to fit on a poster line, keep only the leading authors and abbreviate the rest:
        * Use "et al." after the last kept author (e.g., "Alice¹, Bob², Carol¹ et al.").
    2) "affiliations": a single string (may be empty "") describing affiliations/institutions if present.
    - keep a compact readable format.
    - If there are too many affiliations to fit, keep only the leading affiliations and abbreviate the rest using an ellipsis "…" (e.g., "¹ Univ A; ² Inst B; …").

    Return ONLY valid JSON with exactly these keys:
    {{"authors": "...", "affiliations": "..."}}

    Snippet:
    \"\"\"{header_block}\"\"\"
    """.strip()


def _parse_llm_json(raw: str) -> Optional[Dict[str, str]]:
    """
    Parse JSON from LLM response. Accepts if it contains the required keys.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # If the model accidentally wraps JSON in code fences, strip them.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        obj = json.loads(raw)
    except Exception:
        # Try to extract first JSON object substring
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None

    authors = obj.get("authors", "")
    affiliations = obj.get("affiliations", "")

    if not isinstance(authors, str) or not isinstance(affiliations, str):
        return None

    return {"authors": authors.strip(), "affiliations": affiliations.strip()}

def _call_agent_for_metadata(agent, header_block: str, max_retries: int = 2) -> Tuple[str, str]:
    """
    Calls agent.step(prompt=...) and expects JSON output.
    Returns (authors, affiliations). If fails, returns ("", "").
    """
    prompt = _build_metadata_prompt(header_block)

    last_raw = ""
    for attempt in range(max_retries + 1):
        prompt_try = prompt
        if attempt > 0:
            prompt_try = (
                prompt
                + "\n\nIMPORTANT: Your previous output was invalid. "
                  "Return ONLY valid JSON with keys authors and affiliations. No extra text."
            )

        resp = agent.step(prompt=prompt_try, images_base64=[])
        raw = (resp.get("content", "") or "").strip()
        last_raw = raw

        parsed = _parse_llm_json(raw)
        if parsed is not None:
            return parsed["authors"], parsed["affiliations"]

    # fallback
    return "", ""


def extract_paper_poster_meta(agent, input_dir: str, paper_id: str) -> Dict[str, str]:
    """
    Input:
      - agent: your Agent instance (must support agent.step(prompt=..., images_base64=[]))
      - input_dir: root input folder
      - paper_id: doc_id/paper_id

    Reads:
      {input_dir}/{paper_id}/full.md

    Extracts:
      - poster_title: derived from paper_id (strip leading digits/underscores; underscores->spaces)
      - authors: via LLM from content before #Abstract (or fallback first 10 lines)
      - affiliations: via LLM (may be empty)

    Output:
      {"poster_title": "...", "authors": "...", "affiliations": "..."}
    """
    input_root = Path(input_dir)
    md_path = input_root / paper_id / "full.md"

    if not md_path.exists():
        raise FileNotFoundError(f"full.md not found: {md_path}")

    header_block = load_md_header_block(md_path, fallback_lines=10)
    poster_title = make_poster_title_from_id(paper_id)

    authors, affiliations = _call_agent_for_metadata(agent, header_block, max_retries=2)

    return {
        "poster_title": poster_title,
        "authors": authors,
        "affiliations": affiliations,
    }
