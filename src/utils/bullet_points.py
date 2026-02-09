import os
import re
import time
import json
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.agent_utils import ModelFactory, Agent
from config import config
import argparse
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =========================
# Helpers
# =========================

def encode_image(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {path}: {e}")
        return ""


def sanitize(name: str) -> str:
    """用于文件夹名安全化"""
    return re.sub(r"[^\w\-\.]", "_", name).strip("_")


# =========================
# Prompt
# =========================

def build_prompt(max_bullets: int = 4, include_formulas: bool = False) -> str:
    formula_instruction = "Include key formulas if critical" if include_formulas else "Omit formula"
    prompt = f"""You are an expert Academic Editor and CVPR/ICCV Area Chair. Your goal is to assist a researcher in condensing a complex paper section into a visual Scientific Poster.

    Here are examples of the expected input-output format:

    Input: "The proposed model is based on the learned-domain masking approach [14, 15, 17–22] and employs an encoder, a decoder, and a masking network, as shown in Figure 1. The encoder is fully convolutional, while the masking network employs two Transformers embedded inside the dual-path processing block proposed in [17]. The decoder finally reconstructs the separated signals in the time domain by using the masks predicted by the masking network. To foster reproducibility, the SepFormer will be made available within the SpeechBrain toolkit."
    # Output:
    - Adopts learned-domain masking with convolutional encoder
    - Uses dual-path Transformers in masking network
    - Releases SepFormer in SpeechBrain toolkit
    TITLE: SepFormer Overview

    (1) OCR & Denoise:
    Read the text from images, strictly ignore headers, footers, page numbers, and citation brackets (e.g., [1], (Lee et al.)).

    (2) Signal Extraction (IMPORTANT):
    Treat input as a unit and decide how many bullets it deserves.
    Write MORE bullets for high-novelty, high-impact, poster-worthy content (new method/insight/strong results).
    Write FEWER bullets for generic background, motivation, or standard setup.
    Across the whole section, output at most {max_bullets} bullets total (STRICT).

    (3) Active Rewriting:
    Convert passive sentences into strong active points (e.g., "Proposes a module").

    (4) Length Control :
    Each bullet MUST be short and poster-friendly:
    - Prefer ≤16 words per bullet 
    - If a point is longer, compress by removing qualifiers, examples, and subordinate clauses

    (5) Compressed Section Title (STRICT):
    After the bullet list, output ONE extra line that is a compressed version of the ORIGINAL section title.
    Requirements:
    - EXACT format: "TITLE: <title>"
    - <title> MUST be at most 3 words total (hard constraint).
    - Keep the original meaning and topic; do NOT invent a new title.
    - Prefer using key nouns from the original title; remove numbering, punctuation, and filler words.
    - Do NOT start this line with a hyphen "-".
    - Output exactly ONE TITLE line and nothing else besides the bullets.
    - Remove section indices like "1", "I", "4", "6", "A.", etc.

    (6) Output Formatting (STRICT):
    - Output bullets as a Markdown list using hyphens (-).
    - Then output the TITLE line as the final line (same indentation level).
    - Output ONLY the final bullet list + the final TITLE line. No preamble, no explanation.
    {formula_instruction.lower()}
    """
    return prompt


# =========================
# Validation & Parsing
# =========================

TITLE_RE = re.compile(r"^\s*TITLE:\s*(.+?)\s*$", re.IGNORECASE)

def _count_words_english(s: str) -> int:
    # English-ish token count; safe for Title Case words
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", s)
    return len(tokens)


def parse_and_validate_llm_output(raw_text: str, max_bullets: int) -> Tuple[bool, str, List[str], str]:
    """
    Validate the strict format:
      - zero or more lines starting with '-'
      - exactly one TITLE line at the end: 'TITLE: <<=3 words>'
      - no other lines
    Return:
      ok, title, bullets(cleaned without '-'), reason
    """
    if not raw_text or not raw_text.strip():
        return False, "", [], "empty output"

    lines = [ln.rstrip() for ln in raw_text.splitlines()]

    # drop trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return False, "", [], "empty lines"

    # light filter for accidental preamble
    bad_prefixes = ("HERE ARE", "INTERNAL THOUGHT", "THOUGHT:", "OUTPUT:", "EXPLANATION")
    filtered = []
    for ln in lines:
        if ln.strip().upper().startswith(bad_prefixes):
            continue
        filtered.append(ln)
    lines = filtered

    # find TITLE
    title_idxs = [i for i, ln in enumerate(lines) if TITLE_RE.match(ln)]
    if len(title_idxs) != 1:
        return False, "", [], f"TITLE lines count != 1 (got {len(title_idxs)})"

    title_idx = title_idxs[0]
    if title_idx != len(lines) - 1:
        return False, "", [], "TITLE is not the last non-empty line"

    title_line = lines[title_idx].strip()
    if title_line.startswith("-"):
        return False, "", [], "TITLE line starts with '-'"

    m = TITLE_RE.match(title_line)
    title = (m.group(1) if m else "").strip()
    if not title:
        return False, "", [], "empty TITLE value"

    wc = _count_words_english(title)
    if wc > 3:
        return False, "", [], f"TITLE has >3 words (got {wc})"

    # bullets before TITLE
    bullets: List[str] = []
    for ln in lines[:title_idx]:
        if not ln.strip():
            continue
        if not ln.strip().startswith("-"):
            return False, "", [], f"non-bullet line before TITLE: {ln[:60]}"
        b = ln.strip().lstrip("-").strip()
        if b:
            bullets.append(b)

    if len(bullets) > max_bullets:
        bullets = bullets[:max_bullets]  # clamp rather than fail

    return True, title, bullets, ""


def llm_step_with_validation(
    agent: Agent,
    prompt: str,
    images_base64: List[str],
    max_bullets: int,
    max_retries: int = 2,
) -> Tuple[str, List[str], Dict]:
    """
    Calls the agent, validates output, retries if invalid.
    Returns:
      compressed_title, bullets, total_usage
    """
    last_reason = ""
    total_usage = {"input_text": 0, "input_image": 0, "input_total": 0, "output": 0}

    # ensure no empty images
    images_base64 = [b for b in images_base64 if b]

    for attempt in range(max_retries + 1):
        prompt_try = prompt
        if attempt > 0:
            prompt_try = (
                prompt
                + "\n\nIMPORTANT: Your last output violated the required format. "
                  "Return ONLY hyphen bullets followed by a final line 'TITLE: <max 3 words>'. "
                  "No extra text.\n"
            )

        resp = agent.step(prompt=prompt_try, images_base64=images_base64)
        raw_text = resp.get("content", "") or ""
        usage = resp.get("usage", {}) or {}

        for k in total_usage.keys():
            total_usage[k] += int(usage.get(k, 0) or 0)

        ok, title, bullets, reason = parse_and_validate_llm_output(raw_text, max_bullets=max_bullets)
        if ok:
            return title, bullets, total_usage

        last_reason = reason
        logger.warning(f"Invalid LLM output (attempt {attempt+1}/{max_retries+1}): {reason}")

    # fallback
    logger.warning(f"LLM output invalid after retries: {last_reason}")
    return "Section", [], total_usage


# =========================
# Main summarization
# =========================

def summarize_single_paper(
    agent: Agent,
    output_root: str,
    paper_id: str
) -> Dict:
    """
    returns:
    {
      "paper_id": ...,
      "content": {section: [bullets..., "TITLE: xxx"]},
      "token_usage": {...}
    }
    """
    safe_paper_id = sanitize(paper_id)
    paper_dir = Path(output_root) / safe_paper_id

    if not paper_dir.exists():
        logger.warning(f"Image directory not found for paper: {paper_id} ({paper_dir})")
        return {}

    logger.info(f"Summarizing Paper: {paper_id} from {paper_dir}")

    current_paper_content: Dict[str, List[str]] = {}
    current_paper_usage = {"input_text": 0, "input_image": 0, "input_total": 0, "output": 0}

    max_bullets = config['bullet_points']['MAX_BULLETS']  # keep consistent with build_prompt default unless you pass it in
    prompt_template = build_prompt(max_bullets=max_bullets)
    has_valid_content = False

    for section_dir in sorted(paper_dir.iterdir()):
        if not section_dir.is_dir():
            continue

        png_files = sorted(section_dir.glob("*.png"))
        if not png_files:
            continue

        has_valid_content = True
        logger.info(f"  > Processing Section: {section_dir.name} ({len(png_files)} images)")

        try:
            images_base64 = [encode_image(png) for png in png_files]

            compressed_title, bullets, usage = llm_step_with_validation(
                agent=agent,
                prompt=prompt_template,
                images_base64=images_base64,
                max_bullets=max_bullets,
                max_retries=2,
            )

            bullets_with_title = list(bullets)
            bullets_with_title.append(f"TITLE: {compressed_title}")

            current_paper_content[section_dir.name] = bullets_with_title

            current_paper_usage["input_text"] += usage.get("input_text", 0)
            current_paper_usage["input_image"] += usage.get("input_image", 0)
            current_paper_usage["input_total"] += usage.get("input_total", 0)
            current_paper_usage["output"] += usage.get("output", 0)

        except Exception as e:
            logger.error(f"Failed to summarize section {section_dir.name}: {e}")
            current_paper_content[section_dir.name] = ["Error: Processing failed.", "TITLE: Section"]

    if not has_valid_content:
        return {}

    return {
        "paper_id": paper_id,
        "content": current_paper_content,
        "token_usage": current_paper_usage
    }


# =========================
# CLI Entrypoint
# =========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Bullet Point Generation Pipeline with Title Compression + Validation")
    parser.add_argument("--provider", type=str, default="openkey",
                        choices=["openai", "openkey", "siliconflow", "google"], help="LLM provider")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use")
    parser.add_argument("--output_root", type=str, default="./output",
                        help="Root directory containing the processed paper images")
    parser.add_argument("--summary_dir", type=str, default="./summary",
                        help="Directory to save individual summary JSONs")

    args = parser.parse_args()

    OUTPUT_ROOT = Path(args.output_root)
    SUMMARY_DIR = Path(args.summary_dir)

    if not OUTPUT_ROOT.exists():
        logger.error(f"Input directory not found: {OUTPUT_ROOT}")
        logger.error("Please provide a valid path using --output_root")
        raise SystemExit(1)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Starting Bullet Point Generation Pipeline...")
        print(f" > Model: {args.model_name} ({args.provider})")
        print(f" > Input: {OUTPUT_ROOT}")
        print(f" > Output Dir: {SUMMARY_DIR}")

        agent = ModelFactory.create(provider=args.provider, model_name=args.model_name)

        valid_paper_dirs: List[Path] = []
        logger.info("Scanning for valid paper directories...")

        for item in OUTPUT_ROOT.iterdir():
            if item.is_dir():
                has_png = any(item.glob("*/*.png"))
                if has_png:
                    valid_paper_dirs.append(item)
                else:
                    logger.debug(f"Skipping non-target directory: {item.name}")

        logger.info(f"Found {len(valid_paper_dirs)} valid papers in {OUTPUT_ROOT}")

        for paper_dir in tqdm(valid_paper_dirs, desc="Processing Papers"):
            paper_id = paper_dir.name
            save_path = SUMMARY_DIR / f"{paper_id}.json"

            result = summarize_single_paper(
                agent=agent,
                output_root=str(OUTPUT_ROOT),
                paper_id=paper_id,
            )

            if result:
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Failed to save JSON for {paper_id}: {e}")

        print("Bullet Point Generation Completed.")
        logger.info(f"[Done] All summaries saved to {SUMMARY_DIR}")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
    except Exception as e:
        logger.critical(f"Critical failure: {e}", exc_info=True)
        raise
