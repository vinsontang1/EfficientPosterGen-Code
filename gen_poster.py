import argparse
import copy
import json, time, csv
import os
import re
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
from contextlib import contextmanager
from config import config

from src.utils.detectorV1 import detect_text_overflow
from src.utils.detectorV2 import detect_panel_status

from PosterAgent.gen_pptx_code import generate_poster_code
from PosterAgent.tree_split_layout import (
    get_arrangments_in_inches,
    main_inference,
    main_train,
    split_textbox,
    to_inches,
)
from utils.config_utils import (
    extract_colors,
    extract_font_sizes,
    extract_section_title_symbol,
    extract_vertical_alignment,
    load_poster_yaml_config,
    normalize_config_values,
)
from utils.logo_utils import add_logos_to_poster_code
from utils.style_utils import apply_all_styles
from utils.theme_utils import create_theme_with_alignment, get_default_theme, resolve_colors
from utils.wei_utils import run_code, scale_to_target_area, utils_functions
from utils.src.utils import ppt_to_images

UNITS_PER_INCH = 25
YELLOW_FILL = (255, 255, 0)
MIN_TEXT_FONT_SIZE = 18
MIN_TITLE_FONT_SIZE = 24
FONT_DECREMENT = 2
DETECTION_PAD_INCHES = 2.0
BULLET_STAGE_DIR  = "05_summary"

@contextmanager
def timer(task_name: str, doc_id: str = "N/A", csv_path: str = None):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"[{doc_id}] Task '{task_name}' finished in {elapsed:.4f}s")
        
        if csv_path:
            file_exists = os.path.isfile(csv_path)
            try:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['DocID', 'TaskName', 'Duration_Seconds', 'Timestamp'])
                    writer.writerow([doc_id, task_name, f"{elapsed:.4f}", time.strftime("%Y-%m-%d %H:%M:%S")])
            except Exception as e:
                print(f"Failed to write to CSV: {e}")

def load_bullet_points(
    bullet_json_path: str,
    doc_id: Optional[str] = None,
) -> Tuple[str, Dict[str, List[str]], Dict[str, str]]:
    bullet_json = json.load(open(bullet_json_path, "r", encoding="utf-8"))
    if not isinstance(bullet_json, dict) or not bullet_json:
        raise ValueError("Bullet JSON must be a non-empty dict.")

    content = bullet_json.get("content")
    if not isinstance(content, dict):
        raise ValueError("Bullet JSON missing 'content' dict at root.")

    resolved_id = doc_id or bullet_json.get("paper_id") or "paper"
    cleaned_content: Dict[str, List[str]] = {}
    section_title_map: Dict[str, str] = {}
    for section, bullets in content.items():
        if not isinstance(bullets, list):
            cleaned_content[section] = []
            continue
        cleaned_bullets: List[str] = []
        for bullet in bullets:
            bullet_text = str(bullet).strip()
            if bullet_text.upper().startswith("TITLE:"):
                title_value = bullet_text.split(":", 1)[-1].strip()
                if title_value:
                    section_title_map[section] = title_value
                continue
            cleaned_bullets.append(bullet_text)
        cleaned_content[section] = cleaned_bullets

    return resolved_id, cleaned_content, section_title_map


def load_asset_map(path: Optional[str]) -> Dict:
    if path is None:
        return {}
    return json.load(open(path, "r", encoding="utf-8"))


def norm_key(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"^(?:\d+|[IVX]+|[A-Z])(?:[\._-]+)", "", s, flags=re.IGNORECASE)
    return s.upper().strip("_")


def normalize_title_section(title: str) -> str:
    return title.strip() if title else "Untitled"


def build_title_content(title: str, authors: str, title_fs: int, author_fs: int) -> Dict:
    lines = [ln.strip() for ln in str(authors).splitlines() if ln.strip()]

    textbox1_items = []
    for i, ln in enumerate(lines):
        textbox1_items.append(
            {
                "alignment": "center",   
                "align": "center",       
                "bullet": False,
                "level": 0,
                "font_size": author_fs if i == 0 else max(10, author_fs - 4),
                "runs": [{"text": ln}],
            }
        )

    return {
        "title": [
            {
                "alignment": "center",
                "align": "center",       
                "bullet": False,
                "level": 0,
                "font_size": title_fs,
                "runs": [{"text": title, "bold": True}],
            }
        ],
        "textbox1": textbox1_items 
        if textbox1_items else [
            {
                "alignment": "center",
                "align": "center",
                "bullet": False,
                "level": 0,
                "font_size": author_fs,
                "runs": [{"text": ""}],
            }
        ],
    }



def build_section_content(
    section_title: str,
    bullets: List[str],
    bullet_fs: int,
    title_fs: int,
    num_textboxes: int,
) -> Dict:
    section_title = normalize_title_section(section_title)
    title_item = {
        "alignment": "left",
        "bullet": False,
        "level": 0,
        "font_size": title_fs,
        "runs": [{"text": section_title, "bold": True}],
    }

    def bullet_item(text: str) -> Dict:
        return {
            "alignment": "left",
            "bullet": True,
            "level": 0,
            "font_size": bullet_fs,
            "runs": [{"text": text}],
        }

    bullets = [b for b in bullets if str(b).strip()]
    if not bullets:
        bullets = [""]

    if num_textboxes == 1:
        return {
            "title": [title_item],
            "textbox1": [bullet_item(b) for b in bullets],
        }

    split_idx = max(1, len(bullets) // 2)
    left = bullets[:split_idx]
    right = bullets[split_idx:] or [""]
    return {
        "title": [title_item],
        "textbox1": [bullet_item(b) for b in left],
        "textbox2": [bullet_item(b) for b in right],
    }


def build_panels(
    section_titles: List[str],
    section_bullets: Dict[str, List[str]],
    sec2pic: Dict,
    include_title_panel: bool = True,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    panels = []
    section_assets = {}

    total_text_len = 0

    panel_titles = list(section_titles)
    if include_title_panel and (not panel_titles or panel_titles[0].lower() != "title"):
        panel_titles = ["Title"] + panel_titles

    sec2pic_norm = {}
    if isinstance(sec2pic, dict):
        for key, value in sec2pic.items():
            sec2pic_norm[norm_key(key)] = value

    candidates: List[Tuple[str, int]] = []
    info_by_title: Dict[str, Dict] = {}

    for idx, title in enumerate(panel_titles):
        bullets = section_bullets.get(title, [])
        text_len = sum(len(str(b)) for b in bullets) + len(title)
        total_text_len += text_len

        panel = {
            "panel_id": idx,
            "section_name": title,
            "tp": 0,
            "text_len": text_len,
            "gp": 0,
            "figure_size": 0,
            "figure_aspect": 1,
        }
        panels.append(panel)

        if title.lower() == "title":
            continue

        sec_info = sec2pic_norm.get(norm_key(title))
        if isinstance(sec_info, dict) and sec_info.get("path"):
            candidates.append((title, text_len))
            info_by_title[title] = sec_info

    if len(candidates) <= 3:
        selected_titles = {title for title, _ in candidates}
    else:
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected_titles = {title for title, _ in candidates[:3]}

    total_figure_area = 0
    for panel in panels:
        if panel["section_name"].lower() == "title":
            continue
        if panel["section_name"] not in selected_titles:
            continue

        sec_info = info_by_title.get(panel["section_name"])
        if not sec_info:
            continue

        figure_path = sec_info.get("path")
        if not figure_path or not os.path.exists(figure_path):
            continue

        with Image.open(figure_path) as img:
            figure_size = img.width * img.height
            figure_aspect = img.width / img.height if img.height else 1

        total_figure_area += figure_size
        panel["figure_size"] = figure_size
        panel["figure_aspect"] = figure_aspect
        section_assets[panel["section_name"]] = {
            "figure_path": figure_path,
            "figure_size": figure_size,
            "figure_aspect": figure_aspect,
        }

    for panel in panels:
        panel["tp"] = panel["text_len"] / max(total_text_len, 1)
        if total_figure_area > 0:
            panel["gp"] = panel["figure_size"] / total_figure_area

    return panels, section_assets


def assign_figure_paths(figure_arrangement: List[Dict], panels: List[Dict], section_assets: Dict):
    panel_by_id = {p["panel_id"]: p for p in panels}
    for fig in figure_arrangement:
        panel = panel_by_id.get(fig["panel_id"])
        if not panel:
            continue
        assets = section_assets.get(panel["section_name"])
        if not assets:
            continue
        fig["figure_path"] = assets["figure_path"]


def apply_yellow_fill(textbox_content: List[Dict]) -> List[Dict]:
    content_copy = copy.deepcopy(textbox_content)
    for item in content_copy:
        runs = item.get("runs")
        if isinstance(runs, list):
            for run in runs:
                run["fill_color"] = YELLOW_FILL
    return content_copy


def render_textbox_for_detection(
    textbox: Dict,
    textbox_content: List[Dict],
    slide_size: Tuple[float, float],
    tmp_root: str,
) -> Tuple[str, str, Tuple[float, float, float, float], Tuple[float, float]]:
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    pptx_path = os.path.join(tmp_dir, "poster.pptx")
    pad = DETECTION_PAD_INCHES
    padded_slide_size = (
    min(slide_size[0] + 2 * pad, 56),
    min(slide_size[1] + 2 * pad, 56),
    )
    padded_textbox = dict(textbox)
    padded_textbox["x"] = textbox["x"] + pad
    padded_textbox["y"] = textbox["y"] + pad
    poster_code = generate_poster_code(
        [],
        [padded_textbox],
        [],
        presentation_object_name="poster_presentation",
        slide_object_name="poster_slide",
        utils_functions=utils_functions,
        slide_width=padded_slide_size[0],
        slide_height=padded_slide_size[1],
        img_path=None,
        save_path=pptx_path,
        visible=False,
        content=textbox_content,
        check_overflow=True,
        theme=None,
        tmp_dir=tmp_dir,
        overflow_fill_color=YELLOW_FILL,
        slide_background_color=YELLOW_FILL,
    )
    output, err = run_code(poster_code)
    if err is not None:
        raise RuntimeError(f"Error rendering textbox for detection: {err}")
    ppt_to_images(pptx_path, tmp_dir, output_type="png")
    image_path = os.path.join(tmp_dir, "poster.png")
    layer_coords = (
        padded_textbox["x"],
        padded_textbox["y"],
        padded_textbox["width"],
        padded_textbox["height"],
    )
    return image_path, tmp_dir, layer_coords, padded_slide_size


def textbox_key_from_id(textbox_id: int) -> Optional[str]:
    if textbox_id == 0:
        return "title"
    if textbox_id >= 1:
        return f"textbox{textbox_id}"
    return None


def shrink_section_content(section_content: Dict) -> bool:
    changed = False
    for key in ("title", "textbox1", "textbox2"):
        items = section_content.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            font_size = item.get("font_size")
            min_font_size = MIN_TITLE_FONT_SIZE if key == "title" else MIN_TEXT_FONT_SIZE
            if isinstance(font_size, int) and font_size > min_font_size:
                item["font_size"] = max(min_font_size, font_size - FONT_DECREMENT)
                changed = True

    if changed:
        return True

    truncated = False
    for key in ("textbox1", "textbox2"):
        items = section_content.get(key)
        if isinstance(items, list) and len(items) > 1:
            items.pop()
            truncated = True
    return truncated


def run_deterministic_overflow_commenter(
    panels: List[Dict],
    text_arrangement_inches: List[Dict],
    content_sections: List[Dict],
    panel_content_index: Dict[int, int],
    slide_size: Tuple[float, float],
    tmp_root: str,
    max_iters: int,
) -> bool:
    textboxes_by_panel = {}
    for textbox in text_arrangement_inches:
        textboxes_by_panel.setdefault(textbox["panel_id"], []).append(textbox)

    stalled_overflow = False
    for _ in range(max_iters):
        any_overflow = False
        for panel in panels:
            if panel["section_name"].lower() == "title":
                continue
            panel_id = panel["panel_id"]
            content_idx = panel_content_index.get(panel_id)
            if content_idx is None:
                continue
            section_content = content_sections[content_idx]
            panel_textboxes = textboxes_by_panel.get(panel_id, [])
            panel_overflow = False

            for textbox in panel_textboxes:
                textbox_key = textbox_key_from_id(textbox.get("textbox_id", 0))
                if textbox_key is None:
                    continue
                textbox_content = section_content.get(textbox_key)
                if not isinstance(textbox_content, list):
                    continue

                detect_content = apply_yellow_fill(textbox_content)
                image_path, tmp_dir, layer_coords, effective_slide_size = render_textbox_for_detection(
                    textbox, detect_content, slide_size, tmp_root
                )
                try:
                    # result = detect_text_overflow(
                    #     image_path=image_path,
                    #     layer_coords=layer_coords,
                    #     slide_size=effective_slide_size,
                    # )
                    result = detect_panel_status(
                        image_path=image_path,
                        layer_coords=layer_coords,
                        slide_size=effective_slide_size,
                    )
                    
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                if result == 1:
                    panel_overflow = True
                    break

            if panel_overflow:
                any_overflow = True
                if not shrink_section_content(section_content):
                    stalled_overflow = True

        if not any_overflow:
            return False
        if stalled_overflow:
            return True
    return True


def gen_poster(args):
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    doc_id, section_bullets, section_title_map = load_bullet_points(args.bullet_json, args.doc_id)
    poster_title = args.poster_title or doc_id
    sec2pic = load_asset_map(args.sec2pic)

    section_keys = list(section_bullets.keys())

    panels, section_assets = build_panels(section_keys, section_bullets, sec2pic)

    meta_json_path = os.path.join(
        config["files"]["save_dir"],
        "parsed_papers",
        str(args.doc_id),
        "meta.json",
    )
    if args.poster_width_inches is not None and args.poster_height_inches is not None:
        poster_width = args.poster_width_inches * UNITS_PER_INCH
        poster_height = args.poster_height_inches * UNITS_PER_INCH
    elif os.path.exists(meta_json_path):
        meta_json = json.load(open(meta_json_path, "r"))
        poster_width = meta_json["width"]
        poster_height = meta_json["height"]
    else:
        poster_width = 48 * UNITS_PER_INCH
        poster_height = 36 * UNITS_PER_INCH

    poster_width, poster_height = scale_to_target_area(poster_width, poster_height)
    poster_width_inches = to_inches(poster_width, UNITS_PER_INCH)
    poster_height_inches = to_inches(poster_height, UNITS_PER_INCH)

    if poster_width_inches > 56 or poster_height_inches > 56:
        scale_factor = 56 / max(poster_width_inches, poster_height_inches)
        poster_width_inches *= scale_factor
        poster_height_inches *= scale_factor
        poster_width = poster_width_inches * UNITS_PER_INCH
        poster_height = poster_height_inches * UNITS_PER_INCH

    panel_model_params, figure_model_params = main_train()
    panel_arrangement, figure_arrangement, text_arrangement = main_inference(
        panels,
        panel_model_params,
        figure_model_params,
        poster_width,
        poster_height,
        shrink_margin=3,
    )

    # Split title textbox into title + authors
    title_box = text_arrangement[0]
    title_top, title_bottom = split_textbox(title_box, 0.8)
    text_arrangement = [title_top, title_bottom] + text_arrangement[1:]

    assign_figure_paths(figure_arrangement, panels, section_assets)

    width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches = (
        get_arrangments_in_inches(
            poster_width,
            poster_height,
            panel_arrangement,
            figure_arrangement,
            text_arrangement,
            UNITS_PER_INCH,
        )
    )

    yaml_cfg = load_poster_yaml_config(args.poster_path)
    bullet_fs, title_fs, poster_title_fs, poster_author_fs = extract_font_sizes(yaml_cfg)
    title_text_color, title_fill_color, main_text_color, main_text_fill_color = extract_colors(yaml_cfg)
    section_title_vertical_align = extract_vertical_alignment(yaml_cfg)
    section_title_symbol = extract_section_title_symbol(yaml_cfg)

    (
        bullet_fs,
        title_fs,
        poster_title_fs,
        poster_author_fs,
        title_text_color,
        title_fill_color,
        main_text_color,
        main_text_fill_color,
    ) = normalize_config_values(
        bullet_fs,
        title_fs,
        poster_title_fs,
        poster_author_fs,
        title_text_color,
        title_fill_color,
        main_text_color,
        main_text_fill_color,
    )

    bullet_fs = bullet_fs or 36
    title_fs = title_fs or 40
    poster_title_fs = poster_title_fs or 60
    poster_author_fs = poster_author_fs or 32

    content_sections = [
        build_title_content(poster_title, args.poster_authors, poster_title_fs, poster_author_fs)
    ]
    panel_content_index = {}
    title_panel_id = None
    for panel in panels:
        if panel["section_name"].lower() == "title":
            title_panel_id = panel["panel_id"]
            break
    if title_panel_id is not None:
        panel_content_index[title_panel_id] = 0
    content_index = 1

    # Determine how many textboxes per section based on text arrangement
    textboxes_by_panel = {}
    for t in text_arrangement[2:]:
        textboxes_by_panel.setdefault(t["panel_id"], []).append(t)

    for panel in panels:
        if panel["section_name"].lower() == "title":
            continue
        num_textboxes = max(1, len(textboxes_by_panel.get(panel["panel_id"], [])) - 1)
        section_key = panel["section_name"]
        section_title = section_title_map.get(section_key, section_key)
        bullets = section_bullets.get(section_key, [])
        content_sections.append(
            build_section_content(section_title, bullets, bullet_fs, title_fs, num_textboxes)
        )
        panel_content_index[panel["panel_id"]] = content_index
        content_index += 1

    final_title_text_color, final_title_fill_color, final_main_text_color, final_main_text_fill_color = resolve_colors(
        title_text_color,
        title_fill_color,
        main_text_color,
        main_text_fill_color,
    )

    content_sections = apply_all_styles(
        content_sections,
        title_text_color=final_title_text_color,
        title_fill_color=final_title_fill_color,
        main_text_color=final_main_text_color,
        main_text_fill_color=final_main_text_fill_color,
        section_title_symbol=section_title_symbol,
        main_text_font_size=bullet_fs,
    )

    overflow_remaining = run_deterministic_overflow_commenter(
        panels,
        text_arrangement_inches,
        content_sections,
        panel_content_index,
        (width_inch, height_inch),
        args.tmp_dir,
        args.overflow_max_iters,
    )

    base_theme = get_default_theme()
    theme_with_alignment = create_theme_with_alignment(base_theme, section_title_vertical_align)

    poster_code = generate_poster_code(
        panel_arrangement_inches,
        text_arrangement_inches,
        figure_arrangement_inches,
        presentation_object_name="poster_presentation",
        slide_object_name="poster_slide",
        utils_functions=utils_functions,
        slide_width=width_inch,
        slide_height=height_inch,
        img_path=None,
        save_path=f"{args.tmp_dir}/poster.pptx",
        visible=False,
        content=content_sections,
        theme=theme_with_alignment,
        tmp_dir=args.tmp_dir,
        auto_shrink_text=overflow_remaining,
    )

    poster_code = add_logos_to_poster_code(
        poster_code,
        width_inch,
        height_inch,
        institution_logo_path=args.institution_logo_path,
        conference_logo_path=args.conference_logo_path,
    )

    output, err = run_code(poster_code)
    if err is not None:
        raise RuntimeError(f"Error generating PowerPoint: {err}")

    output_dir = os.path.join(args.output_dir, doc_id)
    os.makedirs(output_dir, exist_ok=True)

    pptx_path = os.path.join(output_dir, f"{doc_id}.pptx")
    shutil.move(f"{args.tmp_dir}/poster.pptx", pptx_path)
    ppt_to_images(pptx_path, output_dir)

    print(f"Poster saved to {pptx_path}")


def make_poster_title(doc_id: str) -> str:
    s = re.sub(r"^\d+_+", "", doc_id)
    s = s.replace("_", " ")
    return s.strip()


def fill_args_from_input_dir(args, doc_id: str):
    input_dir = Path(args.input_dir)

    args.doc_id = doc_id
    args.bullet_json  = str(input_dir / BULLET_STAGE_DIR  / f"{doc_id}.json")
    args.poster_title = make_poster_title(doc_id)


def iter_doc_ids(input_dir: str):
    bullet_dir = Path(input_dir) / BULLET_STAGE_DIR
    for p in sorted(bullet_dir.glob("*.json")):
        yield p.stem
 

def fill_poster_authors_from_meta(args, doc_id: str):
    """
    Read {input_dir}/00_meta/{doc_id}.json and set args.poster_authors to:
      "<authors>\n<affiliations>"
    Affiliations may be empty.
    """
    meta_path = Path(args.input_dir) / "00_meta" / f"{doc_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"poster_authors meta not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    authors = (meta.get("authors") or "").strip()
    affiliations = (meta.get("affiliations") or "").strip()

    if authors and affiliations:
        args.poster_authors = f"{authors}\n{affiliations}"
    else:
        # if affiliations missing, just keep authors (or empty string if authors missing)
        args.poster_authors = authors or ""

def sanitize(name: str) -> str:
    """用于文件夹名安全化"""
    return re.sub(r'[^\w\-\.]', '_', name).strip('_')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate poster from bullet points and asset paths.")
    parser.add_argument("--input_dir", default=None, help="Input directory for processing.")
    # parser.add_argument("--bullet_json", required=True, help="Path to bullet point JSON.")
    # parser.add_argument("--doc_id", default=None, help="Document key inside bullet JSON.")
    # parser.add_argument("--poster_title", default=None, help="Poster title text.")
    # parser.add_argument("--poster_authors", required=True, help="Poster authors/affiliations text.")
    parser.add_argument("--poster_path", default="./", help="Path to poster PDF for YAML config lookup.")
    parser.add_argument("--output_dir", default="generated_posters", help="Directory for output assets.")
    parser.add_argument("--poster_width_inches", type=float, default=None)
    parser.add_argument("--poster_height_inches", type=float, default=None)
    parser.add_argument("--institution_logo_path", default=None)
    parser.add_argument("--conference_logo_path", default=None)
    parser.add_argument("--tmp_dir", default="tmp")
    parser.add_argument("--overflow_max_iters", type=int, default=5)
    parser.add_argument("--sec2pic_dir", default="./sec2pic", help="Section-to-picture JSON mapping.")
    parser.add_argument("--pid", type=str, default="qwen")
    args = parser.parse_args()
    TIME_LOG_PATH = os.path.join(args.output_dir, f"gen_poster_time_log_{args.pid}.csv")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    with timer("Global_Execution", doc_id="ALL", csv_path=TIME_LOG_PATH):
        if args.input_dir:
            for doc_id in iter_doc_ids(args.input_dir):
                doc_id = sanitize(doc_id)
                with timer("Total_Single_Paper", doc_id=doc_id, csv_path=TIME_LOG_PATH):
                    args.sec2pic = Path(args.sec2pic_dir) / f"{doc_id}.json"
                    fill_args_from_input_dir(args, doc_id)
                    try:
                        fill_poster_authors_from_meta(args, doc_id)
                    except FileNotFoundError as e:
                        print(f"[skip] {e}")
                        continue

                    if not Path(args.bullet_json).exists():
                        print(f"[skip] missing bullet_json: {args.bullet_json}")
                        continue

                    try:
                        with timer("gen_poster_func", doc_id=doc_id, csv_path=TIME_LOG_PATH):
                            gen_poster(args)
                    except Exception as e:
                        print(f"[Error] Failed to generate poster for {doc_id}: {e}")
                        continue
        else:
            if not args.bullet_json or not Path(args.bullet_json).exists():
                raise FileNotFoundError(f"bullet_json not found: {args.bullet_json}")
            gen_poster(args)
