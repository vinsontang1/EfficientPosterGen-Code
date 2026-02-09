"""
Style utilities for applying colors, fonts, and styling to poster content.

This module handles style application including colors, fonts, symbols,
and other visual styling options.
"""

from typing import List, Dict, Any, Optional, Tuple
from utils.wei_utils import style_bullet_content


def apply_main_text_font_size_override(bullet_content: List[Dict], font_size_val: Any) -> List[Dict]:
    """
    Apply font size override to main text (not titles).

    Args:
        bullet_content: Bullet point content list
        font_size_val: Font size value to apply

    Returns:
        Modified bullet content
    """
    if font_size_val is None:
        return bullet_content

    try:
        fs_int = int(font_size_val)
    except Exception:
        print(f"Warning: invalid main_text_font_size '{font_size_val}', skipping override")
        return bullet_content

    # Sections: index 0 is global title area; indexes >=1 are poster sections
    for si in range(1, len(bullet_content)):
        sect = bullet_content[si]
        # Only apply to main text boxes, not titles
        for key in ('textbox1', 'textbox2'):
            paras = sect.get(key)
            if isinstance(paras, list):
                for para in paras:
                    if isinstance(para, dict):
                        # Set/override paragraph-level default font size
                        para['font_size'] = fs_int

    return bullet_content


def apply_section_title_symbol(bullet_content: List[Dict], symbol: str) -> List[Dict]:
    """
    Add a prefix symbol to each section title (except global poster title).

    Args:
        bullet_content: Bullet point content list
        symbol: Symbol to prefix (e.g., '▶ ', '• ')

    Returns:
        Modified bullet content
    """
    if not symbol:
        return bullet_content

    try:
        # Skip index 0 (global title/author area)
        for i in range(1, len(bullet_content)):
            curr_content = bullet_content[i]
            title_paras = curr_content.get('title')
            if not title_paras or not isinstance(title_paras, list):
                continue

            # Only prefix on the first paragraph's first run
            first_para = title_paras[0]
            runs = first_para.get('runs') if isinstance(first_para, dict) else None
            if runs and isinstance(runs, list) and len(runs) > 0 and isinstance(runs[0], dict):
                text = runs[0].get('text', '')
                if not str(text).startswith(symbol):
                    runs[0]['text'] = f"{symbol}{text}"
    except Exception as e:
        # Non-fatal: continue without symbol if structure differs
        print(f"Warning: failed to apply section title symbol: {e}")

    return bullet_content


def apply_title_colors(
    bullet_content: List[Dict],
    title_text_color: Optional[Tuple[int, int, int]],
    title_fill_color: Optional[Tuple[int, int, int]]
) -> List[Dict]:
    """
    Apply colors to all titles (global title/author area and section titles).

    Args:
        bullet_content: Bullet point content list
        title_text_color: RGB tuple for title text
        title_fill_color: RGB tuple for title background

    Returns:
        Modified bullet content
    """
    # Apply to global title/author area (index 0)
    for k, v in bullet_content[0].items():
        style_bullet_content(v, title_text_color, title_fill_color)

    # Apply to section titles (indexes >= 1)
    for i in range(1, len(bullet_content)):
        curr_content = bullet_content[i]
        style_bullet_content(curr_content['title'], title_text_color, title_fill_color)

    return bullet_content


def apply_main_text_colors(
    bullet_content: List[Dict],
    main_text_color: Optional[Tuple[int, int, int]],
    main_text_fill_color: Optional[Tuple[int, int, int]]
) -> List[Dict]:
    """
    Apply colors to main text (bullet points, not titles).

    Args:
        bullet_content: Bullet point content list
        main_text_color: RGB tuple for main text
        main_text_fill_color: RGB tuple for main text background

    Returns:
        Modified bullet content
    """
    if main_text_color is None and main_text_fill_color is None:
        return bullet_content

    # Skip index 0 (global title/author area)
    for i in range(1, len(bullet_content)):
        curr_content = bullet_content[i]
        # Apply to bullet point textboxes
        for key in ('textbox1', 'textbox2'):
            if key in curr_content:
                text_color = main_text_color if main_text_color is not None else (0, 0, 0)
                fill_color = main_text_fill_color if main_text_fill_color is not None else (255, 255, 255)
                style_bullet_content(curr_content[key], text_color, fill_color)

    return bullet_content


def apply_all_styles(
    bullet_content: List[Dict],
    title_text_color: Optional[Tuple[int, int, int]],
    title_fill_color: Optional[Tuple[int, int, int]],
    main_text_color: Optional[Tuple[int, int, int]],
    main_text_fill_color: Optional[Tuple[int, int, int]],
    section_title_symbol: str = '',
    main_text_font_size: Any = None
) -> List[Dict]:
    """
    Apply all style configurations to bullet content.

    This is a convenience function that applies all styling in the correct order.

    Args:
        bullet_content: Bullet point content list
        title_text_color: RGB tuple for title text
        title_fill_color: RGB tuple for title background
        main_text_color: RGB tuple for main text
        main_text_fill_color: RGB tuple for main text background
        section_title_symbol: Symbol to prefix to section titles
        main_text_font_size: Font size for main text

    Returns:
        Fully styled bullet content
    """
    # Apply font size override first
    bullet_content = apply_main_text_font_size_override(bullet_content, main_text_font_size)

    # Apply section title symbol
    bullet_content = apply_section_title_symbol(bullet_content, section_title_symbol)

    # Apply title colors
    bullet_content = apply_title_colors(bullet_content, title_text_color, title_fill_color)

    # Apply main text colors
    bullet_content = apply_main_text_colors(bullet_content, main_text_color, main_text_fill_color)

    return bullet_content
