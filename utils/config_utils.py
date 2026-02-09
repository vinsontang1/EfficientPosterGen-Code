"""
Configuration utilities for loading and managing poster configurations.

This module handles YAML configuration loading, type conversions, and
configuration management for the poster generation pipeline.
"""

import os
import yaml
from typing import Dict, Any, Optional, Tuple


def load_poster_yaml_config(poster_path: str) -> Dict[str, Any]:
    """
    Load poster configuration from YAML files.

    Searches for poster.yaml in multiple locations:
    1. Next to the poster PDF (per-poster config)
    2. Project config directory
    3. Root directory

    Later configs override earlier ones.

    Args:
        poster_path: Path to the poster PDF file

    Returns:
        Dictionary containing the merged configuration
    """
    cfg = {}
    try_paths = []

    # Prefer per-poster YAML next to the PDF
    poster_dir = os.path.dirname(poster_path)
    try_paths.append(os.path.join(poster_dir, 'poster.yaml'))

    # Project-level defaults
    try_paths.append(os.path.join('config', 'poster.yaml'))
    try_paths.append('poster.yaml')

    for p in try_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f) or {}
                    if isinstance(loaded, dict):
                        cfg.update(loaded)
            except Exception as e:
                print(f"Warning: failed to read YAML config {p}: {e}")

    return cfg


def extract_font_sizes(yaml_cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract font size configurations from YAML config.

    Supports both flat and nested structures:
    - Flat: main_text_font_size, section_title_font_size, etc.
    - Nested: text.main_font_size, title.font_size, etc.

    Args:
        yaml_cfg: YAML configuration dictionary

    Returns:
        Tuple of (bullet_fs, title_fs, poster_title_fs, poster_author_fs)
    """
    bullet_fs = None
    title_fs = None
    poster_title_fs = None
    poster_author_fs = None

    if not isinstance(yaml_cfg, dict):
        return bullet_fs, title_fs, poster_title_fs, poster_author_fs

    # Main/bullet font size
    bullet_fs = yaml_cfg.get('main_text_font_size')
    text_cfg = yaml_cfg.get('text') if isinstance(yaml_cfg.get('text'), dict) else None
    if bullet_fs is None and text_cfg:
        bullet_fs = text_cfg.get('main_font_size') or text_cfg.get('main_text_font_size')

    # Section title font size
    title_fs = yaml_cfg.get('section_title_font_size')
    if title_fs is None and text_cfg:
        title_fs = text_cfg.get('section_title_font_size') or text_cfg.get('title_font_size')

    # Poster header (global) sizes
    poster_title_fs = yaml_cfg.get('poster_title_font_size')
    poster_author_fs = yaml_cfg.get('poster_author_font_size')

    title_cfg = yaml_cfg.get('title') if isinstance(yaml_cfg.get('title'), dict) else None
    authors_cfg = yaml_cfg.get('authors') if isinstance(yaml_cfg.get('authors'), dict) else None

    if poster_title_fs is None and title_cfg:
        poster_title_fs = title_cfg.get('font_size') or title_cfg.get('title_font_size')
    if poster_author_fs is None and authors_cfg:
        poster_author_fs = authors_cfg.get('font_size') or authors_cfg.get('author_font_size')

    return bullet_fs, title_fs, poster_title_fs, poster_author_fs


def extract_colors(yaml_cfg: Dict[str, Any]) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """
    Extract color configurations from YAML config.

    Supports both flat and nested structures:
    - Flat: title_text_color, main_text_color, etc.
    - Nested: colors.title_text, colors.main_text, etc.

    Args:
        yaml_cfg: YAML configuration dictionary

    Returns:
        Tuple of (title_text_color, title_fill_color, main_text_color, main_text_fill_color)
    """
    title_text_color = None
    title_fill_color = None
    main_text_color = None
    main_text_fill_color = None

    if not isinstance(yaml_cfg, dict):
        return title_text_color, title_fill_color, main_text_color, main_text_fill_color

    # Load flat color configurations
    title_text_color = yaml_cfg.get('title_text_color')
    title_fill_color = yaml_cfg.get('title_fill_color')
    main_text_color = yaml_cfg.get('main_text_color')
    main_text_fill_color = yaml_cfg.get('main_text_fill_color')

    # Support nested color structure
    colors_cfg = yaml_cfg.get('colors') if isinstance(yaml_cfg.get('colors'), dict) else None
    if colors_cfg:
        if title_text_color is None:
            title_text_color = colors_cfg.get('title_text')
        if title_fill_color is None:
            title_fill_color = colors_cfg.get('title_fill')
        if main_text_color is None:
            main_text_color = colors_cfg.get('main_text')
        if main_text_fill_color is None:
            main_text_fill_color = colors_cfg.get('main_fill')

    return title_text_color, title_fill_color, main_text_color, main_text_fill_color


def extract_vertical_alignment(yaml_cfg: Dict[str, Any]) -> Optional[str]:
    """
    Extract vertical alignment configuration from YAML config.

    Args:
        yaml_cfg: YAML configuration dictionary

    Returns:
        Vertical alignment string ("top", "middle", or "bottom") or None
    """
    if not isinstance(yaml_cfg, dict):
        return None
    return yaml_cfg.get('section_title_vertical_align')


def extract_section_title_symbol(yaml_cfg: Dict[str, Any]) -> str:
    """
    Extract section title symbol configuration from YAML config.

    Args:
        yaml_cfg: YAML configuration dictionary

    Returns:
        Section title symbol string (default: empty string)
    """
    if not isinstance(yaml_cfg, dict):
        return ''
    return yaml_cfg.get('section_title_symbol', '')


def to_int_or_none(value: Any) -> Optional[int]:
    """
    Convert value to int or return None if conversion fails.

    Args:
        value: Value to convert

    Returns:
        Integer value or None
    """
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def to_color_tuple_or_none(value: Any) -> Optional[Tuple[int, int, int]]:
    """
    Convert YAML color list [R, G, B] to tuple (R, G, B).

    Args:
        value: Color value (list, tuple, or None)

    Returns:
        RGB tuple or None
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(c) for c in value)
        except (ValueError, TypeError):
            return None
    return None


def normalize_config_values(
    bullet_fs, title_fs, poster_title_fs, poster_author_fs,
    title_text_color, title_fill_color, main_text_color, main_text_fill_color
):
    """
    Normalize configuration values to proper types.

    Args:
        Font sizes and colors (raw from YAML)

    Returns:
        Tuple of normalized values
    """
    bullet_fs = to_int_or_none(bullet_fs)
    title_fs = to_int_or_none(title_fs)
    poster_title_fs = to_int_or_none(poster_title_fs)
    poster_author_fs = to_int_or_none(poster_author_fs)

    # Default: use bullet font size for title if not specified
    if title_fs is None:
        title_fs = bullet_fs

    title_text_color = to_color_tuple_or_none(title_text_color)
    title_fill_color = to_color_tuple_or_none(title_fill_color)
    main_text_color = to_color_tuple_or_none(main_text_color)
    main_text_fill_color = to_color_tuple_or_none(main_text_fill_color)

    return (
        bullet_fs, title_fs, poster_title_fs, poster_author_fs,
        title_text_color, title_fill_color, main_text_color, main_text_fill_color
    )
