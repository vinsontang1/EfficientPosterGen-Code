"""
Theme utilities for managing poster visual themes.

This module handles theme configuration, including colors, borders,
and visual styling options.
"""

from typing import Dict, Any, Optional


# Default theme configuration
DEFAULT_THEME = {
    'panel_visible': True,
    'textbox_visible': False,
    'figure_visible': False,
    'panel_theme': {
        'color': (47, 85, 151),
        'thickness': 0,
        'line_style': 'solid',
    },
    'textbox_theme': None,
    'figure_theme': None,
}

# Default colors
DEFAULT_TITLE_TEXT_COLOR = (255, 255, 255)
DEFAULT_TITLE_FILL_COLOR = (255, 255, 255)


def get_default_theme() -> Dict[str, Any]:
    """
    Get the default theme configuration.

    Returns:
        Default theme dictionary
    """
    return DEFAULT_THEME.copy()


def create_theme_with_alignment(
    base_theme: Dict[str, Any],
    section_title_vertical_align: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new theme with vertical alignment configuration.

    Args:
        base_theme: Base theme dictionary
        section_title_vertical_align: Vertical alignment ("top", "middle", "bottom")

    Returns:
        Theme dictionary with alignment configuration
    """
    theme = base_theme.copy()
    if section_title_vertical_align:
        theme['section_title_vertical_align'] = section_title_vertical_align
    return theme


def resolve_colors(
    title_text_color: Optional[tuple],
    title_fill_color: Optional[tuple],
    main_text_color: Optional[tuple],
    main_text_fill_color: Optional[tuple]
) -> tuple:
    """
    Resolve color configuration with fallbacks to defaults.

    Args:
        title_text_color: Title text color from config
        title_fill_color: Title fill color from config
        main_text_color: Main text color from config
        main_text_fill_color: Main text fill color from config

    Returns:
        Tuple of (title_text, title_fill, main_text, main_text_fill)
    """
    final_title_text = title_text_color if title_text_color is not None else DEFAULT_TITLE_TEXT_COLOR
    final_title_fill = title_fill_color if title_fill_color is not None else DEFAULT_TITLE_FILL_COLOR
    final_main_text = main_text_color
    final_main_fill = main_text_fill_color

    return final_title_text, final_title_fill, final_main_text, final_main_fill
