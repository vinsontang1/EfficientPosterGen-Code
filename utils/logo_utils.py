"""
Logo management utilities for Paper2Poster project.
Handles searching, downloading, and retrieving logos for conferences and institutions.
Uses file-based matching - just drop PNG files in conferences/ or institutes/ folders.
"""

import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from PIL import Image
from io import BytesIO
import re
from difflib import SequenceMatcher
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogoManager:
    """Manages logo storage and retrieval using file-based matching."""

    def __init__(self, base_path: str = "logo_store"):
        """
        Initialize the LogoManager.

        Args:
            base_path: Base directory for logo storage
        """
        self.base_path = Path(base_path)
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for logo storage."""
        directories = [
            self.base_path,
            self.base_path / "conferences",
            self.base_path / "institutes",
            self.base_path / "raw_downloads"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for matching."""
        # Remove common suffixes like years
        name = re.sub(r'\s*\d{4}\s*$', '', name)
        # Convert to lowercase and replace spaces/special chars
        name = name.lower()
        name = re.sub(r'[^a-z0-9]+', '_', name)
        name = name.strip('_')
        return name

    def _fuzzy_match(self, query: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        """
        Find the best fuzzy match for a query among candidates.

        Args:
            query: The search query
            candidates: List of candidate strings

        Returns:
            Best matching candidate and similarity score (0-1)
        """
        query_norm = self._normalize_name(query)
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            # Check exact match first
            if query_norm == candidate:
                return candidate, 1.0

            # Check if query is contained in candidate or vice versa
            if query_norm in candidate or candidate in query_norm:
                score = 0.9
                if score > best_score:
                    best_match = candidate
                    best_score = score
                continue

            # Use sequence matching for fuzzy comparison
            score = SequenceMatcher(None, query_norm, candidate).ratio()
            if score > best_score:
                best_match = candidate
                best_score = score

        # Return match only if score is high enough
        if best_score >= 0.6:  # 60% similarity threshold
            return best_match, best_score

        return None, 0.0

    def _scan_directory(self, directory: Path) -> Dict[str, Path]:
        """
        Scan a directory for PNG files.

        Returns:
            Dictionary mapping normalized names to file paths
        """
        logos = {}
        if directory.exists():
            for file in directory.glob("*.png"):
                # Use the filename stem as the key
                name = file.stem.lower()
                logos[name] = file
        return logos

    def get_logo_path(self, name: str, category: str = "auto", use_google: bool = False) -> Optional[Path]:
        """
        Get the path to a logo file using fuzzy matching.

        Args:
            name: Name of the conference/institution
            category: Type of logo ("conference", "institute", or "auto")
            use_google: Whether to use Google Custom Search for web search

        Returns:
            Path to the logo file or None if not found
        """
        print(f"\n   🔍 Looking for logo: '{name}' (category: {category})")

        # Scan available logos
        conference_logos = self._scan_directory(self.base_path / "conferences")
        institute_logos = self._scan_directory(self.base_path / "institutes")

        # Determine which directories to search
        if category == "conference":
            search_dirs = [("conferences", conference_logos)]
            print(f"   📂 Searching in: conferences/ ({len(conference_logos)} logos)")
        elif category == "institute":
            search_dirs = [("institutes", institute_logos)]
            print(f"   📂 Searching in: institutes/ ({len(institute_logos)} logos)")
        else:  # auto
            search_dirs = [("conferences", conference_logos), ("institutes", institute_logos)]
            print(f"   📂 Searching in: conferences/ ({len(conference_logos)} logos), institutes/ ({len(institute_logos)} logos)")

        # Try to find best match
        best_match = None
        best_score = 0.0
        best_path = None
        best_dir = None

        for dir_name, logos in search_dirs:
            if logos:
                match, score = self._fuzzy_match(name, list(logos.keys()))
                if match and score > best_score:
                    best_match = match
                    best_score = score
                    best_path = logos[match]
                    best_dir = dir_name

        if best_path and best_path.exists():
            print(f"   ✅ MATCH FOUND: '{best_match}' in {best_dir}/ (similarity: {best_score:.1%})")
            print(f"   📄 File: {best_path.name}")
            return best_path

        # If no match found, try to download
        print(f"   ❌ No local match found (threshold: 60%)")
        print(f"   🌐 Attempting to download from web...")
        return self._download_and_save_logo(name, category, use_google=use_google)

    def _download_and_save_logo(self, name: str, category: str, use_google: bool = False) -> Optional[Path]:
        """
        Try to download a logo from the web and save it.

        Args:
            name: Name to search for
            category: Category for saving
            use_google: Whether to use Google Custom Search

        Returns:
            Path to downloaded logo or None
        """
        search_query = f"{name} logo"
        print(f"   🔎 Web search query: '{search_query}'")
        if use_google:
            print(f"   🌐 Using Google Custom Search API")
        url = self.search_logo_web(search_query, use_google=use_google)

        if not url:
            print(f"   ❌ No logo found online for: {name}")
            return None

        print(f"   🌐 Found URL: {url[:80]}...")

        # Determine save directory
        if category == "conference":
            save_dir = self.base_path / "conferences"
        else:
            save_dir = self.base_path / "institutes"

        # Generate filename
        filename = self._normalize_name(name) + ".png"
        save_path = save_dir / filename

        print(f"   💾 Downloading to: {save_path}")
        if self.download_logo(url, save_path):
            print(f"   ✅ Successfully downloaded and saved: {filename}")
            return save_path
        else:
            print(f"   ❌ Failed to download/convert logo")
            return None

    def search_logo_web(self, query: str, use_google: bool = False) -> Optional[str]:
        """
        Search for a logo on the web using DuckDuckGo or Google.

        Args:
            query: Search query
            use_google: Whether to use Google Custom Search (requires API key)

        Returns:
            URL of the found logo image or None
        """
        # Try DuckDuckGo first (no API key required)
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                # Search for images
                results = ddgs.images(
                    f"{query} official transparent PNG SVG",
                    max_results=5
                )

                # Filter for likely logo images
                for result in results:
                    url = result.get('image')
                    if url and any(ext in url.lower() for ext in ['.png', '.svg', '.jpg', '.jpeg']):
                        logger.info(f"Found potential logo: {url}")
                        return url

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        # Try Google Custom Search if enabled and API key is available
        if use_google:
            try:
                google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
                google_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

                if google_api_key and google_engine_id:
                    url = "https://www.googleapis.com/customsearch/v1"
                    params = {
                        'key': google_api_key,
                        'cx': google_engine_id,
                        'q': f"{query} official logo transparent PNG",
                        'searchType': 'image',
                        'num': 5,
                        'fileType': 'png|svg'
                    }

                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])
                        if items:
                            return items[0].get('link')
                else:
                    logger.warning("Google API keys not found in environment")

            except Exception as e:
                logger.warning(f"Google search failed: {e}")

        return None

    def download_logo(self, url: str, save_path: Path) -> bool:
        """
        Download a logo from a URL.

        Args:
            url: URL of the logo
            save_path: Path where to save the logo

        Returns:
            True if successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Save the file
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # If it's an SVG, try to convert to PNG
            if url.lower().endswith('.svg'):
                try:
                    import cairosvg
                    # Convert SVG to PNG
                    png_bytes = cairosvg.svg2png(bytestring=response.content, output_width=800)
                    img = Image.open(BytesIO(png_bytes))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.save(save_path, 'PNG', optimize=True)
                    logger.info(f"Converted SVG to PNG and saved to {save_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not convert SVG: {e}")
                    return False

            # If it's another image format, convert to PNG
            elif any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.gif', '.bmp', '.png']):
                try:
                    img = Image.open(BytesIO(response.content))
                    # Convert to RGBA for transparency support
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.save(save_path, 'PNG')
                    logger.info(f"Downloaded and saved logo to {save_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not process image: {e}")
                    return False

            else:
                logger.warning(f"Unsupported file format: {url}")
                return False

        except Exception as e:
            logger.error(f"Failed to download logo from {url}: {e}")
            return False

    def list_available_logos(self) -> Dict[str, List[str]]:
        """List all available logos in the system."""
        conference_logos = self._scan_directory(self.base_path / "conferences")
        institute_logos = self._scan_directory(self.base_path / "institutes")

        return {
            "conferences": sorted(conference_logos.keys()),
            "institutes": sorted(institute_logos.keys())
        }

    def extract_first_author_institution(self, paper_content: str) -> Optional[str]:
        """
        Extract the first author's institution from paper content.

        Args:
            paper_content: Text content of the paper (markdown format)

        Returns:
            First author's institution if found and matched with available logos
        """
        print("   📝 Looking for first author's institution...")

        # Look for authors section in the beginning of the paper
        lines = paper_content.split('\n')[:100]  # Focus on first 100 lines where authors usually appear

        # Common institution patterns
        institution_patterns = [
            r"(?:University of|University) [\w\s]+",
            r"[\w\s]+ University",
            r"[\w\s]+ Institute of Technology",
            r"[\w\s]+ Institute",
            r"MIT|CMU|UCLA|UCSD|NYU|ETH|EPFL|Stanford|Berkeley|Harvard|Princeton|Oxford|Cambridge",
            r"Google Research|DeepMind|Microsoft Research|Facebook AI Research|OpenAI|NVIDIA Research",
            r"Max Planck Institute",
            r"[\w\s]+ College",
            r"[\w\s]+ Research",
            r"[\w\s]+ Lab",
            r"[\w\s]+ Laboratory"
        ]

        all_pattern = '|'.join(f'({p})' for p in institution_patterns)

        # First pass: Look for the first line with superscript 1 (¹) which typically indicates first author affiliation
        first_institution = None
        for i, line in enumerate(lines):
            # Stop at abstract or introduction
            if 'abstract' in line.lower() or 'introduction' in line.lower():
                break

            # Look for lines with ¹ (first affiliation marker) at the beginning
            if '¹' in line:
                # Use finditer on each pattern individually to find all matches
                # This avoids the issue where re.findall() with alternation only finds non-overlapping matches
                all_matches = []
                for pattern_idx, pattern in enumerate(institution_patterns):
                    for match in re.finditer(pattern, line, re.IGNORECASE):
                        matched_text = match.group()
                        # Filter out very short matches (1-2 chars) but allow valid abbreviations (MIT, NYU, etc.)
                        if matched_text and len(matched_text.strip()) >= 3:
                            all_matches.append((match.start(), matched_text.strip(), pattern_idx))
                
                if all_matches:
                    # Prioritize pattern 0 (University of X) over pattern 1 (X University)
                    pattern0_matches = [m for m in all_matches if m[2] == 0]
                    if pattern0_matches:
                        first_institution = pattern0_matches[0][1]  # Take first pattern 0 match
                        print(f"   🎯 Found first author institution (from affiliation marker): {first_institution}")
                        break
                    # If no pattern 0, use other patterns but filter out author names
                    for start_pos, inst, pattern_idx in sorted(all_matches, key=lambda x: (x[2], x[0])):
                        if pattern_idx == 1:  # This is the "X University" pattern
                            # Skip if it looks like an author name (very short or contains common names)
                            inst_lower = inst.lower()
                            if len(inst.split()) <= 2 and any(name in inst_lower for name in ['chen', 'wang', 'li', 'zhang', 'smith', 'john']):
                                continue
                        first_institution = inst
                        print(f"   🎯 Found first author institution (from affiliation marker): {first_institution}")
                        break
            if first_institution:
                break

        # Second pass: If no superscript found, look for institution after the first author name
        if not first_institution:
            # Look for patterns like "Author Name (Institution)" or similar
            for i, line in enumerate(lines):
                if 'abstract' in line.lower() or 'introduction' in line.lower():
                    break

                # Skip title lines (usually the first few lines)
                if i < 2:
                    continue

                # Look for lines that might contain author + institution
                if '(' in line and ')' in line:
                    # Extract content in parentheses
                    paren_content = re.findall(r'\((.*?)\)', line)
                    for content in paren_content:
                        # Use finditer on each pattern individually to find all matches
                        all_matches = []
                        for pattern_idx, pattern in enumerate(institution_patterns):
                            for match in re.finditer(pattern, content, re.IGNORECASE):
                                matched_text = match.group()
                                # Filter out very short matches (1-2 chars) but allow valid abbreviations (MIT, NYU, etc.)
                                if matched_text and len(matched_text.strip()) >= 3:
                                    all_matches.append((match.start(), matched_text.strip(), pattern_idx))
                        
                        if all_matches:
                            # Prioritize pattern 0 (University of X) over pattern 1 (X University)
                            pattern0_matches = [m for m in all_matches if m[2] == 0]
                            if pattern0_matches:
                                first_institution = pattern0_matches[0][1]  # Take first pattern 0 match
                                print(f"   🎯 Found first author institution (from parentheses): {first_institution}")
                                break
                            # If no pattern 0, use other patterns but filter out author names
                            for start_pos, inst, pattern_idx in sorted(all_matches, key=lambda x: (x[2], x[0])):
                                if pattern_idx == 1:  # This is the "X University" pattern
                                    # Skip if it looks like an author name (very short or contains common names)
                                    inst_lower = inst.lower()
                                    if len(inst.split()) <= 2 and any(name in inst_lower for name in ['chen', 'wang', 'li', 'zhang', 'smith', 'john']):
                                        continue
                                first_institution = inst
                                print(f"   🎯 Found first author institution (from parentheses): {first_institution}")
                                break
                        if first_institution:
                            break
                if first_institution:
                    break

        # Third pass: If still nothing, just find the first institution mentioned
        if not first_institution:
            for line in lines[:30]:  # Only check first 30 lines for general search
                if 'abstract' in line.lower() or 'introduction' in line.lower():
                    break

                # Use finditer on each pattern individually to find all matches
                # This avoids the issue where re.findall() with alternation only finds non-overlapping matches
                all_matches = []
                for pattern_idx, pattern in enumerate(institution_patterns):
                    for match in re.finditer(pattern, line, re.IGNORECASE):
                        matched_text = match.group()
                        # Filter out very short matches (1-2 chars) but allow valid abbreviations (MIT, NYU, etc.)
                        if matched_text and len(matched_text.strip()) >= 3:
                            all_matches.append((match.start(), matched_text.strip(), pattern_idx))
                
                if all_matches:
                    # Prioritize pattern 0 (University of X) over pattern 1 (X University)
                    # First, try to find pattern 0 matches
                    pattern0_matches = [m for m in all_matches if m[2] == 0]
                    if pattern0_matches:
                        first_institution = pattern0_matches[0][1]  # Take first pattern 0 match
                        print(f"   🎯 Found institution (general search, pattern 0): {first_institution}")
                        break
                    
                    # If no pattern 0, use other patterns but filter out author names
                    for start_pos, inst, pattern_idx in sorted(all_matches, key=lambda x: (x[2], x[0])):
                        if pattern_idx == 1:  # This is the "X University" pattern
                            # Skip if it looks like an author name (very short or contains common names)
                            inst_lower = inst.lower()
                            if len(inst.split()) <= 2 and any(name in inst_lower for name in ['chen', 'wang', 'li', 'zhang', 'smith', 'john']):
                                continue
                        first_institution = inst
                        print(f"   🎯 Found institution (general search, pattern {pattern_idx}): {first_institution}")
                        break
                if first_institution:
                    break

        if not first_institution:
            print("   ❌ No institution found in author section")
            return None

        # Return the institution name regardless of whether there's a local match
        # The get_logo_path() method will handle downloading if not found locally
        print(f"   ✅ Extracted institution: '{first_institution}'")
        return first_institution

    def extract_institution_from_paper(self, paper_content: str) -> List[str]:
        """
        Extract institution names from paper content.
        DEPRECATED: Use extract_first_author_institution() instead for better accuracy.

        Args:
            paper_content: Text content of the paper

        Returns:
            List of detected institution names that match available logos
        """
        institutions = []

        # Common patterns for affiliations
        patterns = [
            r"(?:University of|University) [\w\s]+",
            r"[\w\s]+ University",
            r"[\w\s]+ Institute of Technology",
            r"[\w\s]+ Institute",
            r"MIT|CMU|UCLA|UCSD|NYU|ETH|EPFL|Stanford|Berkeley|Harvard|Princeton",
            r"Google Research|DeepMind|Microsoft Research|Facebook AI Research|OpenAI|NVIDIA Research",
            r"Max Planck Institute",
            r"[\w\s]+ College"
        ]

        # Extract potential institution names
        potential_institutions = []
        print("   📝 Searching for institution patterns...")
        for pattern in patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            if matches:
                potential_institutions.extend(matches)

        # Normalize and deduplicate
        potential_institutions = list(set([inst.strip() for inst in potential_institutions]))

        if potential_institutions:
            print(f"   🔎 Found {len(potential_institutions)} potential institutions in paper")
            # Show first 5 for brevity
            for i, inst in enumerate(potential_institutions[:5], 1):
                print(f"      {i}. {inst}")
            if len(potential_institutions) > 5:
                print(f"      ... and {len(potential_institutions) - 5} more")

        # Get available logos
        available_logos = self.list_available_logos()
        all_available = available_logos["institutes"]

        print(f"   🔍 Matching against {len(all_available)} available institute logos...")

        # Match against available logos using fuzzy matching
        matched_count = 0
        for inst in potential_institutions:
            match, score = self._fuzzy_match(inst, all_available)
            if match and score >= 0.7:  # Higher threshold for institution matching
                institutions.append(inst)
                matched_count += 1
                print(f"   ✅ MATCH: '{inst}' → '{match}' (similarity: {score:.1%})")

        if matched_count == 0 and potential_institutions:
            print(f"   ❌ No matches found (threshold: 70%)")

        return institutions


def main():
    """Example usage of LogoManager."""
    # Initialize manager
    manager = LogoManager()

    # List available logos
    available = manager.list_available_logos()
    print("Available logos:")
    for category, items in available.items():
        if items:
            print(f"\n{category}: {', '.join(items)}")

    # Example: Get a conference logo with fuzzy matching
    test_names = ["NeurIPS", "neurips 2024", "NIPS", "neural information"]
    for name in test_names:
        logo_path = manager.get_logo_path(name, "conference")
        if logo_path:
            print(f"\nLogo for '{name}' -> {logo_path}")

    # Example: Test institute matching
    test_institutes = ["MIT", "Massachusetts Institute of Technology", "Stanford University", "stanford"]
    for inst in test_institutes:
        logo_path = manager.get_logo_path(inst, "institute")
        if logo_path:
            print(f"\nLogo for '{inst}' -> {logo_path}")


def get_logo_dimensions(logo_path: str, target_height: float) -> Tuple[float, float]:
    """
    Calculate logo width to preserve aspect ratio.

    Args:
        logo_path: Path to logo image
        target_height: Desired height in inches

    Returns:
        Tuple of (width, height) in inches
    """
    try:
        with Image.open(logo_path) as img:
            aspect_ratio = img.width / img.height
            target_width = target_height * aspect_ratio
            return target_width, target_height
    except Exception:
        # Fallback to square if can't read image
        return target_height, target_height


def add_logos_to_poster_code(
    poster_code: str,
    width_inch: float,
    height_inch: float,
    institution_logo_path: Optional[str] = None,
    conference_logo_path: Optional[str] = None,
    logo_height: float = 2.0,
    logo_margin: float = 0.5
) -> str:
    """
    Add institution and conference logos to poster code.

    Args:
        poster_code: Existing poster generation code
        width_inch: Width of poster in inches
        height_inch: Height of poster in inches
        institution_logo_path: Path to institution logo (top-left)
        conference_logo_path: Path to conference logo (top-right)
        logo_height: Height of logos in inches (default: 2.0)
        logo_margin: Margin from edges in inches (default: 0.5)

    Returns:
        Modified poster code with logos added
    """
    import re

    logo_code = ""

    # Add institution logo to top-left
    if institution_logo_path and os.path.exists(institution_logo_path):
        inst_width, inst_height = get_logo_dimensions(institution_logo_path, logo_height)
        logo_code += f'''
# Add institution logo to top-left (aspect ratio preserved)
institution_logo = add_image(
    poster_slide,
    'institution_logo',
    {logo_margin},  # left
    {logo_margin},  # top
    {inst_width},   # width (calculated from aspect ratio)
    {inst_height},  # height (fixed)
    image_path="{institution_logo_path}"
)'''

    # Add conference logo to top-right
    if conference_logo_path and os.path.exists(conference_logo_path):
        conf_width, conf_height = get_logo_dimensions(conference_logo_path, logo_height)
        if logo_code:  # Add newline if there's already institution logo code
            logo_code += '\n'
        logo_code += f'''
# Add conference logo to top-right (aspect ratio preserved)
conference_logo = add_image(
    poster_slide,
    'conference_logo',
    {width_inch - conf_width - logo_margin},  # left (right-aligned with calculated width)
    {logo_margin},  # top
    {conf_width},   # width (calculated from aspect ratio)
    {conf_height},  # height (fixed)
    image_path="{conference_logo_path}"
)'''

    # Insert logo code before saving the presentation
    if logo_code:
        # Find the position to insert (before save_presentation)
        save_pos = poster_code.find('\n# Save the presentation')
        if save_pos != -1:
            # Insert before the newline that precedes "# Save the presentation"
            poster_code = poster_code[:save_pos] + '\n' + logo_code + poster_code[save_pos:]
        else:
            # Fallback: Find just the comment without newline
            save_pos = poster_code.find('# Save the presentation')
            if save_pos != -1:
                poster_code = poster_code[:save_pos] + logo_code + '\n\n' + poster_code[save_pos:]
            else:
                # Try alternative search pattern for save_presentation call
                pattern = r'(save_presentation\s*\([^)]+\))'
                match = re.search(pattern, poster_code)
                if match:
                    # Insert before the save_presentation call
                    insert_pos = match.start()
                    poster_code = poster_code[:insert_pos] + logo_code + '\n\n' + poster_code[insert_pos:]

    return poster_code


if __name__ == "__main__":
    main()