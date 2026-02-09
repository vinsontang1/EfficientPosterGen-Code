import uuid
import re
import logging
from typing import Optional
from pathlib import Path
from pypdf import PdfReader
from src.utils.cleaner import MarkdownCleaner
from config import config

logger = logging.getLogger(__name__)

import re
import uuid

class DocumentParser:
    def __init__(self):
        self.cleaner = MarkdownCleaner()

    def _create_node(self, title: str, level: int) -> dict:
        """Create a standard tree node."""
        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "level": level,
            "children": [],
            "paragraph_ids": []
        }

    def _merge_short_paragraphs(
        self,
        node,
        content_by_id,
        min_alpha_chars=config['parser']['MIN_ALPHA_CHARS']
    ):
        para_ids = node["paragraph_ids"]
        if not para_ids:
            return

        def alpha_len(text: str) -> int:
            return sum(c.isalpha() for c in text)

        new_para_ids = []
        i = 0

        while i < len(para_ids):
            pid = para_ids[i]
            text = content_by_id[pid]

            if alpha_len(text) >= min_alpha_chars:
                new_para_ids.append(pid)
                i += 1
                continue

            # short paragraph
            if not new_para_ids:
                # first paragraph → merge with next
                if i + 1 < len(para_ids):
                    next_pid = para_ids[i + 1]
                    content_by_id[next_pid] = (
                        text.rstrip() + " " + content_by_id[next_pid].lstrip()
                    )
                    content_by_id.pop(pid, None)
                else:
                    new_para_ids.append(pid)
            else:
                # merge with previous
                prev_pid = new_para_ids[-1]
                content_by_id[prev_pid] = (
                    content_by_id[prev_pid].rstrip() + " " + text.lstrip()
                )
                content_by_id.pop(pid, None)

            i += 1

        node["paragraph_ids"] = new_para_ids

    def _get_pdf_outlines(self, pdf_path):
        """Extract PDF outlines (bookmarks) and build an initial tree."""
        try:
            reader = PdfReader(pdf_path)
            if not reader.outline:
                return None

            def parse_items(items, level=1):
                nodes = []
                for item in items:
                    if isinstance(item, list):
                        if nodes:
                            nodes[-1]["children"].extend(
                                parse_items(item, level + 1)
                            )
                    else:
                        title = getattr(item, "title", "Untitled")
                        nodes.append(self._create_node(title, level))
                return nodes

            return parse_items(reader.outline)

        except Exception as exc:
            logger.warning(f"PDF outline extraction failed: {exc}")
            return None

    def _find_node_by_title(self, nodes, target_title):
        def normalize(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", text.lower())

        target_clean = normalize(target_title)
        if not target_clean:
            return None

        for node in nodes:
            node_clean = normalize(node["title"])
            if target_clean in node_clean or node_clean in target_clean:
                return node

            found = self._find_node_by_title(node["children"], target_title)
            if found:
                return found
        return None

    def _get_node_by_md_level(self, root_nodes, level, stack):
        new_node = self._create_node("New Section", level)

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if stack:
            stack[-1]["children"].append(new_node)
        else:
            root_nodes.append(new_node)

        stack.append(new_node)
        return new_node

    def _shift_levels(self, node: dict, delta: int):
        node["level"] = max(1, node["level"] + delta)
        for ch in node.get("children", []):
            self._shift_levels(ch, delta)

    def _detach_abstract_children_to_roots(self, tree_roots: list, abstract_node: dict):
        """Enforce: abstract_node['children'] must be empty.
        Move its children to tree_roots as independent roots, and normalize levels.
        """
        if not abstract_node:
            return
        children = abstract_node.get("children", [])
        if not children:
            return

        abstract_node["children"] = []

        for child in children:
            delta = 1 - child.get("level", 1)
            self._shift_levels(child, delta)
            tree_roots.append(child)

    def parse(self, md_path, pdf_path):
        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        text = self.cleaner.remove_head_sections(raw_text)
        text = self.cleaner.remove_html_tables(text)
        text = self.cleaner.remove_tail_sections(text)

        lines = text.split("\n")
        garbage_mask = self.cleaner.pre_mark_garbage(lines)

        tree_roots = []
        pdf_tree = None

        if pdf_path and pdf_path.exists():
            pdf_tree = self._get_pdf_outlines(pdf_path)
            if pdf_tree:
                tree_roots = pdf_tree
                logger.info("Using PDF structure.")
            else:
                logger.info("Using Markdown structure (PDF outline unavailable).")

        content_by_id = {}

        abstract_node = self._find_node_by_title(tree_roots, "Abstract")
        if not abstract_node:
            abstract_node = self._create_node("Abstract", 1)
            tree_roots.insert(0, abstract_node)

        self._detach_abstract_children_to_roots(tree_roots, abstract_node)

        current_node = abstract_node
        node_stack = []  
        paragraph_buffer = ""

        # Line-by-line parsing
        for idx, line in enumerate(lines):
            if garbage_mask[idx]:
                continue

            line = self.cleaner.clean_inline_images(line)
            stripped = line.strip()

            header_match = re.match(r"^(#+)\s*(.*)", line)

            if header_match:
                if paragraph_buffer:
                    paragraph_id = str(uuid.uuid4())
                    content_by_id[paragraph_id] = paragraph_buffer
                    current_node["paragraph_ids"].append(paragraph_id)
                    paragraph_buffer = ""

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                if pdf_tree:
                    found_node = self._find_node_by_title(tree_roots, title)
                    if found_node:
                        current_node = found_node
                        current_node["title"] = title

                        self._detach_abstract_children_to_roots(tree_roots, abstract_node)
                        continue
                    else:
                        stripped = title
                else:
                    if title.strip().lower() == "abstract":
                        current_node = abstract_node
                        node_stack.clear()
                        continue

                    num_match = re.match(r"^(\d+(?:\.\d+)*)", title)
                    if num_match:
                        num_str = num_match.group(1)
                        level = len(num_str.split("."))

                    if not node_stack and level > 1:
                        level = 1

                    current_node = self._get_node_by_md_level(
                        tree_roots, level, node_stack
                    )
                    current_node["title"] = title
                    continue

            if not stripped:
                continue

            # Paragraph merging logic
            if not paragraph_buffer:
                paragraph_buffer = stripped
            else:
                if self.cleaner.should_merge(paragraph_buffer, line):
                    if paragraph_buffer.endswith("-"):
                        paragraph_buffer = paragraph_buffer[:-1] + stripped
                    else:
                        paragraph_buffer += " " + stripped
                else:
                    paragraph_id = str(uuid.uuid4())
                    content_by_id[paragraph_id] = paragraph_buffer
                    current_node["paragraph_ids"].append(paragraph_id)
                    paragraph_buffer = stripped

        if paragraph_buffer:
            paragraph_id = str(uuid.uuid4())
            content_by_id[paragraph_id] = paragraph_buffer
            current_node["paragraph_ids"].append(paragraph_id)

        self._detach_abstract_children_to_roots(tree_roots, abstract_node)

        def process_node(node):
            self._merge_short_paragraphs(node, content_by_id)
            for child in node["children"]:
                process_node(child)

        for root in tree_roots:
            process_node(root)

        return {
            "structure": tree_roots,
            "content": content_by_id
        }


def find_matching_pdf(directory: Path) -> Optional[Path]:
    """
    在指定目录中寻找匹配的 PDF 文件

    策略：
    1. 优先寻找与目录同名的 PDF
    2. 否则，若目录下只有一个 PDF，则使用该 PDF
    3. 多个或没有 PDF 时返回 None
    """
    # 策略 1：同名 PDF
    named_pdf = directory / f"{directory.name}.pdf"
    if named_pdf.exists():
        return named_pdf

    # 策略 2：唯一 PDF
    pdf_files = list(directory.glob("*.pdf"))
    if len(pdf_files) == 1:
        return pdf_files[0]

    if len(pdf_files) > 1:
        logger.warning(
            "Skipping directory '%s': multiple PDFs found, unable to determine match",
            directory.name
        )
    else:
        logger.warning(
            "No PDF found in directory '%s', falling back to MD-only parsing",
            directory.name
        )

    return None