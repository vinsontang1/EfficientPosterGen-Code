import json, time, csv
import logging
import os, re
import argparse
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
from src.utils.parser import DocumentParser, find_matching_pdf
from src.utils.grouper import group_sequence_by_ppl_12
from src.utils.ppl_engine import PPLEngine
from src.utils.tree_walker import TreeWalker
from src.utils.graph_builder import DependencyGraphBuilder
from src.utils.graph_rank import GraphRankSelector
from config import config
from src.utils.word2png import text_to_images
from collections import defaultdict
from src.utils.bullet_points import summarize_single_paper, Agent
from src.utils.agent_utils import ModelFactory
from src.utils.image_matcher import get_matched_assets
from src.utils.extract_auth import extract_paper_poster_meta
from src.utils.agent_sec2pic import match_sections_to_pics
from contextlib import contextmanager
            
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

@contextmanager
def timer(logger, task_name: str, paper_id: str = "N/A", csv_path: str = None):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        prefix = f"[{paper_id}] " if paper_id != "N/A" else ""
        logger.info(f"{prefix}Task '{task_name}' finished in {elapsed:.4f}s")
        
        if csv_path:
            file_exists = os.path.isfile(csv_path)
            try:
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['PaperID', 'TaskName', 'Duration_Seconds', 'Timestamp'])
                
                    writer.writerow([paper_id, task_name, f"{elapsed:.4f}", time.strftime("%Y-%m-%d %H:%M:%S")])
            except Exception as e:
                logger.error(f"Failed to write to CSV: {e}")

def _accum(dst: dict, inc: dict):
    if not isinstance(inc, dict):
        return
    for k in ("input_text", "input_image", "input_total", "output"):
        dst[k] += int(inc.get(k, 0) or 0)

def normalize_text(text: str, max_len: int = 200) -> str:
    return " ".join(text.strip().split())[:max_len]


def build_md_order_index(md_path: Path) -> Dict[str, int]:
    if not md_path.exists():
        logger.warning(f"Markdown file not found: {md_path}")
        return {}

    order_map: Dict[str, int] = {}
    idx = 0

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            norm = normalize_text(line)
            if norm and norm not in order_map:
                order_map[norm] = idx
                idx += 1

    logger.info(f"Built md order index with {len(order_map)} entries.")
    return order_map


def get_md_order_for_node(node: Dict, md_order_map: Dict[str, int]) -> int:
    text = node.get("context", "")
    if not text:
        return 10**9

    norm = normalize_text(text)
    return md_order_map.get(norm, 10**9)

def sanitize(name: str) -> str:
    """用于文件夹名安全化"""
    return re.sub(r'[^\w\-\.]', '_', name).strip('_')

def save_stage_output(
    data: dict,
    stage_name: str,
    output_root: str,
    paper_id: str,
    skip_if_exists: bool = True
):
    """
    保存 paper 在某个 stage 的输出
    """
    stage_dir = os.path.join(output_root, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    safe_paper_id = sanitize(paper_id)
    output_path = os.path.join(stage_dir, f"{safe_paper_id}.json")

    if skip_if_exists and os.path.exists(output_path):
        logger.info(
            f"[Stage Skipped] {stage_name} / {paper_id} already exists"
        )
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[Stage Saved] {stage_name} / {paper_id} -> {output_path}"
    )



def parse_stage_stream(input_dir_str: str):
    """
    解析 MD + PDF
    """
    input_path = Path(input_dir_str)
    parser = DocumentParser()

    logger.info("Starting parse processing from: %s", input_path)

    for md_file in input_path.rglob("*.md"):
        current_dir = md_file.parent
        paper_id = current_dir.name

        logger.info("Processing document: %s", paper_id)

        try:
            pdf_file = find_matching_pdf(current_dir)

            if pdf_file:
                logger.info("Using PDF: %s", pdf_file.name)
            else:
                logger.info("Using Markdown structure only")

            result = parser.parse(md_file, pdf_file)

            yield paper_id, result

        except Exception:
            logger.exception(
                "Error processing Markdown file: %s",
                md_file
            )
            continue



def grouping(
    paper_id: str,
    paper_data: dict,
    engine
):
    """
    对单篇 paper 进行 PPL-based grouping
    """
    logger.info(f"Grouping Paper: {paper_id}")

    structure = paper_data.get("structure", [])
    content_map = paper_data.get("content", {})

    paper_groups = []

    # 遍历 Level 1 节点
    for node in structure:
        if node.get("level") != 1:
            continue

        section_title = node.get("title", "Untitled")

        sequence = TreeWalker.extract_sequence_from_section(
            node,
            content_map,
            [config['tree']['ROOT_UUID']]
        )

        if not sequence:
            continue

        logger.info(
            f"  > Grouping Section: {section_title} "
            f"(Total paragraphs: {len(sequence)})"
        )

        grouped_sequence = group_sequence_by_ppl_12(
            engine,
            sequence
        )

        paper_groups.append({
            "section_root_title": section_title,
            "groups": grouped_sequence
        })

    return paper_groups


def graphing(
    paper_id: str,
    grouped_sections: list,
    threshold: float,
    engine
):
    """
    基于 grouped sections 构建有向图
    """
    graph_builder = DependencyGraphBuilder(
        ppl_engine=engine,
        threshold=threshold,
        min_edge_count=config['graph']['min_edge_count']
    )

    graph_result = graph_builder.compute_graph(
        paper_id,
        grouped_sections
    )

    return graph_result


def selection_and_render(
    paper_id: str,
    paper_graph: dict,
    output_root: str,
    config_path: str,
    threshold: float
):
    """
    做 selection + render
    """
    selector = GraphRankSelector(
        alpha=config['selection']['alpha'],
        beta=config['selection']['beta'],
        min_count=config['selection']['min_count'],
        k=config['selection']['pagerank_k']
    )

    logger.info(f"Processing paper: {paper_id}")

    selected_nodes = selector.select_nodes(
        paper_graph,
        threshold=threshold,
    )

    if not selected_nodes:
        logger.warning(f"No nodes selected for {paper_id}")
        return {
            "paper_id": paper_id,
            "selected_nodes": []
        }
    
    md_path = (
        Path(config['files']['save_dir'])
        / "parsed_papers"
        / paper_id
        / "full.md"
    )

    md_order_map = build_md_order_index(md_path)


    # 按 section_title 聚合
    section_groups = defaultdict(list)
    for node in selected_nodes:
        node["_md_order"] = get_md_order_for_node(node, md_order_map)
        section_groups[node["section_title"]].append(node)

    for section_title, nodes in section_groups.items():
        nodes.sort(key=lambda x: x["_md_order"])

        for n in nodes:
            n.pop("_md_order", None)

    render_sections_to_images(
        paper_id=paper_id,
        section_groups=section_groups,
        output_root=output_root,
        config_path=config_path
    )

    # 数据返回
    return {
        "paper_id": paper_id,
        "selected_nodes": selected_nodes
    }


def render_sections_to_images(
    paper_id: str,
    section_groups: dict,
    output_root: str,
    config_path: str
):
    """
    将按 section 分组后的节点内容渲染为 PNG
    """
    for section_title, nodes in section_groups.items():
        logger.info(f"  Rendering section: {section_title}")

        full_text = (
            section_title
            + "\n\n"
            + "\n\n".join(
                n["context"] for n in nodes if n["context"].strip()
            )
        )

        if not full_text.strip():
            continue

        paper_dir = os.path.join(output_root, sanitize(paper_id))
        section_name = sanitize(section_title)

        text_to_images(
            text=full_text,
            output_dir=paper_dir,
            config_path=config_path,
            unique_id=section_name
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Poster Generation Pipeline with Logo Support')
    parser.add_argument('--poster_path', type=str, default='./test_dataset')
    parser.add_argument('--output_root', type=str, default='./output')
    parser.add_argument('--summ_provider', type=str, default='openkey', choices=['openai', 'openkey', 'siliconflow', 'google', "dashscope", "aliyun", "bailian"])
    parser.add_argument('--summ_model', type=str, default='gpt-5-2025-08-07')
    parser.add_argument('--gpu_id', type=int, default='1')
    args = parser.parse_args()
    
    TIME_LOG_PATH = os.path.join(args.output_root, "time_log.csv")
    logger.info(f"Time logs will be saved to: {TIME_LOG_PATH}")
    PNG_CONFIG_PATH = config['png']['CONFIG_PATH']
    
    sec2pic_tokens = {"input_text": 0, "input_image": 0, "input_total": 0, "output": 0}

    graphing_threshold = config['graph']['graph_threshold']
    selection_threshold = config['selection']['selection_threshold']

    engine = PPLEngine(config=config, args=args)
    print("PPLEngine initialized.")
    
    logger.info(f"Initializing Agent [{args.summ_provider} : {args.summ_model}]...")
    try:
        agent_config = ModelFactory.create_config(
            provider=args.summ_provider,
            model_name=args.summ_model
        )
        summary_agent = Agent(agent_config)
        
        agent_sec2pic_config = ModelFactory.create_config(
            provider="openkey",
            model_name="gpt-4o"
        )
        sec2pic_agent = Agent(agent_sec2pic_config)
        
        agent_authors_config = ModelFactory.create_config(
            provider="openkey",
            model_name="qwen3-1.7b"
        )
        extract_authors_agent = Agent(agent_authors_config)
        logger.info("Agent initialized successfully.")
    except RuntimeError as e:
        logger.critical(f"API initialization failed: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during agent initialization: {e}")
        exit(1)

    logger.info("Starting streaming pipeline...")

    # paper-level streaming
    with timer(logger, "Global_Pipeline_Execution", paper_id="ALL_PAPERS", csv_path=TIME_LOG_PATH):
        for paper_id, paper_data in parse_stage_stream(args.poster_path):
            with timer(logger, "Total_Single_Paper", paper_id=paper_id, csv_path=TIME_LOG_PATH):
                try:
                    paper_id2 = sanitize(paper_id)
                    meta = extract_paper_poster_meta(summary_agent, input_dir=args.poster_path, paper_id=paper_id)
                    with timer(logger, "00_meta", paper_id, csv_path=TIME_LOG_PATH):
                        save_stage_output(
                            data=meta,
                            stage_name="00_meta",
                            output_root=args.output_root,
                            paper_id=paper_id
                        )
                    # 解析 tree 和 clean
                    save_stage_output(
                        data=paper_data,
                        stage_name="01_parse",
                        output_root=args.output_root,
                        paper_id=paper_id
                    )
                    with timer(logger, "sec2pic", paper_id, csv_path=TIME_LOG_PATH):
                        res = match_sections_to_pics(
                            content_dir=str(Path(args.poster_path) / str(paper_id)),
                            structure_json_path= str(Path(args.output_root) / "01_parse" / f"{paper_id2}.json"),
                            out_path= str(Path("sec2pic") / f"{paper_id2}.json"),
                            target_level=1,
                            max_page_idx=8,
                            agent=sec2pic_agent
                            )
                        paper_tokens = {}
                        if isinstance(res, dict):
                            paper_tokens = res.get("__token_usage__", {}) or {}
                        _accum(sec2pic_tokens, paper_tokens)

                    # 段落分组
                    with timer(logger, "02_grouping", paper_id, csv_path=TIME_LOG_PATH):
                        grouped = grouping(
                            paper_id=paper_id,
                            paper_data=paper_data,
                            engine=engine
                        )
                        save_stage_output(
                            data=grouped,
                            stage_name="02_grouping",
                            output_root=args.output_root,
                            paper_id=paper_id
                        )

                    # 构建有向图
                    with timer(logger, "03_graph", paper_id, csv_path=TIME_LOG_PATH):
                        graph = graphing(
                            paper_id=paper_id,
                            grouped_sections=grouped,
                            threshold=graphing_threshold,
                            engine=engine
                        )
                        save_stage_output(
                            data=graph,
                            stage_name="03_graph",
                            output_root=args.output_root,
                            paper_id=paper_id
                        )

                    # pagerank + 选择 + word2png
                    with timer(logger, "04_selection", paper_id, csv_path=TIME_LOG_PATH):
                        selection = selection_and_render(
                            paper_id=paper_id,
                            paper_graph=graph,
                            output_root=args.output_root,
                            config_path=PNG_CONFIG_PATH,
                            threshold=selection_threshold
                        )
                        save_stage_output(
                            data=selection,
                            stage_name="04_selection",
                            output_root=args.output_root,
                            paper_id=paper_id
                        )
                    with timer(logger, "05_summary", paper_id, csv_path=TIME_LOG_PATH):
                        if selection and selection.get("selected_nodes"):
                            summary_result = summarize_single_paper(
                                agent=summary_agent,
                                output_root=args.output_root, 
                                paper_id=paper_id
                            )
                            if summary_result:
                                save_stage_output(
                                    data=summary_result,
                                    stage_name="05_summary", 
                                    output_root=args.output_root,
                                    paper_id=paper_id
                                )
                            else:
                                logger.warning(f"Summarization yielded no content for {paper_id}")

                except Exception:
                    logger.exception(f"Pipeline failed for paper: {paper_id}")
                    continue

    logger.info("Streaming pipeline completed successfully.")
    logger.info(f"[TokenUsage][ALL_PAPERS] {sec2pic_tokens}")

