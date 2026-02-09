import os
import sys
import re
import logging
import json 
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image  

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mineru_client import MinerUClient
from config import config

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "paper_download.log"

logger = logging.getLogger("paper_downloader")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

if not config:
    logger.error("Config file config.yaml not loaded or empty.")
    sys.exit(1)

FILES_CONF = config["files"]
miner_client = MinerUClient()

def sanitize_filename(title: str) -> str:
    clean_name = re.sub(r'[\\/*?:"<>|]', "", str(title))
    clean_name = clean_name.replace(" ", "_")
    return clean_name[:80]

def download_file(url, save_path, file_type="File"):
    if os.path.exists(save_path):
        logger.info(f"[{file_type}] Skipped (Already exists)")
        return True

    try:
        timeout = config["settings"].get("timeout", 30)
        
        parsed_uri = urlparse(url)
        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Referer": domain, 
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        response = requests.get(url, headers=headers, stream=True, timeout=timeout, verify=False)
        
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"[{file_type}] Download success")
            return True
        else:
            logger.warning(
                f"[{file_type}] Download failed (HTTP {response.status_code}) | URL: {url}"
            )
            return False

    except Exception as e:
        logger.error(f"[{file_type}] Exception: {e}")
        return False

def clean_redundant_pdf(folder_path, keep_pdf_name):
    if not os.path.exists(folder_path):
        return

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf") and file != keep_pdf_name:
            file_to_remove = os.path.join(folder_path, file)
            try:
                os.remove(file_to_remove)
                logger.info(f"[Cleanup] Removed redundant PDF from server result: {file}")
            except OSError as e:
                logger.error(f"[Cleanup] Error removing {file}: {e}")

def generate_meta_json(poster_path, meta_path):
    if os.path.exists(meta_path):
        return

    if not os.path.exists(poster_path):
        logger.warning(f"[Meta] Poster not found at {poster_path}, skipping meta generation.")
        return

    try:
        with Image.open(poster_path) as img:
            width, height = img.size
        
        metadata = {
            'width': width,
            'height': height
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"[Meta] Generated meta.json for {os.path.basename(os.path.dirname(poster_path))}")
    except Exception as e:
        logger.error(f"[Meta] Failed to generate meta.json: {e}")

def paper_download():
    base_save_dir = FILES_CONF["save_dir"]
    parquet_path = FILES_CONF["parquet_path"]

    papers_out_dir = os.path.join(base_save_dir, "parsed_papers")
    os.makedirs(papers_out_dir, exist_ok=True)

    logger.info(f"Loading index file: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Error reading Parquet file: {e}")
        return

    total = len(df)
    logger.info(f"Found {total} records")

    for index, row in df.iterrows():
        title = row["title"]
        paper_url = row["paper_url"]
        image_url = row["image_url"]
        
        qa_data = row.get("qa", None)

        safe_title = sanitize_filename(title)
        base_name = f"{index}_{safe_title}"
        current_paper_dir = os.path.join(papers_out_dir, base_name)
        os.makedirs(current_paper_dir, exist_ok=True)

        logger.info(f"[{index + 1}/{total}] {title}")

        # Paths
        target_poster_path = os.path.join(current_paper_dir, "poster.png")
        target_pdf_name = f"{base_name}.pdf"
        target_pdf_path = os.path.join(current_paper_dir, target_pdf_name)
        target_qa_path = os.path.join(current_paper_dir, "o3_qa.json")  
        target_meta_path = os.path.join(current_paper_dir, "meta.json") 

        if qa_data is not None and not os.path.exists(target_qa_path):
            try:
                if isinstance(qa_data, str):
                    qa_content = json.loads(qa_data)
                else:
                    qa_content = qa_data 

                with open(target_qa_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_content, f, indent=4, ensure_ascii=False)
                logger.info("[QA] Saved o3_qa.json")
            except Exception as e:
                logger.error(f"[QA] Failed to save JSON: {e}")
        elif os.path.exists(target_qa_path):
            logger.info("[QA] Skipped (Already exists)")
        else:
            logger.warning("[QA] No 'qa' column data found for this row.")

        # 下载文件
        poster_downloaded = download_file(image_url, target_poster_path, file_type="Poster")
        pdf_downloaded = download_file(paper_url, target_pdf_path, file_type="PDF")

        if os.path.exists(target_poster_path):
            generate_meta_json(target_poster_path, target_meta_path)

        # Check if already parsed
        if any(f.endswith(".md") for f in os.listdir(current_paper_dir)):
            logger.info("[API] Skipped (Already parsed)")
            clean_redundant_pdf(current_paper_dir, target_pdf_name)
            continue

        local_file_arg = target_pdf_path if pdf_downloaded else None

        try:
            miner_client.process_pipeline(paper_url, current_paper_dir, local_file_path=local_file_arg)
            logger.info("[API] Parse success")
            
            clean_redundant_pdf(current_paper_dir, target_pdf_name)

        except Exception as e:
            logger.error(f"[API] Failed: {e}")

if __name__ == "__main__":
    paper_download()