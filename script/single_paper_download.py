import os
import requests
import re
from urllib.parse import urlparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.mineru_client import MinerUClient
from config import config

# 校验配置
if not config:
    print("Error: Config file config.yaml not loaded or empty.")
    exit(1)

FILES_CONF = config['files']
miner_client = MinerUClient()

def sanitize_filename(title):
    clean_name = re.sub(r'[\\/*?:"<>|]', "", str(title))
    clean_name = clean_name.replace(" ", "_")
    return clean_name[:80]

def download_file(url, save_path, file_type="File"):
    if os.path.exists(save_path):
        print(f"   [{file_type}] Skipped (Already exists)")
        return True
    
    try:
        timeout = config['settings'].get('timeout', 30)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   [{file_type}] Download success")
            return True
        else:
            print(f"   [{file_type}] Download failed (Code: {response.status_code})")
            return False
    except Exception as e:
        print(f"   [{file_type}] Exception: {e}")
        return False

def clean_redundant_pdf(folder_path, keep_pdf_name):
    if not os.path.exists(folder_path):
        return

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            if file != keep_pdf_name:
                file_to_remove = os.path.join(folder_path, file)
                try:
                    os.remove(file_to_remove)
                    print(f"   [Cleanup] Removed redundant PDF: {file}")
                except OSError as e:
                    print(f"   [Cleanup] Error removing {file}: {e}")

def process_single_pdf(pdf_url, title, index_prefix="manual"):
    base_save_dir = FILES_CONF['save_dir']

    papers_out_dir = os.path.join(base_save_dir, "parsed_papers")
    os.makedirs(papers_out_dir, exist_ok=True)

    safe_title = sanitize_filename(title)
    
    base_name = f"{index_prefix}_{safe_title}"
    current_paper_dir = os.path.join(papers_out_dir, base_name)
    
    os.makedirs(current_paper_dir, exist_ok=True)
    
    print(f"\n--- Processing: {title} ---")
    print(f"   [Dir] {current_paper_dir}")

    target_pdf_name = f"{base_name}.pdf"
    target_pdf_path = os.path.join(current_paper_dir, target_pdf_name)

    download_success = download_file(pdf_url, target_pdf_path, file_type="PDF")
    
    if not download_success:
        print("   [Error] PDF download failed, aborting pipeline.")
        return

    # 2. Run API Parsing
    if any(f.endswith('.md') for f in os.listdir(current_paper_dir)):
        print(f"   [API] Skipped (Already parsed)")
        clean_redundant_pdf(current_paper_dir, target_pdf_name)
        return

    try:
        print(f"   [API] calling MinerU pipeline...")
        miner_client.process_pipeline(pdf_url, current_paper_dir)
        
        # 3. Cleanup
        clean_redundant_pdf(current_paper_dir, target_pdf_name)
        print("   [Done] Processing complete.")
        
    except Exception as e:
        print(f"   [API] Failed: {e}")

if __name__ == "__main__":
    
    TARGET_URL = "" 
    
    TARGET_TITLE = ""
    
    process_single_pdf(TARGET_URL, TARGET_TITLE)