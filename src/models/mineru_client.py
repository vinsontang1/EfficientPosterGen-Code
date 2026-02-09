import requests
import time
import os
import zipfile
import shutil
from pathlib import Path
from config import config

class MinerUClient:
    def __init__(self):
        self.conf = config['mineru']
        self.settings = config['settings']
        
        token = self.conf['api_token']
        if not token.startswith("Bearer"):
            token = f"Bearer {token}"
            
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": token
        }

    def submit_task(self, pdf_url):
        data = {
            "url": pdf_url, 
            "model_version": self.conf['model_version']
        }
        try:
            res = requests.post(
                self.conf['base_url'], 
                headers=self.headers, 
                json=data, 
                timeout=self.settings['timeout']
            )
            res_json = res.json()
            if res_json.get("code") != 0:
                raise Exception(f"API Error: {res_json}")
            return res_json["data"]["task_id"]
        except Exception as e:
            raise Exception(f"Task submission failed: {str(e)}")

    def wait_for_completion(self, task_id):
        check_url = f"{self.conf['base_url']}/{task_id}"
        return self._poll_status(check_url, is_batch=False)

    def upload_file(self, file_path):
        file_path = Path(file_path)
        file_name = file_path.name
        
        batch_url = "https://mineru.net/api/v4/file-urls/batch"
        payload = {
            "files": [{"name": file_name, "data_id": "fallback_upload"}],
            "model_version": self.conf['model_version']
        }
        
        try:
            res = requests.post(batch_url, headers=self.headers, json=payload, timeout=30)
            res_json = res.json()
            if res_json.get("code") != 0:
                raise Exception(f"Get upload url failed: {res_json}")
            
            batch_id = res_json["data"]["batch_id"]
            upload_url = res_json["data"]["file_urls"][0]
            
            print(f"   [Fallback] Got upload URL (Batch ID: {batch_id}). Uploading...")

            with open(file_path, "rb") as f:
                requests.put(upload_url, data=f, timeout=300).raise_for_status()
            
            print(f"   [Fallback] File uploaded successfully!")
            return batch_id
        except Exception as e:
            raise Exception(f"Upload process failed: {e}")

    def wait_for_batch_completion(self, batch_id):
        check_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        return self._poll_status(check_url, is_batch=True)

    def _poll_status(self, url, is_batch=False):
        start_time = time.time()
        while True:
            if time.time() - start_time > self.settings['max_wait_time']:
                raise TimeoutError("Task processing timeout")

            try:
                res = requests.get(url, headers=self.headers, timeout=self.settings['timeout'])
                res_data = res.json()
                
                if is_batch:
                    extract_res = res_data.get("data", {}).get("extract_result", [])
                    if not extract_res:
                        time.sleep(self.settings['polling_interval'])
                        continue
                    item = extract_res[0] 
                    state = item.get("state")
                    download_url = item.get("full_zip_url")
                    err_msg = item.get("err_msg")
                else:
                    data = res_data.get("data", {})
                    state = data.get("state") or data.get("status")
                    download_url = data.get("full_zip_url")
                    err_msg = data.get("err_msg")

                if state == "done":
                    if not download_url:
                        raise Exception("Status is done but no download URL found.")
                    return download_url
                elif state in ["error", "failed"]:
                    raise Exception(f"Server Task Failed: {err_msg}")
                
                time.sleep(self.settings['polling_interval'])

            except Exception as e:
                if "Server Task Failed" in str(e):
                    raise e
                time.sleep(self.settings['polling_interval'])

    def download_and_extract(self, download_url, extract_to_folder):
        if not os.path.exists(extract_to_folder):
            os.makedirs(extract_to_folder)
        zip_path = os.path.join(extract_to_folder, "temp.zip")
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_to_folder)
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def process_pipeline(self, pdf_url, output_dir, local_file_path=None):
        zip_url = None
        
        try:
            print(f"   [API] Submitting URL task...")
            task_id = self.submit_task(pdf_url)
            print(f"   [API] Processing (ID: {task_id})...")
            zip_url = self.wait_for_completion(task_id)
            
        except Exception as e:
            if local_file_path and os.path.exists(local_file_path):
                print(f"   [API] URL Parse Failed ({str(e)}).")
                print(f"   [API] Switching to Local Upload Fallback...")
                
                try:
                    batch_id = self.upload_file(local_file_path)
                    
                    print(f"   [Fallback] Waiting for batch processing (Batch ID: {batch_id})...")
                    zip_url = self.wait_for_batch_completion(batch_id)
                    print(f"   [Fallback] Processing done!")
                    
                except Exception as upload_e:
                    print(f"   [API] Critical: All methods failed. {upload_e}")
                    raise upload_e 
            else:
                raise e

        if zip_url:
            print(f"   [API] Downloading results...")
            self.download_and_extract(zip_url, output_dir)