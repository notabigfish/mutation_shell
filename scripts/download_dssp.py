import pandas as pd
import glob
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
import time
import random

OUTPUT_DIR = "/rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet/data/dssp_downloads/"
MAX_WORKERS = 8
TIMEOUT = (5, 20)
RETRIES = 3

data = pd.read_csv('/rds/homes/s/sxz325/shuo/mutation/MuSRNet/data/SingleMutPairs2024_subset_c1000.csv')
pdb_ids = list(set(data['wt_pdb_id'].dropna().unique().tolist() + data['mut_pdb_id'].dropna().unique().tolist()))
downloaded_dssps = glob.glob(os.path.join(OUTPUT_DIR, '*.dssp'))
downloaded_dssps = [os.path.basename(f).split('.')[0] for f in downloaded_dssps]
remaining_ids = [f for f in pdb_ids if f not in downloaded_dssps]

def download_file(pdb_id):
    pdb_id = str(pdb_id).strip().lower()
    file_path = os.path.join(OUTPUT_DIR, f"{pdb_id}.dssp")

    urls = [
        f"https://pdb-redo.eu/dssp/db/{pdb_id}/legacy",
        f"https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.dssp",
    ]

    with requests.Session() as s:
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=1)
        s.mount("https://", adapter)

        for url in urls:
            for attempt in range(RETRIES):
                try:
                    time.sleep(random.uniform(0.1, 0.3))
                    response = s.get(url, timeout=TIMEOUT)

                    if response.status_code == 404:
                        break

                    response.raise_for_status()

                    if len(response.content) < 100:
                        break

                    with open(file_path, "wb") as f:
                        f.write(response.content)

                    return True

                except requests.RequestException:
                    if attempt < RETRIES - 1:
                        time.sleep(1 + attempt)

        if os.path.exists(file_path):
            os.remove(file_path)

        return False

total_tasks = len(remaining_ids)
	
success_count = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
	future_to_id = {executor.submit(download_file, pid): pid for pid in remaining_ids}
	
	for future in tqdm(as_completed(future_to_id), total=total_tasks, desc="Downloading", unit="file"):
		if future.result():
			success_count += 1
			
print(f"\n success: {success_count}/{total_tasks}")