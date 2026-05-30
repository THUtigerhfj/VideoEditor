import pandas as pd
import os
import requests
import time
from tqdm import tqdm

def download_video(url, file_path, retries=5):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return True

    tmp_path = f"{file_path}.part"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in tqdm(res.iter_content(chunk_size=10240), desc="Downloading video"):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, file_path)
            return True
        except requests.RequestException as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt == retries:
                print(f"Failed after {retries} attempts: {url} -> {file_path}: {exc}")
                return False
            time.sleep(min(2 ** attempt, 30))

output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)


# videovo_df = pd.read_csv("../data/videovo.csv")
pexels_df = pd.read_csv("../data/pexels.csv")


# for index, row in tqdm(videovo_df[:].iterrows(), desc="Downloading videovo videos"):
#     video_url = row["video_url"]
#     file_path = os.path.join(output_dir, "videovo/raw_video", row["file_path"])
#     download_video(video_url, file_path)


failed = []
for index, row in tqdm(pexels_df[:].iterrows(), total=len(pexels_df), desc="Downloading pexels videos"):
    video_url = row["link"]
    dir_prefix = f"{index:012d}"
    formatted_filename = f"{dir_prefix[:9]}/{dir_prefix}_{row['videoId']}.mp4"
    file_path = os.path.join(output_dir, "pexels/pexels/raw_video", formatted_filename)
    if not download_video(video_url, file_path):
        failed.append((index, video_url, file_path))

if failed:
    failed_path = os.path.join(output_dir, "pexels_download_failed.txt")
    with open(failed_path, "w") as f:
        for index, url, file_path in failed:
            f.write(f"{index}\t{url}\t{file_path}\n")
    raise SystemExit(f"{len(failed)} downloads failed. See {failed_path}")
