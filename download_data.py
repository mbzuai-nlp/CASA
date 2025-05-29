import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from http.cookiejar import MozillaCookieJar
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import pandas as pd

# Thread-local storage for sessions
thread_local = threading.local()

def get_session(cookie_path="cookies.txt"):
    """Get a session for the current thread"""
    if not hasattr(thread_local, 'session'):
        session = requests.Session()
        cj = MozillaCookieJar()
        try:
            cj.load(cookie_path, ignore_discard=True, ignore_expires=True)
            session.cookies = cj
        except FileNotFoundError:
            print(f"[ERROR] Cookie file '{cookie_path}' not found. Please provide it.")
            exit(1)
        thread_local.session = session
    return thread_local.session

def load_session_with_cookies(cookie_path="cookies.txt"):
    session = requests.Session()
    cj = MozillaCookieJar()
    try:
        cj.load(cookie_path, ignore_discard=True, ignore_expires=True)
        session.cookies = cj
    except FileNotFoundError:
        print(f"[ERROR] Cookie file '{cookie_path}' not found. Please provide it.")
        exit(1)
    return session

def is_file_link(href):
    # Basic heuristic: file link contains extension and does not end with '/'
    return href and not href.endswith('/') and re.search(r'\.\w+$', href)

def sanitize_filename(url):
    # Returns the path relative to base URL
    return urlparse(url).path.lstrip('/')

def download_file(file_url, base_url, output_root, cookie_path="cookies.txt"):
    """Download a single file with progress tracking"""
    session = get_session(cookie_path)
    
    relative_path = sanitize_filename(file_url).replace(base_url.replace("https://", "").replace("/", "_"), "")
    local_path = os.path.join(output_root, os.path.basename(relative_path))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # append save get request to the file URL
    file_url_with_save = file_url + "?f=save"
    
    try:
        response = session.get(file_url_with_save, stream=True)
        if response.status_code in (200, 206):
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"📥 {os.path.basename(local_path)}", leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            return f"[✓] Downloaded: {file_url} → {local_path}"
        else:
            return f"[✗] Failed: {file_url} (status {response.status_code})"
    except Exception as e:
        return f"[✗] Error downloading {file_url}: {str(e)}"

def scrape_and_download(session, base_url, output_root="datasets/talkbank", file_extensions=None, valid_file_ids=None, max_workers=8):
    print(f"[INFO] Scanning index page: {base_url}")
    response = session.get(base_url)
    if response.status_code != 200:
        print(f"[ERROR] Cannot access {base_url} (status {response.status_code})")
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a")]
    file_links = []

    for href in links:
        if not href or href.startswith("?") or href.startswith("#"):
            continue
        full_url = urljoin(base_url, href)
        if href.endswith("/"):
            # Recurse into subdirectory
            # scrape_and_download(session, full_url, output_root, file_extensions)
            continue
        elif is_file_link(href):
            if file_extensions is None or any(href.lower().endswith(ext) for ext in file_extensions):
                # Check if the file ID is valid
                file_id = os.path.basename(href).split('.')[0] 
                if file_id in valid_file_ids:
                    file_links.append(full_url)

    if not file_links:
        print(f"[INFO] No files found at {base_url}")
        return

    print(f"[INFO] Found {len(file_links)} files to download")
    print(f"[INFO] Starting parallel downloads with {max_workers} workers...")

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_file, file_url, base_url, output_root): file_url 
            for file_url in file_links
        }
        
        # Track progress with tqdm
        with tqdm(total=len(file_links), desc="Overall Progress", unit="files") as overall_pbar:
            for future in as_completed(future_to_url):
                file_url = future_to_url[future]
                try:
                    result = future.result()
                    # print(result)
                except Exception as exc:
                    print(f'[✗] {file_url} generated an exception: {exc}')
                finally:
                    overall_pbar.update(1)

if __name__ == "__main__":

    TALKBANK_INDEX_URL = {
        'interview': "https://media.talkbank.org/fluency/Voices-AWS/interview", 
        'reading': "https://media.talkbank.org/fluency/Voices-AWS/reading",
    }
    
    session = load_session_with_cookies("cookies.txt")
    df = pd.read_csv("data/Voices-AWS/total_dataset.csv")
    # Download all files (or restrict with e.g. ['.mp4', '.cha'])
    for sub_set, url in TALKBANK_INDEX_URL.items():
        print(f"\nStarting downloads for: {sub_set}")
        scrape_and_download(
            session, 
            url, 
            output_root=f"data/Voices-AWS/{sub_set}/videos", 
            file_extensions=None,
            valid_file_ids=df[df['task'] == sub_set]['media_file'].dropna().unique().tolist(),
            max_workers=8  # Adjust this number based on your needs
        )
        print(f"Completed downloads for: {sub_set}")