import json
import os
import time
from pathlib import Path

import requests
from urllib.parse import quote

from pdf2image import convert_from_path
from tqdm import tqdm

# === Configuration ===
EMAIL = "YOUR EMAIL"     # Replace with your email (required by Unpaywall)
INPUT_JSON = "urls.json"            # Path to the input JSON file containing DOIs or URLs
SAVE_DIR = "src/data/raw_pdfs"      # Directory to save downloaded PDFs
IMAGE_DIR = "src/data/images"       # Directory to save converted images

# === Create save directories ===
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def extract_doi(url):
    """Extract DOI from a URL like 'https://doi.org/10.xxxx'"""
    return url.split("doi.org/")[-1].strip()


def get_pdf_link_from_unpaywall(doi):
    """Query Unpaywall API to get the best open-access PDF link for a given DOI"""
    api_url = f"https://api.unpaywall.org/v2/{quote(doi)}?email={EMAIL}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("url_for_pdf"):
                return best_oa["url_for_pdf"]
    except Exception as e:
        print(f"[Unpaywall Error] {doi}: {e}")
    return None


def download_pdf(pdf_url, save_path):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        }
        with requests.get(pdf_url, stream=True, headers=headers, timeout=20) as r:
            print(f"Status: {r.status_code}, Content-Type: {r.headers.get('Content-Type')}")
            if r.status_code == 200 and 'pdf' in r.headers.get('Content-Type', '').lower():
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception as e:
        print(f"[Download Error] {pdf_url}: {e}")
    return False


def is_direct_pdf_url(url):
    """Check if the URL directly points to a PDF file"""
    return url.lower().endswith(".pdf")


def download_all_papers(paper_path):
    """Main logic: download from direct PDF URLs or via Unpaywall using DOI"""
    with open(paper_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    for pid, url in tqdm(papers.items(), desc="Downloading papers"):
        save_path = os.path.join(SAVE_DIR, f"{pid}.pdf")
        if os.path.exists(save_path):
            print(f"{pid}: ✅ Already downloaded.")
            continue

        if is_direct_pdf_url(url):
            print(f"{pid}: 🔗 Direct PDF URL detected, downloading: {url}")
            success = download_pdf(url, save_path)
        else:
            doi = extract_doi(url)
            pdf_url = get_pdf_link_from_unpaywall(doi)
            if pdf_url:
                print(f"{pid}: 🌐 Downloading via Unpaywall: {pdf_url}")
                success = download_pdf(pdf_url, save_path)
            else:
                print(f"{pid}: ⚠️ No open-access PDF found via Unpaywall.")
                continue

        if success:
            print(f"{pid}: ✅ Downloaded successfully.")
        else:
            print(f"{pid}: ❌ Failed to download.")
        time.sleep(1)  # Avoid hitting Unpaywall rate limits


def convert_pdf_to_images(pdf_dir, output_dir):
    """
    Convert PDFs to images. Each PDF will be saved in the output_dir.
    For example: src/data/images/123456_I001.png
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_list = sorted(pdf_dir.glob('*.pdf'))

    for pdf_path in tqdm(pdf_list, desc="Converting PDFs to images"):
        pdf_name = pdf_path.stem
        try:
            images = convert_from_path(str(pdf_path))
            for i, image in enumerate(images):
                image_path = output_dir / f"{pdf_name}_I{i + 1:03d}.png"
                image.save(image_path)
                print(f'✅ Saved: {image_path}')
        except Exception as e:
            print(f'❌ Failed to convert {pdf_path.name}: {e}')


if __name__ == "__main__":
    download_all_papers(INPUT_JSON)
    convert_pdf_to_images(SAVE_DIR, IMAGE_DIR)
