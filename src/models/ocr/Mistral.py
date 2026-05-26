import os
import base64
from pathlib import Path
from mistralai import DocumentURLChunk
from mistralai import Mistral


class ExtractTextByMistral:
    def __init__(self, api_key):
        self.api_key = api_key

    def replace_images_in_markdown(self, markdown_str: str, images_dict: dict) -> str:
        """Replace image paths in Markdown content"""
        for img_name, img_path in images_dict.items():
            markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({img_path})")
        return markdown_str

    def save_ocr_results(self, ocr_response, output_dir: str):
        """Save OCR results, including images and Markdown"""
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        all_markdowns = []

        for i, page in enumerate(ocr_response.pages):
            page_images = {}
            for img in page.images:
                try:
                    img_data = base64.b64decode(img.image_base64.split(',')[1])
                    img_path = os.path.join(images_dir, f"{img.id}")
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    page_images[img.id] = os.path.join("images", f"{img.id}")
                except Exception as e:
                    print(f"Image processing failed: {e}")

            # Process Markdown content
            page_markdown = self.replace_images_in_markdown(page.markdown, page_images)
            all_markdowns.append(page_markdown)

        return all_markdowns

    def extract_text(self, pdf_path: str, page_numbers: list):
        """
        Process the PDF and perform OCR.
        Only return the Markdown content of the first page whose index is in the specified image_index list.
        Note: image_index refers to page indices (page.index), not image IDs.
        """
        page_numbers=range(len(page_numbers))

        client = Mistral(api_key=self.api_key)
        pdf_file = Path(pdf_path)
        if not pdf_file.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        with open(pdf_file, "rb") as file:
            uploaded_file = client.files.upload(
                file={"file_name": pdf_file.name, "content": file.read()},
                purpose="ocr",
            )

        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=24)
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

        # Collect markdowns from all matching pages
        matched_markdowns = {}
        for page in pdf_response.pages:
            if page_numbers and page.index not in page_numbers:
                continue  # Only process specified pages
            matched_markdowns[page.index] = page.markdown
        return matched_markdowns  # Return all matched markdowns (could be empty)



