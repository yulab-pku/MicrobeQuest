from typing import Dict, List
import fitz  # PyMuPDF

class ExtractTextByPyMuPDF:

    def extract_text(self, pdf_path: str, page_numbers: List[int] = None) -> Dict[int, str]:
        """
        Extracts text from a PDF using PyMuPDF.
        :param pdf_path: Path to the PDF file.
        :param page_numbers: List of page numbers to extract (1-based). If None, extract all pages.
        :return: Dictionary {page_number: extracted_text}
        """
        doc = fitz.open(pdf_path)
        page_numbers = page_numbers if page_numbers is not None else []

        full_text = {}
        try:
            for page_number, page in enumerate(doc, start=1):
                if not page_numbers or page_number in page_numbers:
                    text = page.get_text("text")
                    if text.strip():
                        full_text[page_number] = text
        except Exception as e:
            print(f"PDF extraction failed: {e}")
        return full_text
