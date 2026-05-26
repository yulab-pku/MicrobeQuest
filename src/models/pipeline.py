import os
from typing import Dict
import yaml



class Pipeline:
    def __init__(self, ocr_model=None, ir_model=None, vlm_model = None):
        self.ocr_model = ocr_model
        self.ir_model = ir_model
        self.vlm_model = vlm_model

    def run(self, test_case: Dict, pdf_root_path: str) -> Dict:
        """
        Pipeline steps:
        1. Extract OCR text based on test_case['test_case']['input']
        2. Put the OCR text into test_case['test_case']['input']['text']
        3. Pass the input to the IR model to obtain the final result
        """
        if self.vlm_model is None:
            # 1. Extract OCR input information
            pdf_index = test_case['test_case']['input']['pdf_index']
            pdf_index = pdf_index.split(',')
            raw_image_indices = test_case['test_case']['input'].get('image_index', '')
            if isinstance(raw_image_indices, list):
                image_indices = [int(img.split('_')[-1].replace('I', '')) for img in raw_image_indices]
            elif isinstance(raw_image_indices, str):
                image_indices = [int(raw_image_indices.split('_')[-1].replace('I', ''))]
            else:
                image_indices = []

            is_full_pdf = test_case['test_case']['input'].get('is_full_pdf', 'false') == 'true'

            # 2. Construct the full PDF path
            for p in pdf_index:
                all_ocr_text = ''
                pdf_path = os.path.join(pdf_root_path, f"{p}.pdf")
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

                # 3. Use the OCR model to extract text
                ocr_text = self.ocr_model.extract_text(pdf_path, page_numbers=image_indices if not is_full_pdf else None)
                all_ocr_text += str(ocr_text)

            # 4. Add the OCR result into the test_case
                test_case['test_case']['input']['text'] = all_ocr_text
            # 5. Call the IR model
            result = self.ir_model.generate_answer(test_case)
        else:
            result = self.vlm_model.generate_answer(test_case, image_root_path=r"src\data\images")
        return result