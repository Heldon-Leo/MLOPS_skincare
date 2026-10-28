"""
Packaging Label OCR Service
Accepts packaging photographs, validates image constraints, preprocesses to enhance contrast,
and invokes pytesseract OCR if available. If tesseract binary is not installed on the host OS,
returns informative status and provides graceful fallback to manual entry.
"""

import os
import io
import re
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance, ImageFilter

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class LabelOCRService:
    """Extracts raw ingredient text from product packaging images."""

    def __init__(self):
        self.tesseract_binary_detected = self._check_tesseract_binary()

    def _check_tesseract_binary(self) -> bool:
        """Verifies if tesseract binary is executable on local environment."""
        if not PYTESSERACT_AVAILABLE:
            return False

        # 1. Check custom path configured via environment variable
        custom_tesseract = os.getenv("TESSERACT_CMD")
        if custom_tesseract and os.path.exists(custom_tesseract):
            pytesseract.pytesseract.tesseract_cmd = custom_tesseract
            return True

        # 2. Check standard Windows and Unix installation paths
        standard_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract"
        ]
        # Also check local user AppData dynamically
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            standard_paths.append(os.path.join(local_app_data, "Programs", "Tesseract-OCR", "tesseract.exe"))

        for p in standard_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                return True

        # 3. Try default PATH invocation
        try:
            ver = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhances image contrast and binarizes for better character recognition."""
        # Convert to grayscale
        gray = image.convert("L")
        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        contrasted = enhancer.enhance(1.8)
        # Apply slight sharpening filter
        sharpened = contrasted.filter(ImageFilter.SHARPEN)
        return sharpened

    def extract_text_from_bytes(self, image_bytes: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
        """
        Processes image bytes and extracts text.
        Returns:
            status: SUCCESS, OCR_ENGINE_UNAVAILABLE, or ERROR
            extracted_text: raw text found
            notes: helpful user guidance
        """
        # Validate format
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {
                "status": "ERROR",
                "extracted_text": "",
                "error": f"Invalid image file: {str(e)}",
                "notes": "Please upload a valid JPG or PNG image."
            }

        # Size constraint check (e.g. max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return {
                "status": "ERROR",
                "extracted_text": "",
                "error": "Image file exceeds maximum 10MB limit.",
                "notes": "Please upload a smaller image file."
            }

        if not self.tesseract_binary_detected:
            # Re-check in case user configured environment
            self.tesseract_binary_detected = self._check_tesseract_binary()

        if not self.tesseract_binary_detected:
            return {
                "status": "OCR_ENGINE_UNAVAILABLE",
                "extracted_text": "",
                "error": "Tesseract OCR binary is not installed on this host environment.",
                "notes": (
                    "To enable automatic OCR on Windows, install Tesseract OCR from UB-Mannheim "
                    "(https://github.com/UB-Mannheim/tesseract/wiki) or use the Docker deployment where "
                    "tesseract-ocr is containerized. In the meantime, please paste your ingredient list in the text editor below."
                )
            }

        try:
            processed = self.preprocess_image(img)
            text = pytesseract.image_to_string(processed, config="--psm 6")
            cleaned_text = re.sub(r"\n+", " ", text).strip()
            return {
                "status": "SUCCESS",
                "extracted_text": cleaned_text,
                "notes": "Text extracted successfully. Please review and edit any misrecognized characters before analyzing."
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "extracted_text": "",
                "error": f"OCR processing failed: {str(e)}",
                "notes": "Please paste ingredients manually in the input box."
            }


label_ocr_service = LabelOCRService()
