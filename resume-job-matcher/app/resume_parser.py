import fitz  # PyMuPDF
import logging

def extract_text_from_pdf(pdf_path):
    """
    Extracts and cleans text from a PDF file using PyMuPDF (fitz).
    Returns the full concatenated text as a string.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_number, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                # Strip unwanted characters and normalize spaces
                cleaned_text = ' '.join(page_text.split())
                text += cleaned_text + " "
        doc.close()
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return ""
    
    return text.strip()
