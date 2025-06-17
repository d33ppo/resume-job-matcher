import fitz  # PyMuPDF
import logging

def extract_text_from_pdf(pdf_path):
    """
    Extracts and cleans text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the uploaded PDF resume.
    
    Returns:
        str: Extracted and normalized text from all pages.
    """
    text = ""

    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)

        # Iterate through each page and extract text
        for page_number, page in enumerate(doc):
            page_text = page.get_text()

            if page_text:
                # Normalize spaces and remove line breaks
                cleaned_text = ' '.join(page_text.split())
                text += cleaned_text + " "  # Add space between pages

        # Close the document to free memory
        doc.close()

    except Exception as e:
        # Log error for debugging if something goes wrong (e.g., corrupt PDF)
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

    # Return final cleaned text
    return text.strip()
