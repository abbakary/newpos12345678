"""
PDF and image text extraction without OCR.
Uses PyMuPDF (fitz) and PyPDF2 for PDF text extraction.
Falls back to pattern matching for invoice data extraction.
"""

import io
import logging
import re
from decimal import Decimal
from datetime import datetime

try:
    import fitz
except ImportError:
    fitz = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes) -> str:
    """Extract text from PDF file using PyMuPDF or PyPDF2.
    
    Args:
        file_bytes: Raw bytes of PDF file
        
    Returns:
        Extracted text string
        
    Raises:
        RuntimeError: If no PDF extraction library is available
    """
    text = ""
    
    # Try PyMuPDF first (fitz) - best for text extraction
    if fitz is not None:
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf_doc:
                text += page.get_text()
            pdf_doc.close()
            logger.info(f"Extracted {len(text)} characters from PDF using PyMuPDF")
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            text = ""
    
    # Fallback to PyPDF2
    if PyPDF2 is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text()
            logger.info(f"Extracted {len(text)} characters from PDF using PyPDF2")
            return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            text = ""
    
    if not text:
        raise RuntimeError('No PDF text extraction library available. Install PyMuPDF or PyPDF2.')
    
    return text


def extract_text_from_image(file_bytes) -> str:
    """Extract text from image file.
    Since OCR is not available, this returns empty string.
    Images should be uploaded as PDFs or entered manually.
    
    Args:
        file_bytes: Raw bytes of image file
        
    Returns:
        Empty string (manual entry required for images)
    """
    logger.info("Image file detected. OCR not available. Manual entry required.")
    return ""


def parse_invoice_data(text: str) -> dict:
    """Parse invoice data from extracted text using pattern matching.
    
    This method uses regex patterns to extract invoice fields from raw text.
    It's designed to work with common invoice formats.
    
    Args:
        text: Raw extracted text from PDF/image
        
    Returns:
        dict with extracted invoice data
    """
    if not text or not text.strip():
        return {
            'invoice_no': None,
            'date': None,
            'customer_name': None,
            'address': None,
            'subtotal': None,
            'tax': None,
            'total': None,
            'items': []
        }
    
    # Helper to find first match group
    def find(pattern):
        m = re.search(pattern, text, re.I | re.MULTILINE)
        return m.group(1).strip() if m else None
    
    # Extract invoice number
    invoice_no = (
        find(r'(?:Invoice\s*(?:Number|No\.?|#)[\s:\-]*)?([A-Z0-9\-\/]+?)(?:\s|$|[\n\r])') or
        find(r'(?:PI|PI\.?|Code\s*No|Code)[\s:\-]*([A-Z0-9\-]+)') or
        None
    )
    
    # Extract date
    date_str = find(r'(?:Date|Invoice\s*Date)[\s:\-]*([0-3]?\d[\-/][01]?\d[\-/]\d{2,4})')
    
    # Extract customer name
    customer_name = (
        find(r'(?:Customer\s*Name|Customer|Bill\s*To|Buyer|Name)[\s:\-]*([^\n\r\:]{3,120})') or
        find(r'(?:To\s*[:.\s]+)([^\n\r]{3,100})')
    )

    # Extract address
    address = find(r'(?:Address|Addr\.|Add|Location)[\s:\-]*([^\n\r]{5,200})')

    # Extract phone
    phone = find(r'(?:Tel|Phone|Mobile|Contact|Phone\s*Number)[\s:\-]*(\+?[0-9\s\-\(\)]{7,20})')

    # Extract email
    email = find(r'(?:Email|E-mail|Contact\s*Email)[\s:\-]*([^\s\n\r:@]+@[^\s\n\r:]+)')
    
    # Extract monetary amounts
    subtotal = find(r'(?:Sub\s*Total|Subtotal|Net\s*(?:Value|Amount))[\s:\-]*([0-9\,]+\.?\d{0,2})')
    tax = find(r'(?:VAT|Tax|GST|Sales\s*Tax)[\s:\-]*([0-9\,]+\.?\d{0,2})')
    total = find(r'(?:Total|Grand\s*Total|Amount\s*Due|Total\s*Amount)[\s:\-]*(?:TSH|TZS|UGX)?\s*([0-9\,]+\.?\d{0,2})')
    
    # Parse monetary values
    def to_decimal(s):
        try:
            if s:
                return Decimal(str(s).replace(',', ''))
        except Exception:
            pass
        return None
    
    # Extract line items (simple heuristic)
    items = []
    lines = text.split('\n')
    item_section_started = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect item section start
        if re.search(r'\b(Item|Description|Qty|Quantity|Unit|Price|Amount|Value)\b', line, re.I):
            item_section_started = True
            continue
        
        # Stop at total/footer
        if item_section_started and re.search(r'\b(Sub\s*Total|Total|Grand\s*Total|VAT|Tax|Payment)\b', line, re.I):
            break
        
        # Try to parse line as item
        if item_section_started and len(line) > 5:
            # Find numbers in the line
            numbers = re.findall(r'[0-9\,]+\.?\d*', line)
            if len(numbers) >= 1:  # At least a quantity or amount
                # Extract description (remove all numbers)
                desc = re.sub(r'[0-9\,]+\.?\d*', '', line).strip()
                if desc and len(desc) > 2:
                    # Last number is usually value
                    value = numbers[-1] if numbers else None
                    qty = None
                    # Second-to-last might be qty
                    if len(numbers) >= 2:
                        qty = numbers[-2]
                    
                    items.append({
                        'description': desc[:255],
                        'qty': int(float(qty.replace(',', ''))) if qty else 1,
                        'value': to_decimal(value)
                    })
    
    return {
        'invoice_no': invoice_no,
        'date': date_str,
        'customer_name': customer_name,
        'phone': phone,
        'email': email,
        'address': address,
        'subtotal': to_decimal(subtotal),
        'tax': to_decimal(tax),
        'total': to_decimal(total),
        'items': items
    }


def extract_from_bytes(file_bytes, filename: str = '') -> dict:
    """Main entry point: extract text from file and parse invoice data.
    
    Supports:
    - PDF files: Uses PyMuPDF/PyPDF2 for text extraction
    - Image files: Requires manual entry (OCR not available)
    
    Args:
        file_bytes: Raw bytes of uploaded file
        filename: Original filename (to detect file type)
        
    Returns:
        dict with keys: success, header, items, raw_text, ocr_available, error, message
    """
    if not file_bytes:
        return {
            'success': False,
            'error': 'empty_file',
            'message': 'File is empty',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }
    
    # Detect file type
    is_pdf = filename.lower().endswith('.pdf') or file_bytes[:4] == b'%PDF'
    is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp'))
    
    text = ""
    extraction_error = None
    
    # Try to extract text
    if is_pdf:
        try:
            text = extract_text_from_pdf(file_bytes)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            extraction_error = str(e)
            return {
                'success': False,
                'error': 'pdf_extraction_failed',
                'message': f'Failed to extract text from PDF: {str(e)}. Please enter invoice details manually.',
                'ocr_available': False,
                'header': {},
                'items': [],
                'raw_text': ''
            }
    elif is_image:
        return {
            'success': False,
            'error': 'image_file_not_supported',
            'message': 'Image files require manual entry (OCR not available). Please save as PDF or enter details manually.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }
    else:
        return {
            'success': False,
            'error': 'unsupported_file_type',
            'message': 'Please upload a PDF file (images are not supported without OCR).',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }
    
    # Parse extracted text
    if text:
        try:
            parsed = parse_invoice_data(text)
            return {
                'success': True,
                'header': parsed,
                'items': parsed.get('items', []),
                'raw_text': text,
                'ocr_available': False,  # Using text extraction, not OCR
                'message': 'Invoice data extracted successfully from PDF'
            }
        except Exception as e:
            logger.warning(f"Failed to parse invoice data: {e}")
            return {
                'success': False,
                'error': 'parsing_failed',
                'message': 'Could not extract structured data from PDF. Please enter invoice details manually.',
                'ocr_available': False,
                'header': {},
                'items': [],
                'raw_text': text
            }
    
    # If no text was extracted
    return {
        'success': False,
        'error': 'no_text_extracted',
        'message': 'No text found in PDF. Please enter invoice details manually.',
        'ocr_available': False,
        'header': {},
        'items': [],
        'raw_text': ''
    }
