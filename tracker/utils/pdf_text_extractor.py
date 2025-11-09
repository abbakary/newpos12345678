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
    It's designed to work with common professional invoice formats including:
    - Pro forma invoices with Code No, Customer Name, Address, Tel, Reference
    - Traditional invoices with Invoice Number, Date, Customer, etc.

    Args:
        text: Raw extracted text from PDF/image

    Returns:
        dict with extracted invoice data
    """
    if not text or not text.strip():
        return {
            'invoice_no': None,
            'code_no': None,
            'date': None,
            'customer_name': None,
            'address': None,
            'phone': None,
            'email': None,
            'reference': None,
            'subtotal': None,
            'tax': None,
            'total': None,
            'items': []
        }

    # Normalize text for pattern matching
    normalized_text = text.strip()

    # Helper to find first match group with flexible spacing
    def find(pattern, flags=re.I | re.MULTILINE | re.DOTALL):
        m = re.search(pattern, normalized_text, flags)
        if m:
            result = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group(0).strip()
            # Clean extra whitespace
            result = ' '.join(result.split())
            return result if result else None
        return None

    # Extract Code No (pro forma invoice format)
    code_no = find(r'(?:Code\s*(?:No|Number)\.?|Code\s*#)[\s:\-]*([A-Z0-9\-\/]+?)(?:\n|$|[,;])')

    # Extract invoice number (multiple formats)
    invoice_no = (
        find(r'(?:Invoice\s*(?:Number|No\.?|#|Date)[\s:\-]*)?([A-Z]{0,3}\d{5,20})(?:\n|$|\s)') or
        find(r'(?:PI|P\.?I\.?|Invoice|INV)[\s:\-]*([A-Z0-9\-\/]+?)(?:\n|$)') or
        code_no  # Use Code No as fallback for invoice number
    )

    # Extract date (flexible format)
    date_str = find(r'(?:Date|Invoice\s*Date|Dated)[\s:\-]*([0-3]?\d[\s/\-][01]?\d[\s/\-]\d{2,4})')

    # Extract customer name (next non-empty line after label or standalone)
    customer_name = find(r'(?:Customer\s*Name|Customer|Bill\s*To|Buyer|Name|TO)[\s:\-]+([A-Z][^\n\r]{3,150})(?:\n|$)')
    if not customer_name:
        # Try to find all caps lines that look like company names
        caps_lines = [line.strip() for line in normalized_text.split('\n') if line.strip() and line.strip().isupper() and len(line.strip()) > 5]
        if caps_lines:
            customer_name = caps_lines[0]

    # Extract address (handle multiline addresses)
    address_match = find(r'(?:Address|Addr\.|ADD)[\s:\-]+((?:[^\n]{5,100}(?:\n(?!\w+[\s:\-]))?)+)', re.I | re.MULTILINE)
    if address_match:
        # Clean up multiline address
        address = ' '.join(address_match.split())
    else:
        address = None

    # Extract phone/tel
    phone = find(r'(?:Tel|Telephone|Phone|Mobile|Contact\s*(?:Number|Tel)|Fax)[\s:\-]*(\+?[0-9\s\-\(\)\.]{7,25})')

    # Extract email
    email = find(r'(?:Email|E-mail|Mail)[\s:\-]*([^\s\n\r:@]+@[^\s\n\r:]+)')

    # Extract reference (for pro forma invoices)
    reference = find(r'(?:Reference|Ref\.?|Order\s*(?:Number|Ref)|FOR)[\s:\-]*([A-Z0-9\s\-\/]{3,50})')

    # Extract monetary amounts (with currency support)
    def extract_amount(label_pattern):
        patterns = [
            rf'{label_pattern}[\s:\-]*(?:TSH|TZS|UGX|USD)?\s*([0-9\,]+\.?\d{{0,2}})',
            rf'{label_pattern}[\s:\-]*([0-9\,]+\.?\d{{0,2}})',
        ]
        for pattern in patterns:
            match = find(pattern)
            if match:
                return match
        return None

    subtotal = extract_amount(r'(?:Sub\s*Total|Subtotal|Net\s*(?:Value|Amount)|Net)')
    tax = extract_amount(r'(?:VAT|Tax|GST|Sales\s*Tax|Vat\s*@)')
    total = extract_amount(r'(?:Grand\s*Total|Total\s*Amount|Total(?:\s|:)|Amount\s*Due)')

    # Parse monetary values
    def to_decimal(s):
        try:
            if s:
                # Remove currency symbols and extra whitespace
                cleaned = re.sub(r'[^\d\.\,\-]', '', str(s)).strip()
                return Decimal(cleaned.replace(',', ''))
        except Exception:
            pass
        return None

    # Extract line items with improved heuristics
    items = []
    lines = normalized_text.split('\n')
    item_section_started = False
    item_header_idx = -1

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Detect item section header
        if re.search(r'(?:Item|Description|Qty|Quantity|Unit|Price|Amount|Value|Sr\.|S\.N)', line, re.I) and \
           re.search(r'(?:Description|Qty|Quantity|Price|Amount|Value)', line, re.I):
            item_section_started = True
            item_header_idx = idx
            continue

        # Stop at summary/footer sections
        if item_section_started and idx > item_header_idx + 1 and \
           re.search(r'(?:Sub\s*Total|Total|Grand\s*Total|VAT|Tax|Payment|Amount\s*Due|Summary)', line, re.I):
            item_section_started = False
            break

        # Parse line as item (must be after header and contain numbers)
        if item_section_started and idx > item_header_idx:
            # Look for lines with numeric values
            numbers = re.findall(r'[0-9\,]+\.?\d*', line)
            if len(numbers) >= 1 and len(line) > 5:
                # Extract description by removing numbers
                desc = re.sub(r'\s*[0-9\,]+\.?\d*\s*', ' ', line).strip()
                desc = ' '.join(desc.split())  # Clean whitespace

                if desc and len(desc) > 2 and not re.match(r'^\d+$', desc):
                    # Last number is usually the amount
                    value = numbers[-1] if numbers else None
                    qty = None

                    # If we have 2+ numbers, second-to-last might be qty
                    if len(numbers) >= 2:
                        # Check if it looks like a quantity (small integer)
                        try:
                            qty_val = int(float(numbers[-2].replace(',', '')))
                            if 0 < qty_val < 1000:  # Reasonable qty range
                                qty = numbers[-2]
                        except Exception:
                            pass

                    if not qty:
                        qty = '1'

                    items.append({
                        'description': desc[:255],
                        'qty': int(float(qty.replace(',', ''))) if qty else 1,
                        'value': to_decimal(value)
                    })

    return {
        'invoice_no': invoice_no,
        'code_no': code_no,
        'date': date_str,
        'customer_name': customer_name,
        'phone': phone,
        'email': email,
        'address': address,
        'reference': reference,
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
            # Prepare header with all extracted fields
            header = {
                'invoice_no': parsed.get('invoice_no'),
                'code_no': parsed.get('code_no'),
                'date': parsed.get('date'),
                'customer_name': parsed.get('customer_name'),
                'phone': parsed.get('phone'),
                'email': parsed.get('email'),
                'address': parsed.get('address'),
                'reference': parsed.get('reference'),
                'subtotal': parsed.get('subtotal'),
                'tax': parsed.get('tax'),
                'total': parsed.get('total'),
            }
            return {
                'success': True,
                'header': header,
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
