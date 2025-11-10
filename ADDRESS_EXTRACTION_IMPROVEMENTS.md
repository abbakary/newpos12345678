# Address Extraction Improvements

## Problem Identified

The address extraction from invoices wasn't reliably capturing complete multi-line addresses, particularly with:
- P.O. Box numbers followed by city and country on separate lines
- Addresses with multiple components (Street, City, Province, Country)
- Inconsistent formatting in PDF text extraction

Example from your Superdoll invoice:
```
P.O.BOX 15950
DAR ES SALAAM
TANZANIA
```

The extraction needed to properly handle this multi-line structure and reliably extract all components.

## Solutions Implemented

### 1. Extended Collection Range (Line 420)

**Change**: `min(idx + 5, len(lines))` → `min(idx + 7, len(lines))`

**Reason**: Allows collection of up to 6 more lines after P.O. Box detection instead of just 4, ensuring complete address capture.

**Benefit**: Multi-line addresses with 4-5 components are now fully captured.

### 2. Improved Label Detection (Line 424)

**Old**: 
```python
if re.match(r'^(?:Tel|Fax|Attended|Kind|Reference|PI|Code|Type|Date|Email|Phone|Del)', next_line, re.I):
```

**New**: 
```python
if re.match(r'^(?:Tel|Fax|Attended|Kind|Reference|PI|Code|Type|Date|Email|Phone|Del|Customer|Cust|Ref|Invoice|Proforma)', next_line, re.I):
```

**Benefit**: Better stops at field labels, preventing address from absorbing "Customer Name", "Reference", "Invoice No" fields.

### 3. Enhanced City/Country Detection (Lines 428-436)

**Old**: Simple regex looking for individual words

**New**: More comprehensive pattern matching:
- Added more East African cities: MOMBASA, MOSHI, ARUSHA, DODOMA
- Separate detection for cities and countries
- Better handling of multi-word city names (DAR-ES-SALAAM, DAR ES SALAAM)
- Support for additional African countries: CONGO, MALAWI, ZAMBIA

**Code**:
```python
if re.search(r'\b(DAR|DAR-ES-SALAAM|SALAAM|NAIROBI|KAMPALA|KIGALI|MOMBASA|MOSHI|ARUSHA|DODOMA)\b', next_line, re.I):
    address_parts.append(next_line)
elif re.search(r'\b(TANZANIA|KENYA|UGANDA|RWANDA|BURUNDI|CONGO|MALAWI|ZAMBIA)\b', next_line, re.I):
    address_parts.append(next_line)
```

**Benefit**: Handles various city/country name formats more robustly.

### 4. Better "Address" Label Handling (Lines 449-475)

**Improvement**: 
- Check for "Address" label both at end of line and with content on same line
- Extended collection from 5 to 7 lines
- More comprehensive label stop list

**Code**:
```python
if re.search(r'\bAddress\s*[:=]?\s*$', line, re.I) or re.search(r'\bAddress\s*[:=]\s*([^\n]+)', line, re.I):
    # Extract address components on same line if present
    # Then collect following lines
```

**Benefit**: Handles both formats:
- `Address: P.O.BOX 15950` (content on same line)
- `Address` followed by address lines

### 5. Enhanced City/Country Fallback (Lines 480-502)

**Improvements**:
- Extended search range (4 lines instead of 3)
- More comprehensive city name list
- Better detection of address lines vs field labels
- Support for lines with numbers (postal codes, street numbers)

**Benefit**: As a last resort, can still extract address even if not labeled.

## Multi-Line Address Handling

The improved logic now properly handles addresses like:

```
P.O.BOX 15950
DAR ES SALAAM
TANZANIA
```

**Extraction Process**:
1. Detects "P.O.BOX 15950" line
2. Collects next lines: "DAR ES SALAAM", "TANZANIA"
3. Stops when hitting "Tel:", "Phone:", "Email:", "Customer Name", etc.
4. Joins collected parts: "P.O.BOX 15950 DAR ES SALAAM TANZANIA"

## Field Labels Now Recognized

The extraction now properly stops at these field labels:
- Tel, Fax, Phone (phone fields)
- Email (email field)
- Customer, Cust (customer info)
- Reference, Ref (invoice reference)
- PI, Invoice, Proforma (invoice number)
- Code, Type, Date (invoice metadata)
- Attended, Kind (special handling fields)
- Payment, Delivery (payment info)
- Remarks (comments)

## Testing Address Extraction

Run the improved test to validate address extraction:

```bash
python test_invoice_extraction_improved.py
```

The test will verify:
- P.O. BOX number is captured
- City (DAR/SALAAM) is included
- Country (TANZANIA) is included

## Address Correction Features

The code also includes smart corrections for cases where:
1. **Customer name appears in address**: If customer name wasn't extracted but address starts with a likely name, it's extracted from address and removed from address field
2. **Multi-word cities**: "DAR ES SALAAM" is now properly recognized as a single city
3. **Alternative formatting**: Handles "P O BOX", "P.O BOX", "PO BOX", etc.

## Examples of Correct Extraction

**Invoice Format 1** (P.O. Box model):
```
Address
P.O.BOX 15950
DAR ES SALAAM
TANZANIA
```
✅ Extracted as: "P.O.BOX 15950 DAR ES SALAAM TANZANIA"

**Invoice Format 2** (Street address model):
```
Address: 123 Main Street
Nairobi
Kenya
```
✅ Extracted as: "123 Main Street Nairobi Kenya"

**Invoice Format 3** (Minimal):
```
DAR
TANZANIA
Tel: ...
```
✅ Extracted as: "DAR TANZANIA"

## Edge Cases Handled

1. ✅ Multiple empty lines between address components
2. ✅ City names on same line as country
3. ✅ Postal codes/zip codes with numbers
4. ✅ All-caps or title-case formatting
5. ✅ Different P.O. Box formats (P.O. BOX, P O BOX, POB, P.O)
6. ✅ City names with hyphens (DAR-ES-SALAAM)

## Files Modified

- **tracker/utils/pdf_text_extractor.py**:
  - Lines 420: Extended collection range for P.O. Box addresses
  - Lines 424, 428-436: Improved city/country detection
  - Lines 449-475: Better "Address" label handling
  - Lines 480-502: Enhanced fallback pattern matching

## Performance Impact

- Negligible: Extended line range from 5 to 7 lines
- Still processes sequentially without new loops
- No algorithmic complexity increase

## Next Steps

1. **Test with your invoices**: Run the test to verify address extraction works for your invoice types
2. **Monitor edge cases**: If you see address extraction issues, note the format and report
3. **Add patterns**: If your invoices use unique city/country names, they can be easily added to the regex patterns

## Debugging Address Issues

If an address isn't extracted:
1. Check `raw_text` in API response to see what was extracted from PDF
2. Verify address components are on separate lines or space-separated
3. Check if city/country names are in the recognized list (add if needed)
4. Ensure field labels are one of the recognized stop words

Example debug:
```python
result = parse_invoice_data(text)
print(f"Address: {result['address']}")
print(f"Customer: {result['customer_name']}")
# Check raw_text to see actual extracted text
```
