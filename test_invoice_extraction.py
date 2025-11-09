#!/usr/bin/env python
"""
Test script to verify invoice extraction with the Superdoll invoice sample.
This simulates the actual PDF text extraction and parsing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tracker.utils.pdf_text_extractor import parse_invoice_data

# Sample extracted text from the Superdoll invoice (simulating PyMuPDF extraction)
sample_text = """Superdoll Trailer Manufacture Co. (T) Ltd.
P.O. Box 16541 DSM, Tel.+255-22-2860930-2863467, Fax +255-22-2865412/3, Email: stm@superdoll-tz.com,Tax ID No.100-199-157, VAT Reg. No.10-0085-15-E
P.O.BOX 2517
A01218
27/10/2025
PI-1765684
STEERING AXLE
ALIGNMENT
1 41003
NOS
Sr
No.
Item Code Description 
2180007/2861940
Proforma Invoice
Code No
Customer Name SAID SALIM BAKHRESA CO LTD
Address
Tel Fax
Del. Date
PI No.
Date
Cust Ref
Ref Date
Authorised Signatory
27/10/2025
DAR-ES-SALAAM
TANZANIA
1
Qty
Kind Attn Valued Customer
Reference
 100,000.00
Rate
 100,000.00
Value
Payment
Delivery
Net Value
Type 
 100,000.00
ex-stock
Cash/Chq on Delivery
Attended By Sales Point
Remarks Looking forward to your conformed order
NOTE 1 : Payment in TSHS accepted at the prevailing rate on the date of payment.
2 : Proforma Invoice is Valid for 2 weeks from date of Proforma.
3 : Discount is Valid only for the above Quantity.
TSH TSH
Gross Value
VAT 18,000.00
TSH 118,000.00
: : : : : : : : : : : : :
Dear Sir/Madam,
We thank you for your valued enquiry. As desired please find below our detailed best offer
: : : : : :
FOR T 964 DNA
Page 1 of 1"""

# Parse the invoice data
result = parse_invoice_data(sample_text)

# Display results
print("=" * 60)
print("INVOICE EXTRACTION TEST RESULTS")
print("=" * 60)

print("\n--- HEADER INFORMATION ---")
print(f"Invoice No (PI No): {result['invoice_no']}")
print(f"Code No: {result['code_no']}")
print(f"Date: {result['date']}")
print(f"Customer Name: {result['customer_name']}")
print(f"Address: {result['address']}")
print(f"Phone: {result['phone']}")
print(f"Email: {result['email']}")
print(f"Reference: {result['reference']}")

print("\n--- MONETARY AMOUNTS ---")
print(f"Subtotal: {result['subtotal']}")
print(f"Tax/VAT: {result['tax']}")
print(f"Total: {result['total']}")

print("\n--- ADDITIONAL FIELDS ---")
print(f"Payment Method: {result['payment_method']}")
print(f"Delivery Terms: {result['delivery_terms']}")
print(f"Remarks: {result['remarks']}")
print(f"Attended By: {result['attended_by']}")
print(f"Kind Attention: {result['kind_attention']}")

print("\n--- LINE ITEMS ---")
if result['items']:
    for idx, item in enumerate(result['items'], 1):
        print(f"\nItem {idx}:")
        print(f"  Code: {item.get('code')}")
        print(f"  Description: {item.get('description')}")
        print(f"  Type/Unit: {item.get('unit')}")
        print(f"  Qty: {item.get('qty')}")
        print(f"  Rate: {item.get('rate')}")
        print(f"  Value: {item.get('value')}")
else:
    print("No line items extracted")

print("\n" + "=" * 60)
print("EXPECTED VALUES FOR VERIFICATION:")
print("=" * 60)
print("Code No: A01218")
print("Customer Name: SAID SALIM BAKHRESA CO LTD")
print("Address: P.O.BOX 2517 DAR-ES-SALAAM TANZANIA")
print("Phone: 2180007 (first part of 2180007/2861940)")
print("Date: 27/10/2025")
print("PI No: PI-1765684")
print("Reference: FOR T 964 DNA")
print("\nMonetary:")
print("Subtotal (Net Value): 100,000.00")
print("Tax (VAT): 18,000.00")
print("Total (Gross Value): 118,000.00")

print("\nAdditional:")
print("Payment Method: on_delivery (from 'Cash/Chq on Delivery')")
print("Delivery Terms: ex-stock")
print("Remarks: Looking forward to your conformed order")
print("Attended By: Sales Point")
print("Kind Attention: Valued Customer")

print("\nLine Items:")
print("Item 1:")
print("  Code: 41003")
print("  Description: STEERING AXLE ALIGNMENT")
print("  Unit: NOS")
print("  Qty: 1")
print("  Value: 100,000.00")
