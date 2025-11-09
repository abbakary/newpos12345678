"""
Invoice upload and extraction endpoints.
Handles two-step process: extract preview → create/update records
"""

import json
import logging
from decimal import Decimal
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db import transaction

from .models import Order, Customer, Vehicle, Invoice, InvoiceLineItem, InvoicePayment, Branch
from .utils import get_user_branch
from .services import OrderService, CustomerService, VehicleService

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["POST"])
def api_extract_invoice_preview(request):
    """
    Step 1: Extract invoice data from uploaded PDF for preview.
    Returns extracted customer, order, and payment information.
    Does NOT create any records yet.
    
    POST fields:
      - file: PDF file to extract
      - selected_order_id (optional): Started order ID to link to
      - plate (optional): Vehicle plate number
      
    Returns:
      - success: true/false
      - header: Customer and payment info {invoice_no, customer_name, address, date, subtotal, tax, total}
      - items: Line items [{description, qty, value}]
      - raw_text: Full extracted text for reference
      - message: Error/status message
    """
    user_branch = get_user_branch(request.user)
    
    # Validate file upload
    uploaded = request.FILES.get('file')
    if not uploaded:
        return JsonResponse({
            'success': False,
            'message': 'No file uploaded'
        })
    
    try:
        file_bytes = uploaded.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Failed to read uploaded file'
        })
    
    # Extract text from PDF
    try:
        from tracker.utils.pdf_text_extractor import extract_from_bytes as extract_pdf_text
        extracted = extract_pdf_text(file_bytes, uploaded.name)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Failed to extract invoice data: {str(e)}',
            'error': str(e)
        })
    
    # If extraction failed
    if not extracted.get('success'):
        return JsonResponse({
            'success': False,
            'message': extracted.get('message', 'Could not extract data from PDF'),
            'error': extracted.get('error')
        })
    
    # Return extracted preview data
    header = extracted.get('header') or {}
    items = extracted.get('items') or []

    return JsonResponse({
        'success': True,
        'message': 'Invoice data extracted successfully',
        'header': {
            'invoice_no': header.get('invoice_no'),
            'code_no': header.get('code_no'),
            'customer_name': header.get('customer_name'),
            'phone': header.get('phone'),
            'email': header.get('email'),
            'address': header.get('address'),
            'reference': header.get('reference'),
            'date': header.get('date'),
            'subtotal': float(header.get('subtotal') or 0),
            'tax': float(header.get('tax') or 0),
            'total': float(header.get('total') or 0),
            'payment_method': header.get('payment_method'),
            'delivery_terms': header.get('delivery_terms'),
            'remarks': header.get('remarks'),
            'attended_by': header.get('attended_by'),
            'kind_attention': header.get('kind_attention'),
        },
        'items': [
            {
                'description': item.get('description', ''),
                'qty': int(item.get('qty', 1)) if isinstance(item.get('qty'), (int, float)) else 1,
                'unit': item.get('unit'),
                'code': item.get('code'),
                'value': float(item.get('value') or 0)
            }
            for item in items
        ],
        'raw_text': extracted.get('raw_text', '')
    })


@login_required
@require_http_methods(["POST"])
def api_create_invoice_from_upload(request):
    """
    Step 2: Create/update customer, order, and invoice from extracted invoice data.
    This is called after user confirms extracted data.
    
    POST fields:
      - selected_order_id (optional): Existing started order to update
      - plate (optional): Vehicle plate number
      
      Customer fields:
      - customer_name: Customer full name
      - customer_phone: Customer phone number
      - customer_email (optional): Customer email
      - customer_address (optional): Customer address
      - customer_type: personal|company|ngo|government
      
      Invoice fields:
      - invoice_number: Invoice number from invoice
      - invoice_date: Invoice date
      - subtotal: Subtotal amount
      - tax_amount: Tax/VAT amount
      - total_amount: Total amount
      - notes (optional): Additional notes
      
      Line items (arrays):
      - item_description[]: Item description
      - item_qty[]: Item quantity
      - item_price[]: Item unit price
      
    Returns:
      - success: true/false
      - invoice_id: Created invoice ID
      - order_id: Created/updated order ID
      - customer_id: Created/updated customer ID
      - redirect_url: URL to view created invoice
    """
    user_branch = get_user_branch(request.user)
    
    try:
        with transaction.atomic():
            # Get or create customer
            customer_name = request.POST.get('customer_name', '').strip()
            customer_phone = request.POST.get('customer_phone', '').strip()
            customer_email = request.POST.get('customer_email', '').strip() or None
            customer_address = request.POST.get('customer_address', '').strip() or None
            customer_type = request.POST.get('customer_type', 'personal')
            
            if not customer_name or not customer_phone:
                return JsonResponse({
                    'success': False,
                    'message': 'Customer name and phone are required'
                })
            
            # Try to find existing customer or create new one
            customer_obj, created = CustomerService.create_or_get_customer(
                branch=user_branch,
                full_name=customer_name,
                phone=customer_phone,
                email=customer_email,
                address=customer_address,
                customer_type=customer_type,
                create_if_missing=True
            )
            
            if not customer_obj:
                return JsonResponse({
                    'success': False,
                    'message': 'Failed to create/get customer'
                })
            
            # Get or create vehicle if plate provided
            plate = (request.POST.get('plate') or '').strip().upper() or None
            vehicle = None
            if plate:
                try:
                    vehicle = VehicleService.create_or_get_vehicle(customer=customer_obj, plate_number=plate)
                except Exception as e:
                    logger.warning(f"Failed to create/get vehicle: {e}")
                    vehicle = None
            
            # Get existing started order if provided
            selected_order_id = request.POST.get('selected_order_id')
            order = None
            if selected_order_id:
                try:
                    order = Order.objects.get(id=int(selected_order_id), branch=user_branch)
                except Exception:
                    pass
            
            # If no existing order, create new one
            if not order:
                try:
                    order = OrderService.create_order(
                        customer=customer_obj,
                        order_type='service',
                        branch=user_branch,
                        vehicle=vehicle,
                        description='Created from invoice upload'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create order: {e}")
                    return JsonResponse({
                        'success': False,
                        'message': f'Failed to create order: {str(e)}'
                    })
            else:
                # Update existing started order
                order.customer = customer_obj
                order.vehicle = vehicle or order.vehicle
                order.save()
            
            # Create invoice
            inv = Invoice()
            inv.branch = user_branch
            inv.order = order
            inv.customer = customer_obj

            # Parse invoice date
            invoice_date_str = request.POST.get('invoice_date', '')
            try:
                inv.invoice_date = datetime.strptime(invoice_date_str, '%Y-%m-%d').date() if invoice_date_str else timezone.localdate()
            except Exception:
                inv.invoice_date = timezone.localdate()

            # Set invoice fields
            inv.reference = request.POST.get('invoice_number', '').strip() or f"INV-{timezone.now().strftime('%Y%m%d%H%M%S')}"

            # Collect all notes/remarks
            notes_parts = []
            if request.POST.get('notes', '').strip():
                notes_parts.append(request.POST.get('notes', '').strip())
            if request.POST.get('remarks', '').strip():
                notes_parts.append(request.POST.get('remarks', '').strip())
            if request.POST.get('delivery_terms', '').strip():
                notes_parts.append(f"Delivery: {request.POST.get('delivery_terms', '').strip()}")
            inv.notes = ' | '.join(notes_parts) if notes_parts else ''

            # Set additional fields
            inv.attended_by = request.POST.get('attended_by', '').strip() or None
            inv.kind_attention = request.POST.get('kind_attention', '').strip() or None
            inv.remarks = request.POST.get('remarks', '').strip() or None

            # Parse amounts
            subtotal = Decimal(str(request.POST.get('subtotal', '0') or '0').replace(',', ''))
            tax_amount = Decimal(str(request.POST.get('tax_amount', '0') or '0').replace(',', ''))
            total_amount = Decimal(str(request.POST.get('total_amount', '0') or '0').replace(',', ''))

            inv.subtotal = subtotal
            inv.tax_amount = tax_amount
            inv.total_amount = total_amount or (subtotal + tax_amount)
            inv.created_by = request.user

            inv.generate_invoice_number()
            inv.save()

            # Create line items with extracted fields
            item_descriptions = request.POST.getlist('item_description[]')
            item_qtys = request.POST.getlist('item_qty[]')
            item_prices = request.POST.getlist('item_price[]')
            item_codes = request.POST.getlist('item_code[]')
            item_units = request.POST.getlist('item_unit[]')

            for idx, desc in enumerate(item_descriptions):
                if desc and desc.strip():
                    try:
                        qty = int(item_qtys[idx] or 1) if idx < len(item_qtys) else 1
                        price = Decimal(str(item_prices[idx] or '0').replace(',', '')) if idx < len(item_prices) else Decimal('0')
                        code = item_codes[idx].strip() if idx < len(item_codes) and item_codes[idx] else None
                        unit = item_units[idx].strip() if idx < len(item_units) and item_units[idx] else None

                        line = InvoiceLineItem(
                            invoice=inv,
                            code=code,
                            description=desc.strip(),
                            quantity=qty,
                            unit=unit,
                            unit_price=price
                        )
                        line.save()
                    except Exception as e:
                        logger.warning(f"Failed to create line item: {e}")
            
            # Recalculate totals
            inv.calculate_totals()
            inv.save()
            
            # Create payment record if total > 0
            if inv.total_amount > 0:
                try:
                    payment = InvoicePayment()
                    payment.invoice = inv
                    payment.amount = Decimal('0')  # Default to unpaid (amount 0)

                    # Map extracted payment method or use form value or default
                    extracted_method = request.POST.get('payment_method', '').strip().lower() or 'on_delivery'
                    payment_method_map = {
                        'cash': 'cash',
                        'cheque': 'cheque',
                        'chq': 'cheque',
                        'bank': 'bank_transfer',
                        'transfer': 'bank_transfer',
                        'card': 'card',
                        'mpesa': 'mpesa',
                        'credit': 'on_credit',
                        'delivery': 'on_delivery',
                        'cod': 'on_delivery',
                        'on_delivery': 'on_delivery',
                    }

                    # Try to match the extracted method to a valid choice
                    payment.payment_method = 'on_delivery'  # Default
                    for key, val in payment_method_map.items():
                        if key in extracted_method:
                            payment.payment_method = val
                            break

                    payment.payment_date = None
                    payment.reference = None
                    payment.save()
                except Exception as e:
                    logger.warning(f"Failed to create payment record: {e}")
            
            # Update started order with invoice data
            try:
                order = OrderService.update_order_from_invoice(
                    order=order,
                    customer=customer_obj,
                    vehicle=vehicle,
                    description=order.description
                )
            except Exception as e:
                logger.warning(f"Failed to update order from invoice: {e}")
            
            return JsonResponse({
                'success': True,
                'message': 'Invoice created and order updated successfully',
                'invoice_id': inv.id,
                'invoice_number': inv.invoice_number,
                'order_id': order.id,
                'customer_id': customer_obj.id,
                'redirect_url': f'/tracker/invoices/{inv.id}/'
            })
    
    except Exception as e:
        logger.error(f"Error creating invoice from upload: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })
