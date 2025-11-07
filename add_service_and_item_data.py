#!/usr/bin/env python
import os
import sys
import django
from datetime import datetime, timedelta
import random

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pos_tracker.settings')
django.setup()

from django.contrib.auth import get_user_model
from tracker.models import Brand, InventoryItem, ServiceType, ServiceAddon
from django.utils import timezone

User = get_user_model()

def create_service_types():
    """Create sample service types"""
    service_types_data = [
        {"name": "Oil Change", "estimated_minutes": 30},
        {"name": "Brake Service", "estimated_minutes": 60},
        {"name": "Tire Rotation", "estimated_minutes": 45},
        {"name": "Engine Tune-up", "estimated_minutes": 90},
        {"name": "Transmission Service", "estimated_minutes": 120},
        {"name": "Battery Replacement", "estimated_minutes": 20},
        {"name": "Air Filter Change", "estimated_minutes": 15},
        {"name": "Wheel Alignment", "estimated_minutes": 75},
        {"name": "Suspension Repair", "estimated_minutes": 100},
        {"name": "Exhaust System Repair", "estimated_minutes": 80},
        {"name": "Radiator Flush", "estimated_minutes": 50},
        {"name": "AC Service", "estimated_minutes": 70},
        {"name": "Spark Plug Replacement", "estimated_minutes": 40},
        {"name": "Brake Pad Replacement", "estimated_minutes": 35},
        {"name": "Coolant Replacement", "estimated_minutes": 45},
        {"name": "Power Steering Fluid", "estimated_minutes": 30},
    ]
    
    service_types = []
    for service_data in service_types_data:
        service_type, created = ServiceType.objects.get_or_create(
            name=service_data["name"],
            defaults={
                "estimated_minutes": service_data["estimated_minutes"],
                "is_active": True
            }
        )
        service_types.append(service_type)
        if created:
            print(f"✓ Created service type: {service_type.name} ({service_type.estimated_minutes} mins)")
        else:
            print(f"• Service type already exists: {service_type.name}")
    
    return service_types

def create_service_addons():
    """Create sample service addons"""
    service_addons_data = [
        {"name": "Wheel Balancing", "estimated_minutes": 20},
        {"name": "Tire Installation", "estimated_minutes": 30},
        {"name": "Wheel Mounting", "estimated_minutes": 25},
        {"name": "Tire Repair", "estimated_minutes": 15},
        {"name": "Alignment Check", "estimated_minutes": 10},
        {"name": "Suspension Inspection", "estimated_minutes": 20},
        {"name": "Brake Fluid Replacement", "estimated_minutes": 25},
        {"name": "Engine Cleaning", "estimated_minutes": 40},
        {"name": "Cabin Air Filter", "estimated_minutes": 10},
        {"name": "Battery Testing", "estimated_minutes": 10},
        {"name": "Headlight Restoration", "estimated_minutes": 30},
        {"name": "Undercarriage Wash", "estimated_minutes": 15},
        {"name": "Transmission Fluid Flush", "estimated_minutes": 45},
        {"name": "Differential Service", "estimated_minutes": 50},
        {"name": "Engine Oil Top-up", "estimated_minutes": 5},
        {"name": "Windshield Treatment", "estimated_minutes": 15},
    ]
    
    service_addons = []
    for addon_data in service_addons_data:
        addon, created = ServiceAddon.objects.get_or_create(
            name=addon_data["name"],
            defaults={
                "estimated_minutes": addon_data["estimated_minutes"],
                "is_active": True
            }
        )
        service_addons.append(addon)
        if created:
            print(f"✓ Created service addon: {addon.name} ({addon.estimated_minutes} mins)")
        else:
            print(f"• Service addon already exists: {addon.name}")
    
    return service_addons

def create_brands():
    """Create sample brands"""
    brands_data = [
        {"name": "Michelin", "description": "Premium tires for all vehicles", "website": "https://www.michelin.com"},
        {"name": "Bridgestone", "description": "High-performance tires", "website": "https://www.bridgestone.com"},
        {"name": "Goodyear", "description": "Quality tires for all seasons", "website": "https://www.goodyear.com"},
        {"name": "Pirelli", "description": "Luxury and performance tires", "website": "https://www.pirelli.com"},
        {"name": "Dunlop", "description": "Reliable tires for everyday use", "website": "https://www.dunlop.com"},
        {"name": "Continental", "description": "German engineering", "website": "https://www.continental.com"},
        {"name": "Hankook", "description": "Affordable quality tires", "website": "https://www.hankook.com"},
        {"name": "Yokohama", "description": "Japanese precision tires", "website": "https://www.yokohama.com"},
        {"name": "Toyo", "description": "High performance off-road", "website": "https://www.toyotires.com"},
        {"name": "Cooper", "description": "Durable tires for trucks", "website": "https://www.coopertires.com"},
    ]
    
    brands = []
    for brand_data in brands_data:
        brand, created = Brand.objects.get_or_create(
            name=brand_data["name"],
            defaults={
                "description": brand_data["description"],
                "website": brand_data.get("website"),
                "is_active": True
            }
        )
        brands.append(brand)
        if created:
            print(f"✓ Created brand: {brand.name}")
        else:
            print(f"• Brand already exists: {brand.name}")
    
    return brands

def create_inventory_items(brands):
    """Create sample inventory items"""
    tire_types = [
        "All Season", "Summer", "Winter", "Performance", 
        "Off-Road", "Truck", "Economy", "Premium"
    ]
    
    tire_sizes = [
        "185/65R15", "195/65R15", "205/55R16", "215/55R16",
        "225/45R17", "235/45R17", "245/40R18", "255/40R19",
        "265/70R16", "275/65R17", "31x10.5R15", "33x12.5R15"
    ]
    
    items_created = 0
    items_skipped = 0
    
    for brand in brands:
        # Create 5 items per brand for variety
        for i in range(5):
            tire_type = random.choice(tire_types)
            tire_size = random.choice(tire_sizes)
            item_name = f"{brand.name} {tire_type} {tire_size}"
            
            item, created = InventoryItem.objects.get_or_create(
                name=item_name,
                brand=brand,
                defaults={
                    "description": f"{tire_type} tire size {tire_size} from {brand.name}",
                    "quantity": random.randint(5, 100),
                    "price": round(random.uniform(50, 300), 2),
                    "cost_price": round(random.uniform(30, 200), 2),
                    "sku": f"SKU-{brand.name[:3]}-{i+1:04d}".upper(),
                    "barcode": f"{random.randint(100000000000, 999999999999)}",
                    "reorder_level": random.randint(5, 20),
                    "location": f"Aisle {random.randint(1, 10)}, Shelf {random.choice('ABCDE')}",
                    "is_active": True
                }
            )
            
            if created:
                print(f"✓ Created inventory: {brand.name} - {tire_type} {tire_size}")
                items_created += 1
            else:
                items_skipped += 1
    
    print(f"\n  Inventory items created: {items_created}")
    print(f"  Inventory items skipped (already exist): {items_skipped}")
    
    return InventoryItem.objects.all()

def add_sample_data():
    """Create all sample data"""
    print("\n" + "="*60)
    print("CREATING SAMPLE DATA FOR SERVICES AND ITEMS")
    print("="*60 + "\n")
    
    print("Step 1: Creating Service Types...")
    print("-" * 40)
    service_types = create_service_types()
    
    print("\n\nStep 2: Creating Service Addons...")
    print("-" * 40)
    service_addons = create_service_addons()
    
    print("\n\nStep 3: Creating Brands...")
    print("-" * 40)
    brands = create_brands()
    
    print("\n\nStep 4: Creating Inventory Items...")
    print("-" * 40)
    inventory_items = create_inventory_items(brands)
    
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Service Types: {ServiceType.objects.count()}")
    print(f"Service Addons: {ServiceAddon.objects.count()}")
    print(f"Brands: {Brand.objects.count()}")
    print(f"Inventory Items: {InventoryItem.objects.count()}")
    print("\n✅ Sample data creation completed successfully!")
    print("="*60 + "\n")

if __name__ == '__main__':
    add_sample_data()
