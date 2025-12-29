"""
Vehicle Information Generator and Validator
Generates car type, color, and number plate with format: BP/BG/BT-#-Letter####
"""

import random
import re
from typing import Dict, Tuple, Optional


class VehicleGenerator:
    """Generate and validate vehicle information with specific number plate format"""
    
    # Vehicle types
    CAR_TYPES = [
        'Sedan',
        'SUV',
        'Hatchback',
        'Coupe',
        'Wagon',
        'Van',
        'Pickup Truck',
        'Crossover'
    ]
    
    # Vehicle colors
    CAR_COLORS = [
        'White',
        'Black',
        'Silver',
        'Gray',
        'Red',
        'Blue',
        'Green',
        'Yellow',
        'Orange',
        'Brown',
        'Gold',
        'Beige'
    ]
    
    # Number plate prefixes (Bhutan vehicle registration codes)
    PLATE_PREFIXES = ['BP', 'BG', 'BT']
    
    # Alphabets for plate
    ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    @staticmethod
    def generate_number_plate() -> str:
        """
        Generate a random number plate with format: BP/BG/BT-#-Letter####
        
        Returns:
            str: Generated number plate (e.g., "BP-3-K1234")
        """
        prefix = random.choice(VehicleGenerator.PLATE_PREFIXES)
        first_digit = random.randint(0, 9)
        letter = random.choice(VehicleGenerator.ALPHABETS)
        last_digits = random.randint(0, 9999)
        
        # Format with leading zeros for the last 4 digits (no hyphen before digits)
        number_plate = f"{prefix}-{first_digit}-{letter}{last_digits:04d}"
        return number_plate
    
    @staticmethod
    def validate_number_plate(plate: str) -> bool:
        """
        Validate if a number plate follows the required format
        
        Args:
            plate: Number plate string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Pattern: (BP|BG|BT)-[0-9]-[A-Z][0-9]{4}
        pattern = r'^(BP|BG|BT)-[0-9]-[A-Z][0-9]{4}$'
        return bool(re.match(pattern, plate.upper()))
    
    @staticmethod
    def generate_vehicle() -> Dict[str, str]:
        """
        Generate complete vehicle information
        
        Returns:
            dict: Dictionary with car_type, color, and number_plate
        """
        return {
            'car_type': random.choice(VehicleGenerator.CAR_TYPES),
            'color': random.choice(VehicleGenerator.CAR_COLORS),
            'number_plate': VehicleGenerator.generate_number_plate()
        }
    
    @staticmethod
    def generate_multiple_vehicles(count: int = 10) -> list:
        """
        Generate multiple vehicles
        
        Args:
            count: Number of vehicles to generate
            
        Returns:
            list: List of vehicle dictionaries
        """
        return [VehicleGenerator.generate_vehicle() for _ in range(count)]
    
    @staticmethod
    def parse_number_plate(plate: str) -> Optional[Dict[str, str]]:
        """
        Parse a number plate into its components
        
        Args:
            plate: Number plate string
            
        Returns:
            dict: Dictionary with prefix, digit, letter, and suffix or None if invalid
        """
        if not VehicleGenerator.validate_number_plate(plate):
            return None
        
        parts = plate.upper().split('-')
        letter_and_digits = parts[2]  # e.g., "K1234"
        return {
            'prefix': parts[0],              # BP/BG/BT
            'digit': parts[1],               # Single digit
            'letter': letter_and_digits[0],  # Single letter
            'suffix': letter_and_digits[1:]  # 4 digits
        }


def main():
    """Demo function to showcase vehicle generation"""
    print("=" * 70)
    print("VEHICLE INFORMATION GENERATOR")
    print("Number Plate Format: BP/BG/BT-#-Letter####")
    print("=" * 70)
    print()
    
    # Generate single vehicle
    print("📌 SINGLE VEHICLE:")
    print("-" * 70)
    vehicle = VehicleGenerator.generate_vehicle()
    print(f"Car Type:      {vehicle['car_type']}")
    print(f"Color:         {vehicle['color']}")
    print(f"Number Plate:  {vehicle['number_plate']}")
    print()
    
    # Generate multiple vehicles
    print("📌 GENERATED VEHICLES (10 Random):")
    print("-" * 70)
    vehicles = VehicleGenerator.generate_multiple_vehicles(10)
    
    for i, v in enumerate(vehicles, 1):
        print(f"{i:2d}. {v['number_plate']:15s} | {v['color']:10s} | {v['car_type']}")
    print()
    
    # Validate number plates
    print("📌 NUMBER PLATE VALIDATION:")
    print("-" * 70)
    test_plates = [
        "BP-3-K1234",       # Valid
        "BG-5-Z0001",       # Valid
        "BT-9-A9999",       # Valid
        "BP-12-K1234",      # Invalid (2 digits instead of 1)
        "XX-3-K1234",       # Invalid (wrong prefix)
        "BP-3-KK1234",      # Invalid (2 letters)
        "BP-3-K123",        # Invalid (3 digits instead of 4)
        "bp-7-m5678",       # Valid (case insensitive)
        "BT-5-J5259",       # Valid
        "BP-3-K-1234",      # Invalid (extra hyphen)
    ]
    
    for plate in test_plates:
        is_valid = VehicleGenerator.validate_number_plate(plate)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{plate:20s} -> {status}")
        
        if is_valid:
            parsed = VehicleGenerator.parse_number_plate(plate)
            print(f"                     Parts: {parsed}")
    print()
    
    # Generate specific examples
    print("📌 SAMPLE VEHICLES WITH ALL PREFIXES:")
    print("-" * 70)
    for prefix in VehicleGenerator.PLATE_PREFIXES:
        digit = random.randint(0, 9)
        letter = random.choice(VehicleGenerator.ALPHABETS)
        suffix = random.randint(0, 9999)
        plate = f"{prefix}-{digit}-{letter}{suffix:04d}"
        car_type = random.choice(VehicleGenerator.CAR_TYPES)
        color = random.choice(VehicleGenerator.CAR_COLORS)
        
        print(f"{plate} | {color:10s} {car_type}")
    
    print()
    print("=" * 70)
    print("✓ Script completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
