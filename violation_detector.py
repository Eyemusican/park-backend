"""
Violation Detection Module for Smart Parking System
Detects and categorizes parking violations with severity levels
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class ViolationType:
    """Violation type constants"""
    EXPIRED_DURATION = "Expired Duration"
    DOUBLE_PARKING = "Double Parking"
    OUTSIDE_BOUNDARY = "Outside Boundary"
    NO_PERMIT = "No Valid Permit"
    UNAUTHORIZED_ZONE = "Unauthorized Zone"
    OVERSTAY = "Overstay"


class Severity:
    """Severity level constants"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ViolationRule:
    """Defines a violation rule with its conditions and severity"""
    
    def __init__(self, rule_type: str, severity: str, duration_threshold: int = None):
        self.rule_type = rule_type
        self.severity = severity
        self.duration_threshold = duration_threshold  # in minutes
        
    def check(self, **kwargs) -> bool:
        """Check if violation condition is met"""
        if self.rule_type == ViolationType.EXPIRED_DURATION:
            current_duration = kwargs.get('duration_minutes', 0)
            return current_duration >= self.duration_threshold
        elif self.rule_type == ViolationType.OVERSTAY:
            current_duration = kwargs.get('duration_minutes', 0)
            max_duration = kwargs.get('max_allowed_minutes', float('inf'))
            return current_duration > max_duration
        elif self.rule_type == ViolationType.DOUBLE_PARKING:
            return kwargs.get('multiple_vehicles', False)
        elif self.rule_type == ViolationType.OUTSIDE_BOUNDARY:
            return kwargs.get('outside_boundary', False)
        elif self.rule_type == ViolationType.NO_PERMIT:
            return not kwargs.get('has_permit', True)
        return False


class ViolationDetector:
    """
    Main violation detection system
    Monitors parking sessions and detects violations based on configured rules
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.violations = []  # List of active violations
        self.violation_history = []  # Historical record
        
    def _initialize_rules(self) -> List[ViolationRule]:
        """Initialize violation detection rules"""
        return [
            # Duration-based violations
            ViolationRule(ViolationType.EXPIRED_DURATION, Severity.LOW, duration_threshold=120),  # 2 hours
            ViolationRule(ViolationType.EXPIRED_DURATION, Severity.MEDIUM, duration_threshold=240),  # 4 hours
            ViolationRule(ViolationType.EXPIRED_DURATION, Severity.HIGH, duration_threshold=480),  # 8 hours
            
            # Overstay violations (exceeding maximum allowed parking)
            ViolationRule(ViolationType.OVERSTAY, Severity.MEDIUM, duration_threshold=180),  # 3 hours
            ViolationRule(ViolationType.OVERSTAY, Severity.HIGH, duration_threshold=300),  # 5 hours
            
            # Parking violations
            ViolationRule(ViolationType.DOUBLE_PARKING, Severity.HIGH, duration_threshold=0),
            ViolationRule(ViolationType.OUTSIDE_BOUNDARY, Severity.MEDIUM, duration_threshold=0),
            ViolationRule(ViolationType.NO_PERMIT, Severity.HIGH, duration_threshold=0),
        ]
    
    def check_duration_violation(self, slot_id: str, vehicle_id: str, 
                                  duration_minutes: float, parking_area: str,
                                  license_plate: str = None) -> Optional[Dict]:
        """
        Check for duration-based violations
        Returns violation dict if violation detected, None otherwise
        """
        # Find highest severity violation that applies
        applicable_violation = None
        highest_severity_order = {"Low": 0, "Medium": 1, "High": 2}
        current_severity_level = -1
        
        for rule in self.rules:
            if rule.rule_type == ViolationType.EXPIRED_DURATION:
                if rule.check(duration_minutes=duration_minutes):
                    severity_level = highest_severity_order.get(rule.severity, 0)
                    if severity_level > current_severity_level:
                        current_severity_level = severity_level
                        applicable_violation = {
                            'violation_id': f"V-{int(time.time() * 1000)}",
                            'slot_id': slot_id,
                            'vehicle_id': vehicle_id,
                            'license_plate': license_plate or f"VEHICLE-{vehicle_id}",
                            'violation_type': rule.rule_type,
                            'severity': rule.severity,
                            'duration_minutes': round(duration_minutes, 2),
                            'description': self._get_description(rule.rule_type, duration_minutes),
                            'detected_at': datetime.now().isoformat(),
                            'parking_area': parking_area,
                            'status': 'ACTIVE'
                        }
        
        return applicable_violation
    
    def check_overstay_violation(self, slot_id: str, vehicle_id: str,
                                  duration_minutes: float, max_allowed_minutes: int,
                                  parking_area: str, license_plate: str = None) -> Optional[Dict]:
        """Check for overstay violations (exceeding max allowed time)"""
        if duration_minutes <= max_allowed_minutes:
            return None
        
        # Determine severity based on how much over
        overstay_minutes = duration_minutes - max_allowed_minutes
        severity = Severity.LOW
        
        if overstay_minutes > 120:  # 2 hours over
            severity = Severity.HIGH
        elif overstay_minutes > 60:  # 1 hour over
            severity = Severity.MEDIUM
        
        return {
            'violation_id': f"V-{int(time.time() * 1000)}",
            'slot_id': slot_id,
            'vehicle_id': vehicle_id,
            'license_plate': license_plate or f"VEHICLE-{vehicle_id}",
            'violation_type': ViolationType.OVERSTAY,
            'severity': severity,
            'duration_minutes': round(duration_minutes, 2),
            'overstay_minutes': round(overstay_minutes, 2),
            'max_allowed_minutes': max_allowed_minutes,
            'description': f"Exceeded maximum parking time by {int(overstay_minutes)} minutes",
            'detected_at': datetime.now().isoformat(),
            'parking_area': parking_area,
            'status': 'ACTIVE'
        }
    
    def check_double_parking(self, slot_id: str, vehicle_count: int,
                             parking_area: str) -> Optional[Dict]:
        """Check for double parking violation"""
        if vehicle_count <= 1:
            return None
        
        return {
            'violation_id': f"V-{int(time.time() * 1000)}",
            'slot_id': slot_id,
            'violation_type': ViolationType.DOUBLE_PARKING,
            'severity': Severity.HIGH,
            'description': f"{vehicle_count} vehicles detected in single slot",
            'detected_at': datetime.now().isoformat(),
            'parking_area': parking_area,
            'vehicle_count': vehicle_count,
            'status': 'ACTIVE'
        }
    
    def check_boundary_violation(self, slot_id: str, vehicle_id: str,
                                  is_outside_boundary: bool, parking_area: str,
                                  license_plate: str = None) -> Optional[Dict]:
        """Check for parking outside designated boundaries"""
        if not is_outside_boundary:
            return None
        
        return {
            'violation_id': f"V-{int(time.time() * 1000)}",
            'slot_id': slot_id,
            'vehicle_id': vehicle_id,
            'license_plate': license_plate or f"VEHICLE-{vehicle_id}",
            'violation_type': ViolationType.OUTSIDE_BOUNDARY,
            'severity': Severity.MEDIUM,
            'description': "Vehicle parked over line markers",
            'detected_at': datetime.now().isoformat(),
            'parking_area': parking_area,
            'status': 'ACTIVE'
        }
    
    def check_permit_violation(self, slot_id: str, vehicle_id: str,
                                has_valid_permit: bool, parking_area: str,
                                zone_type: str = "VIP", license_plate: str = None) -> Optional[Dict]:
        """Check for unauthorized parking (no valid permit)"""
        if has_valid_permit:
            return None
        
        return {
            'violation_id': f"V-{int(time.time() * 1000)}",
            'slot_id': slot_id,
            'vehicle_id': vehicle_id,
            'license_plate': license_plate or f"VEHICLE-{vehicle_id}",
            'violation_type': ViolationType.NO_PERMIT,
            'severity': Severity.HIGH,
            'description': f"Unauthorized vehicle in {zone_type} zone",
            'detected_at': datetime.now().isoformat(),
            'parking_area': parking_area,
            'zone_type': zone_type,
            'status': 'ACTIVE'
        }
    
    def _get_description(self, violation_type: str, duration_minutes: float = None) -> str:
        """Generate human-readable violation description"""
        if violation_type == ViolationType.EXPIRED_DURATION:
            hours = int(duration_minutes / 60)
            if hours >= 8:
                return f"Vehicle parked for {hours}+ hours"
            elif hours >= 4:
                return f"Vehicle parked for {hours}+ hours (4hr limit exceeded)"
            elif hours >= 2:
                return f"Vehicle parked for {hours}+ hours (2hr limit exceeded)"
            return f"Vehicle parked for {int(duration_minutes)} minutes"
        
        return violation_type
    
    def add_violation(self, violation: Dict):
        """Add a new violation to the active list"""
        # Check if violation already exists
        existing = self._find_existing_violation(
            violation.get('slot_id'),
            violation.get('vehicle_id'),
            violation.get('violation_type')
        )
        
        if existing:
            # Update existing violation
            self.violations.remove(existing)
        
        self.violations.append(violation)
        self.violation_history.append(violation.copy())
        
        # Keep only last 100 active violations
        if len(self.violations) > 100:
            self.violations = self.violations[-100:]
    
    def _find_existing_violation(self, slot_id: str, vehicle_id: str,
                                  violation_type: str) -> Optional[Dict]:
        """Find existing violation for same slot/vehicle/type"""
        for v in self.violations:
            if (v.get('slot_id') == slot_id and 
                v.get('vehicle_id') == vehicle_id and
                v.get('violation_type') == violation_type):
                return v
        return None
    
    def resolve_violation(self, violation_id: str):
        """Mark a violation as resolved"""
        for v in self.violations:
            if v.get('violation_id') == violation_id:
                v['status'] = 'RESOLVED'
                v['resolved_at'] = datetime.now().isoformat()
                self.violations.remove(v)
                break
    
    def get_active_violations(self) -> List[Dict]:
        """Get all active violations"""
        return [v for v in self.violations if v.get('status') == 'ACTIVE']
    
    def get_violations_by_area(self, parking_area: str) -> List[Dict]:
        """Get violations for a specific parking area"""
        return [v for v in self.violations 
                if v.get('parking_area') == parking_area and v.get('status') == 'ACTIVE']
    
    def get_violation_summary(self) -> Dict:
        """Get summary statistics of violations"""
        active = self.get_active_violations()
        
        high_severity = len([v for v in active if v.get('severity') == Severity.HIGH])
        medium_severity = len([v for v in active if v.get('severity') == Severity.MEDIUM])
        low_severity = len([v for v in active if v.get('severity') == Severity.LOW])
        
        # Count unique areas affected
        affected_areas = set(v.get('parking_area') for v in active)
        
        return {
            'total_violations': len(active),
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': low_severity,
            'areas_affected': len(affected_areas),
            'affected_area_names': list(affected_areas)
        }
    
    def cleanup_old_violations(self, max_age_hours: int = 24):
        """Remove violations older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        self.violations = [
            v for v in self.violations
            if datetime.fromisoformat(v['detected_at']) > cutoff_time
        ]
