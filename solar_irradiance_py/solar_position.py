# ============================================================================
# Module: solar_position.py
# Description: Solar position calculations (declination, hour angle,
#              zenith angle, solar elevation)
# ============================================================================

import math
from datetime import date

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def calculate_day_of_year(date_calc: date) -> int:
    """Calculate the day of year (1-365/366)."""
    return date_calc.timetuple().tm_yday


def calculate_solar_declination(day_of_year: int) -> float:
    """
    Calculate solar declination in degrees.
    Cooper formula (1969): δ = 23.45 × sin(360/365 × (284 + n))
    
    Args:
        day_of_year: Day of the year (1-365)
        
    Returns:
        Solar declination in degrees (-23.45° to +23.45°)
    """
    argument = (360.0 / 365.0) * (284.0 + day_of_year) * DEG_TO_RAD
    return 23.45 * math.sin(argument)


def calculate_equation_of_time(day_of_year: int) -> float:
    """
    Calculate the equation of time in minutes.
    Corrects the difference between true solar time and mean solar time.
    """
    B = (360.0 / 365.0) * (day_of_year - 81) * DEG_TO_RAD
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def calculate_true_solar_time(
    local_hour: float,
    longitude: float,
    day_of_year: int,
    timezone: float = 0
) -> float:
    """
    Calculate true solar time from local time.
    
    Args:
        local_hour: Local hour (0-24)
        longitude: Site longitude in degrees
        day_of_year: Day of the year
        timezone: UTC offset in hours (e.g., +1 for Paris)
        
    Returns:
        True solar time
    """
    # Standard meridian for the timezone
    standard_meridian = timezone * 15.0
    
    # Longitude correction (4 minutes per degree)
    longitude_correction = 4.0 * (longitude - standard_meridian)
    
    # Equation of time
    equation_of_time = calculate_equation_of_time(day_of_year)
    
    # True solar time
    return local_hour + (longitude_correction + equation_of_time) / 60.0


def calculate_hour_angle(solar_hour: float) -> float:
    """
    Calculate hour angle in degrees.
    ω = 15° × (solar hour - 12)
    
    Args:
        solar_hour: True solar time (0-24)
        
    Returns:
        Hour angle in degrees (-180° to +180°)
    """
    return 15.0 * (solar_hour - 12.0)


def calculate_zenith_angle(
    latitude: float,
    declination: float,
    hour_angle: float
) -> float:
    """
    Calculate solar zenith angle in degrees.
    cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
    
    Args:
        latitude: Site latitude in degrees
        declination: Solar declination in degrees
        hour_angle: Hour angle in degrees
        
    Returns:
        Zenith angle in degrees (0° = sun at zenith)
    """
    lat_rad = latitude * DEG_TO_RAD
    dec_rad = declination * DEG_TO_RAD
    omega_rad = hour_angle * DEG_TO_RAD
    
    cos_zenith = (
        math.sin(lat_rad) * math.sin(dec_rad) +
        math.cos(lat_rad) * math.cos(dec_rad) * math.cos(omega_rad)
    )
    
    # Clamp between -1 and 1 to avoid rounding errors
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    
    return math.acos(cos_zenith) * RAD_TO_DEG


def calculate_solar_elevation(zenith_angle: float) -> float:
    """
    Calculate solar elevation (complementary to zenith angle).
    α = 90° - θz
    
    Args:
        zenith_angle: Zenith angle in degrees
        
    Returns:
        Solar elevation in degrees (0° = horizon, 90° = zenith)
    """
    return 90.0 - zenith_angle


def calculate_solar_azimuth(
    latitude: float,
    declination: float,
    hour_angle: float,
    zenith_angle: float
) -> float:
    """
    Calculate solar azimuth in degrees (0° = North, 90° = East, 180° = South).
    """
    lat_rad = latitude * DEG_TO_RAD
    dec_rad = declination * DEG_TO_RAD
    zen_rad = zenith_angle * DEG_TO_RAD
    
    cos_azimuth = (
        (math.sin(dec_rad) - math.sin(lat_rad) * math.cos(zen_rad)) /
        (math.cos(lat_rad) * math.sin(zen_rad))
    )
    
    cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
    azimuth = math.acos(cos_azimuth) * RAD_TO_DEG
    
    # Adjust based on hour angle (morning vs afternoon)
    if hour_angle > 0:
        azimuth = 360.0 - azimuth
    
    return azimuth


def calculate_sunrise_sunset(
    latitude: float,
    declination: float
) -> tuple[float, float]:
    """
    Calculate sunrise and sunset hours.
    
    Returns:
        Tuple of (sunrise, sunset) hours in solar time
    """
    lat_rad = latitude * DEG_TO_RAD
    dec_rad = declination * DEG_TO_RAD
    
    # cos(ωs) = -tan(φ)tan(δ)
    cos_omega_s = -math.tan(lat_rad) * math.tan(dec_rad)
    
    # Check polar cases
    if cos_omega_s >= 1.0:
        # Polar night
        return (0, 0)
    elif cos_omega_s <= -1.0:
        # Polar day
        return (0, 24)
    
    omega_s = math.acos(cos_omega_s) * RAD_TO_DEG
    sunrise = 12.0 - omega_s / 15.0
    sunset = 12.0 + omega_s / 15.0
    
    return (sunrise, sunset)


def calculate_angle_of_incidence(
    zenith_angle: float,
    solar_azimuth: float,
    tilt: float,
    panel_azimuth: float
) -> float:
    """
    Calculate angle of incidence (AOI) on a tilted surface.
    
    Args:
        zenith_angle: Solar zenith angle in degrees
        solar_azimuth: Solar azimuth in degrees (North=0, East=90)
        tilt: Surface tilt from horizontal in degrees
        panel_azimuth: Surface azimuth in degrees (North=0, East=90)
        
    Returns:
        Angle of incidence in degrees
    """
    zen_rad = zenith_angle * DEG_TO_RAD
    sol_az_rad = solar_azimuth * DEG_TO_RAD
    tilt_rad = tilt * DEG_TO_RAD
    pan_az_rad = panel_azimuth * DEG_TO_RAD
    
    cos_theta = (
        math.cos(zen_rad) * math.cos(tilt_rad) +
        math.sin(zen_rad) * math.sin(tilt_rad) * math.cos(sol_az_rad - pan_az_rad)
    )
    
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta) * RAD_TO_DEG
