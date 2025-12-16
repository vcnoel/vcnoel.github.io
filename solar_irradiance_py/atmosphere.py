# ============================================================================
# Module: atmosphere.py
# Description: Atmospheric corrections (air mass, transmissivity,
#              altitude correction)
# ============================================================================

import math

# Atmospheric constants
SOLAR_CONSTANT = 1367.0  # W/m² (solar constant)
TRANSMISSIVITY_CLEAR_SKY = 0.7  # Clear sky transmissivity
ALTITUDE_CORRECTION_FACTOR = 0.000125  # +12.5% per 1000m

DEG_TO_RAD = math.pi / 180.0


def get_solar_constant() -> float:
    """Return the solar constant (extraterrestrial irradiance)."""
    return SOLAR_CONSTANT


def calculate_air_mass(zenith_angle: float) -> float:
    """
    Calculate relative air mass (Air Mass).
    Kasten-Young formula (1989) for high zenith angles.
    AM = 1 / [cos(θz) + 0.50572 × (96.07995 - θz)^(-1.6364)]
    
    Args:
        zenith_angle: Zenith angle in degrees
        
    Returns:
        Relative air mass (1.0 at zenith, ~38 at horizon)
    """
    # If sun is below horizon, return infinity
    if zenith_angle >= 90.0:
        return float('inf')
    
    zen_rad = zenith_angle * DEG_TO_RAD
    cos_zen = math.cos(zen_rad)
    
    # Kasten-Young formula for accuracy at large angles
    correction = 0.50572 * math.pow(96.07995 - zenith_angle, -1.6364)
    air_mass = 1.0 / (cos_zen + correction)
    
    return max(1.0, air_mass)


def correct_air_mass_altitude(air_mass: float, altitude: float) -> float:
    """
    Correct air mass for altitude.
    Atmospheric pressure decreases with altitude.
    
    Args:
        air_mass: Air mass at sea level
        altitude: Altitude in meters
        
    Returns:
        Air mass corrected for altitude
    """
    # Atmospheric pressure ratio: P/P0 = exp(-altitude/8500)
    pressure_ratio = math.exp(-altitude / 8500.0)
    return air_mass * pressure_ratio


def calculate_transmissivity(
    air_mass: float,
    base_transmissivity: float = TRANSMISSIVITY_CLEAR_SKY
) -> float:
    """
    Calculate atmospheric transmissivity.
    τ^m where τ is base transmissivity and m is air mass.
    
    Args:
        air_mass: Relative air mass
        base_transmissivity: Base transmissivity (0.7 clear sky, 0.4 cloudy)
        
    Returns:
        Effective transmissivity (0 to 1)
    """
    if math.isinf(air_mass) or air_mass <= 0:
        return 0.0
    
    return math.pow(base_transmissivity, air_mass)


def calculate_altitude_correction(altitude: float) -> float:
    """
    Calculate altitude correction factor for irradiance.
    Irradiance increases by about 12.5% per 1000m altitude.
    
    Args:
        altitude: Altitude in meters
        
    Returns:
        Multiplicative factor (>1 for positive altitudes)
    """
    return 1.0 + (altitude * ALTITUDE_CORRECTION_FACTOR)


def calculate_extraterrestrial_irradiance(day_of_year: int) -> float:
    """
    Calculate extraterrestrial irradiance corrected for Earth-Sun distance.
    Varies by ±3.3% throughout the year.
    
    Args:
        day_of_year: Day of year (1-365)
        
    Returns:
        Extraterrestrial irradiance in W/m²
    """
    # Earth-Sun distance correction (orbital eccentricity)
    argument = (360.0 / 365.0) * day_of_year * DEG_TO_RAD
    distance_correction = 1.0 + 0.033 * math.cos(argument)
    
    return SOLAR_CONSTANT * distance_correction


def get_diffuse_proportion(clear_sky: bool) -> float:
    """
    Estimate diffuse radiation proportion based on sky conditions.
    
    Args:
        clear_sky: True if clear sky, False if cloudy
        
    Returns:
        Proportion of global radiation that is diffuse (0.1-0.8)
    """
    if clear_sky:
        return 0.15  # 15% diffuse in clear sky
    else:
        return 0.70  # 70% diffuse in cloudy sky
