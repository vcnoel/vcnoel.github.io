# ============================================================================
# Module: irradiance_calculator.py
# Description: Solar irradiance calculations (DNI, DHI, GHI)
# ============================================================================

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

from solar_position import (
    calculate_day_of_year,
    calculate_solar_declination,
    calculate_true_solar_time,
    calculate_hour_angle,
    calculate_zenith_angle,
    calculate_solar_elevation,
)
from atmosphere import (
    calculate_extraterrestrial_irradiance,
    calculate_air_mass,
    correct_air_mass_altitude,
    calculate_transmissivity,
    calculate_altitude_correction,
    get_diffuse_proportion,
)

DEG_TO_RAD = math.pi / 180.0


@dataclass
class IrradianceResult:
    """Structure containing irradiance results."""
    dni: float = 0.0  # Direct Normal Irradiance (W/m²)
    dhi: float = 0.0  # Diffuse Horizontal Irradiance (W/m²)
    ghi: float = 0.0  # Global Horizontal Irradiance (W/m²)
    zenith_angle: float = 0.0  # Zenith angle (degrees)
    solar_elevation: float = 0.0  # Solar elevation (degrees)
    hour: float = 0.0  # Calculation hour
    sun_visible: bool = False  # Is sun above horizon?


def calculate_dni(
    extraterrestrial_irradiance: float,
    transmissivity: float,
    altitude_correction: float
) -> float:
    """
    Calculate DNI (Direct Normal Irradiance).
    DNI = S₀ × τ^m × altitude_correction
    
    Args:
        extraterrestrial_irradiance: Extraterrestrial irradiance (W/m²)
        transmissivity: Effective transmissivity
        altitude_correction: Altitude correction factor
        
    Returns:
        DNI in W/m²
    """
    if transmissivity <= 0:
        return 0.0
    
    return extraterrestrial_irradiance * transmissivity * altitude_correction


def calculate_dhi(
    extraterrestrial_irradiance: float,
    transmissivity: float,
    zenith_angle: float,
    diffuse_proportion: float
) -> float:
    """
    Calculate DHI (Diffuse Horizontal Irradiance).
    Simplified model based on diffuse proportion and zenith angle.
    
    Args:
        extraterrestrial_irradiance: Extraterrestrial irradiance (W/m²)
        transmissivity: Effective transmissivity
        zenith_angle: Zenith angle in degrees
        diffuse_proportion: Proportion of diffuse radiation
        
    Returns:
        DHI in W/m²
    """
    if zenith_angle >= 90.0:
        return 0.0
    
    # Isotropic diffuse radiation with angular weighting
    zen_rad = zenith_angle * DEG_TO_RAD
    angular_factor = math.cos(zen_rad / 2.0) ** 2
    
    # Diffuse is what's not transmitted directly
    non_transmitted_radiation = extraterrestrial_irradiance * (1.0 - transmissivity)
    
    return non_transmitted_radiation * diffuse_proportion * angular_factor


def calculate_ghi(
    dni: float,
    dhi: float,
    zenith_angle: float
) -> float:
    """
    Calculate GHI (Global Horizontal Irradiance).
    GHI = DHI + DNI × cos(θz)
    
    Args:
        dni: Direct Normal Irradiance (W/m²)
        dhi: Diffuse Horizontal Irradiance (W/m²)
        zenith_angle: Zenith angle in degrees
        
    Returns:
        GHI in W/m²
    """
    if zenith_angle >= 90.0:
        return dhi  # Only diffuse is present below horizon
    
    zen_rad = zenith_angle * DEG_TO_RAD
    direct_component = dni * math.cos(zen_rad)
    
    return dhi + direct_component


def calculate_complete_irradiance(
    latitude: float,
    longitude: float,
    altitude: float,
    date_calc: date,
    local_hour: float,
    timezone: float = 0,
    clear_sky: bool = True
) -> IrradianceResult:
    """
    Calculate all irradiances for a given moment.
    """
    result = IrradianceResult()
    result.hour = local_hour
    
    # 1. Solar position calculations
    day_of_year = calculate_day_of_year(date_calc)
    declination = calculate_solar_declination(day_of_year)
    solar_hour = calculate_true_solar_time(local_hour, longitude, day_of_year, timezone)
    hour_angle = calculate_hour_angle(solar_hour)
    zenith_angle = calculate_zenith_angle(latitude, declination, hour_angle)
    elevation = calculate_solar_elevation(zenith_angle)
    
    result.zenith_angle = zenith_angle
    result.solar_elevation = elevation
    result.sun_visible = elevation > 0
    
    # 2. If sun is below horizon, no irradiance
    if not result.sun_visible:
        result.dni = 0.0
        result.dhi = 0.0
        result.ghi = 0.0
        return result
    
    # 3. Atmospheric calculations
    extraterrestrial_irradiance = calculate_extraterrestrial_irradiance(day_of_year)
    air_mass = calculate_air_mass(zenith_angle)
    corrected_air_mass = correct_air_mass_altitude(air_mass, altitude)
    transmissivity = calculate_transmissivity(corrected_air_mass)
    altitude_correction = calculate_altitude_correction(altitude)
    diffuse_proportion = get_diffuse_proportion(clear_sky)
    
    # 4. Irradiance calculations
    result.dni = calculate_dni(extraterrestrial_irradiance, transmissivity, altitude_correction)
    result.dhi = calculate_dhi(extraterrestrial_irradiance, transmissivity, zenith_angle, diffuse_proportion)
    result.ghi = calculate_ghi(result.dni, result.dhi, zenith_angle)
    
    return result


def simulate_day(
    latitude: float,
    longitude: float,
    altitude: float,
    date_calc: date,
    timezone: float = 0,
    interval_minutes: int = 30,
    clear_sky: bool = True
) -> List[IrradianceResult]:
    """
    Simulate irradiance over an entire day.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        altitude: Altitude in meters
        date_calc: Simulation date
        timezone: UTC offset in hours
        interval_minutes: Interval between calculations (default 30 min)
        clear_sky: Sky conditions
        
    Returns:
        List of results for each time step
    """
    results = []
    
    # Simulate from 0h to 23:59
    start_hour = 0.0
    end_hour = 24.0
    step = interval_minutes / 60.0
    
    hour = start_hour
    while hour < end_hour:
        result = calculate_complete_irradiance(
            latitude, longitude, altitude,
            date_calc, hour, timezone, clear_sky
        )
        results.append(result)
        hour += step
    
    return results


def calculate_daily_total_energy(
    results: List[IrradianceResult],
    interval_minutes: int
) -> Tuple[float, float, float]:
    """
    Calculate total energy received during the day (kWh/m²).
    
    Returns:
        Tuple of (GHI, DNI, DHI) in kWh/m²
    """
    total_ghi = 0.0
    total_dni = 0.0
    total_dhi = 0.0
    
    for r in results:
        total_ghi += r.ghi
        total_dni += r.dni
        total_dhi += r.dhi
    
    # Convert to kWh/m² (W/m² × hours / 1000)
    hours_per_interval = interval_minutes / 60.0
    total_ghi = total_ghi * hours_per_interval / 1000.0
    total_dni = total_dni * hours_per_interval / 1000.0
    total_dhi = total_dhi * hours_per_interval / 1000.0
    
    return (total_ghi, total_dni, total_dhi)
