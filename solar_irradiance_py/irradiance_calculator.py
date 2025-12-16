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
    gti: float = 0.0  # Global Tilted Irradiance (W/m²)
    zenith_angle: float = 0.0  # Zenith angle (degrees)
    solar_elevation: float = 0.0  # Solar elevation (degrees)
    solar_azimuth: float = 0.0  # Solar azimuth (degrees, N=0)
    angle_of_incidence: float = 0.0  # Angle of incidence (degrees)
    hour: float = 0.0  # Calculation hour
    sun_visible: bool = False  # Is sun above horizon?


def calculate_dni(
    extraterrestrial_irradiance: float,
    transmissivity: float,
    altitude_correction: float
) -> float:
    """
    Calculate DNI (Direct Normal Irradiance).
    DNI = S0 * t^m * altitude_correction
    
    Args:
        extraterrestrial_irradiance: Extraterrestrial irradiance (W/m2)
        transmissivity: Effective transmissivity
        altitude_correction: Altitude correction factor
        
    Returns:
        DNI in W/m2
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
        extraterrestrial_irradiance: Extraterrestrial irradiance (W/m2)
        transmissivity: Effective transmissivity
        zenith_angle: Zenith angle in degrees
        diffuse_proportion: Proportion of diffuse radiation
        
    Returns:
        DHI in W/m2
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
    GHI = DHI + DNI * cos(theta_z)
    
    Args:
        dni: Direct Normal Irradiance (W/m2)
        dhi: Diffuse Horizontal Irradiance (W/m2)
        zenith_angle: Zenith angle in degrees
        
    Returns:
        GHI in W/m2
    """
    if zenith_angle >= 90.0:
        return dhi  # Only diffuse is present below horizon
    
    zen_rad = zenith_angle * DEG_TO_RAD
    direct_component = dni * math.cos(zen_rad)
    
    return dhi + direct_component


def calculate_gti(
    dni: float,
    dhi: float,
    ghi: float,
    angle_of_incidence: float,
    tilt: float,
    albedo: float = 0.2
) -> float:
    """
    Calculate Global Tilted Irradiance (GTI) using Isotropic Sky Model.
    GTI = Beam + Diffuse + Reflected
    """
    tilt_rad = tilt * DEG_TO_RAD
    aoi_rad = angle_of_incidence * DEG_TO_RAD
    
    # beam component
    beam = dni * math.cos(aoi_rad) if angle_of_incidence < 90 else 0.0
    
    # diffuse component (isotropic)
    diffuse = dhi * (1.0 + math.cos(tilt_rad)) / 2.0
    
    # reflected component
    reflected = ghi * albedo * (1.0 - math.cos(tilt_rad)) / 2.0
    
    return beam + diffuse + reflected


def calculate_complete_irradiance(
    latitude: float,
    longitude: float,
    altitude: float,
    date_calc: date,
    local_hour: float,
    timezone: float = 0,
    clear_sky: bool = True,
    tilt: float = 0.0,
    surface_azimuth: float = 180.0  # Default to South (North=0 convention)
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
    
    # Solar Azimuth conversion (solar_position uses N=0)
    from solar_position import calculate_solar_azimuth, calculate_angle_of_incidence
    solar_azimuth = calculate_solar_azimuth(latitude, declination, hour_angle, zenith_angle)
    
    result.zenith_angle = zenith_angle
    result.solar_elevation = elevation
    result.solar_azimuth = solar_azimuth
    result.sun_visible = elevation > 0
    
    # Calculate AOI
    result.angle_of_incidence = calculate_angle_of_incidence(
        zenith_angle, solar_azimuth, tilt, surface_azimuth
    )
    
    # 2. If sun is below horizon, no irradiance
    if not result.sun_visible:
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
    
    result.gti = calculate_gti(result.dni, result.dhi, result.ghi, result.angle_of_incidence, tilt)
    
    return result


def simulate_day(
    latitude: float,
    longitude: float,
    altitude: float,
    date_calc: date,
    timezone: float = 0,
    interval_minutes: int = 30,
    clear_sky: bool = True,
    tilt: float = 0.0,
    surface_azimuth: float = 180.0
) -> List[IrradianceResult]:
    """
    Simulate irradiance over an entire day.
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
            date_calc, hour, timezone, clear_sky,
            tilt, surface_azimuth
        )
        results.append(result)
        hour += step
    
    return results


def calculate_daily_total_energy(
    results: List[IrradianceResult],
    interval_minutes: int
) -> Tuple[float, float, float, float]:
    """
    Calculate total energy received during the day (kWh/m²).
    Returns: (GHI, DNI, DHI, GTI) in kWh/m²
    """
    total_ghi = 0.0
    total_dni = 0.0
    total_dhi = 0.0
    total_gti = 0.0
    
    for r in results:
        total_ghi += r.ghi
        total_dni += r.dni
        total_dhi += r.dhi
        total_gti += r.gti
    
    # Convert to kWh/m² (W/m² × hours / 1000)
    hours_per_interval = interval_minutes / 60.0
    total_ghi = total_ghi * hours_per_interval / 1000.0
    total_dni = total_dni * hours_per_interval / 1000.0
    total_dhi = total_dhi * hours_per_interval / 1000.0
    total_gti = total_gti * hours_per_interval / 1000.0
    
    return (total_ghi, total_dni, total_dhi, total_gti)
