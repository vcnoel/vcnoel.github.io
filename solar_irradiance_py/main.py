#!/usr/bin/env python3
# ============================================================================
# Program: Solar Irradiance Simulator
# Description: Entry point for solar irradiance calculation
# Usage: python main.py --lat <latitude> --long <longitude> [options]
# ============================================================================

import argparse
from datetime import date, datetime
from typing import List

from solar_position import (
    calculate_day_of_year,
    calculate_solar_declination,
    calculate_sunrise_sunset,
)
from irradiance_calculator import (
    IrradianceResult,
    simulate_day,
    calculate_daily_total_energy,
)


# ANSI color codes
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    DARK_GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_title():
    """Display program title."""
    print()
    print(f"{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.CYAN}║     ☀️  SOLAR IRRADIANCE SIMULATOR  ☀️                                  ║{Colors.RESET}")
    print(f"{Colors.CYAN}║         GHI, DNI, DHI Calculation for a Full Day                      ║{Colors.RESET}")
    print(f"{Colors.CYAN}╚═══════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()


def print_parameters(args):
    """Display simulation parameters."""
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.GREEN}                        SITE PARAMETERS{Colors.RESET}")
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"  📍 Latitude:      {args.lat:.4f}°")
    print(f"  📍 Longitude:     {args.long:.4f}°")
    print(f"  ⛰️  Altitude:      {args.alt:.0f} m")
    print(f"  📅 Date:          {args.date}")
    sign = '+' if args.timezone >= 0 else ''
    print(f"  🕐 Timezone:      UTC{sign}{args.timezone}")
    print(f"  ⏱️  Interval:      {args.interval} minutes")
    print(f"  ☁️  Conditions:    {'Clear sky' if args.clear_sky else 'Cloudy'}")
    print()


def format_hour(decimal_hour: float) -> str:
    """Format decimal hour as HH:MM."""
    hours = int(decimal_hour)
    minutes = int((decimal_hour - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


def print_results(results: List[IrradianceResult], interval_minutes: int):
    """Display hourly results as a table."""
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.GREEN}                    SOLAR IRRADIANCE (W/m²){Colors.RESET}")
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print()
    
    # Table header
    print(f"{Colors.WHITE}  Hour   │ Elev. │    GHI    │    DNI    │    DHI    │ Sun{Colors.RESET}")
    print(f"─────────┼───────┼───────────┼───────────┼───────────┼────────")
    
    # Data (show only hours between 5h and 21h for readability)
    for r in results:
        if 5.0 <= r.hour <= 21.0:
            symbol = "  ☀️" if r.sun_visible else "  🌙"
            
            if r.sun_visible:
                color = Colors.YELLOW
            else:
                color = Colors.DARK_GRAY
            
            print(f"{color}  {format_hour(r.hour)}  │ {r.solar_elevation:5.1f}° │ {r.ghi:9.1f} │ {r.dni:9.1f} │ {r.dhi:9.1f} │{symbol}{Colors.RESET}")
    
    print()


def print_totals(totals: tuple):
    """Display daily energy totals."""
    ghi, dni, dhi = totals
    
    print(f"{Colors.MAGENTA}═══════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.MAGENTA}                  TOTAL DAILY ENERGY (kWh/m²){Colors.RESET}")
    print(f"{Colors.MAGENTA}═══════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print()
    print(f"  ⚡ GHI Total (Global):    {ghi:.2f} kWh/m²")
    print(f"  ⚡ DNI Total (Direct):    {dni:.2f} kWh/m²")
    print(f"  ⚡ DHI Total (Diffuse):   {dhi:.2f} kWh/m²")
    print()
    
    # Visual progress bar for GHI
    max_ghi = 8.0  # Typical max kWh/m²
    percentage = int(min(100, (ghi / max_ghi) * 100))
    filled = percentage // 2
    
    bar_filled = '█' * filled
    bar_empty = '░' * (50 - filled)
    
    print(f"  GHI: [{Colors.YELLOW}{bar_filled}{Colors.DARK_GRAY}{bar_empty}{Colors.RESET}] {percentage}%")
    print()


def run_simulation(args):
    """Execute the complete simulation."""
    # Parse date
    if args.date:
        date_calc = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        date_calc = date.today()
    args.date = date_calc
    
    # Display parameters
    print_parameters(args)
    
    # Calculate sunrise/sunset
    day_of_year = calculate_day_of_year(date_calc)
    declination = calculate_solar_declination(day_of_year)
    sunrise, sunset = calculate_sunrise_sunset(args.lat, declination)
    
    print(f"{Colors.YELLOW}🌅 Sunrise (solar):  {format_hour(sunrise)}{Colors.RESET}")
    print(f"{Colors.YELLOW}🌇 Sunset (solar):   {format_hour(sunset)}{Colors.RESET}")
    print(f"{Colors.YELLOW}📅 Day of year: {day_of_year}{Colors.RESET}")
    print(f"{Colors.YELLOW}📐 Solar declination: {declination:.2f}°{Colors.RESET}")
    print()
    
    # Run simulation
    print("Simulation in progress...")
    print()
    
    results = simulate_day(
        args.lat,
        args.long,
        args.alt,
        date_calc,
        args.timezone,
        args.interval,
        args.clear_sky
    )
    
    # Display results
    print_results(results, args.interval)
    
    # Calculate and display daily totals
    totals = calculate_daily_total_energy(results, args.interval)
    print_totals(totals)


def main():
    parser = argparse.ArgumentParser(
        description='☀️ Solar Irradiance Simulator - Calculate GHI, DNI, DHI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paris, France
  python main.py --lat 48.8566 --long 2.3522 --alt 35 --timezone 1

  # Equator
  python main.py --lat 0 --long 0

  # La Paz, Bolivia (high altitude)
  python main.py --lat -16.5 --long -68.15 --alt 3640 --timezone -4

  # Specific date (summer solstice)
  python main.py --lat 48.8566 --long 2.3522 --date 2024-06-21 --timezone 1
"""
    )
    
    # Required arguments
    parser.add_argument('--lat', type=float, required=True,
                        help='Latitude (-90 to 90°)')
    parser.add_argument('--long', type=float, required=True,
                        help='Longitude (-180 to 180°)')
    
    # Optional arguments
    parser.add_argument('--alt', type=float, default=0,
                        help='Altitude in meters (default: 0)')
    parser.add_argument('--date', type=str, default=None,
                        help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--timezone', type=float, default=0,
                        help='Timezone UTC offset (default: 0)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Calculation interval in minutes (default: 30)')
    parser.add_argument('--cloudy', action='store_true',
                        help='Use cloudy sky conditions (default: clear sky)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (-90 <= args.lat <= 90):
        print(f"{Colors.RED}Error: Latitude must be between -90 and 90 degrees{Colors.RESET}")
        return 1
    
    if not (-180 <= args.long <= 180):
        print(f"{Colors.RED}Error: Longitude must be between -180 and 180 degrees{Colors.RESET}")
        return 1
    
    args.clear_sky = not args.cloudy
    
    print_title()
    run_simulation(args)
    
    return 0


if __name__ == "__main__":
    exit(main())
