# ☀️ Solar Irradiance Simulator (Python)

Modular solar irradiance calculator (GHI, DNI, DHI) in Python, based on geographic coordinates.

## 🚀 Installation

```bash
# No external dependencies required - uses only Python standard library
# Requires Python 3.9+

# Run
python main.py --lat 48.8566 --long 2.3522 --alt 35 --timezone 1
```

## 📖 Usage

```bash
python main.py --lat <latitude> --long <longitude> [options]
```

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--lat` | Latitude (-90 to 90°) |
| `--long` | Longitude (-180 to 180°) |

### Optional Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--alt` | Altitude (m) | 0 |
| `--date` | Date (YYYY-MM-DD) | Today |
| `--timezone` | Timezone UTC offset | 0 |
| `--interval` | Interval (min) | 30 |
| `--cloudy` | Cloudy conditions | Clear sky |

### Examples

```bash
# Paris, France
python main.py --lat 48.8566 --long 2.3522 --alt 35 --timezone 1

# Equator
python main.py --lat 0 --long 0

# La Paz, Bolivia (high altitude)
python main.py --lat -16.5 --long -68.15 --alt 3640 --timezone -4

# Specific date (summer solstice)
python main.py --lat 48.8566 --long 2.3522 --date 2024-06-21 --timezone 1
```

## 📊 Output

The program displays:
- Site parameters
- Sunrise/sunset times
- Hourly table: solar elevation, GHI, DNI, DHI
- Total daily energy (kWh/m²)

```
  Hour   │ Elev. │    GHI    │    DNI    │    DHI    │ Sun
─────────┼───────┼───────────┼───────────┼───────────┼────────
  12:00  │  17.1° │     221.0 │     427.4 │      95.5 │  ☀️
  12:30  │  17.7° │     230.2 │     445.6 │      94.6 │  ☀️
```

## 🔬 Formulas

| Component | Formula |
|-----------|---------|
| **GHI** | `DHI + DNI × cos(θz)` |
| **DNI** | `S₀ × τ^m × altitude_correction` |
| **DHI** | `S₀ × Pdiff × (1 - τ^m) × cos²(θz/2)` |

- `S₀` = 1367 W/m² (solar constant)
- `τ` = 0.7 (transmissivity, clear sky)
- `m` = air mass (Kasten-Young 1989)
- `θz` = zenith angle

## 📁 Structure

```
solar_irradiance_py/
├── main.py                  # CLI entry point
├── solar_position.py        # Solar position calculations
├── atmosphere.py            # Atmospheric corrections
├── irradiance_calculator.py # GHI/DNI/DHI calculations
└── README.md
```

## 📄 License

MIT License
