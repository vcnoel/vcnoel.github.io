# ☀️ Simulateur d'Irradiance Solaire

Calculateur modulaire d'irradiance solaire (GHI, DNI, DHI) en VB.NET, basé sur les coordonnées géographiques d'un site.

## 🚀 Installation

```bash
# Prérequis: .NET 8 SDK
# https://dotnet.microsoft.com/download

# Build
dotnet build

# Run
dotnet run -- --lat 48.8566 --long 2.3522 --alt 35 --fuseau 1
```

## 📖 Usage

```bash
IrradianceSolaire --lat <latitude> --long <longitude> [options]
```

### Arguments requis
| Argument | Description |
|----------|-------------|
| `--lat` | Latitude (-90 à 90°) |
| `--long` | Longitude (-180 à 180°) |

### Arguments optionnels
| Argument | Description | Défaut |
|----------|-------------|--------|
| `--alt` | Altitude (m) | 0 |
| `--date` | Date (YYYY-MM-DD) | Aujourd'hui |
| `--fuseau` | Fuseau horaire UTC | 0 |
| `--intervalle` | Intervalle (min) | 30 |
| `--nuageux` | Conditions nuageuses | Ciel clair |

### Exemples

```bash
# Paris, France
dotnet run -- --lat 48.8566 --long 2.3522 --alt 35 --fuseau 1

# Équateur
dotnet run -- --lat 0 --long 0

# La Paz, Bolivie (haute altitude)
dotnet run -- --lat -16.5 --long -68.15 --alt 3640 --fuseau -4

# Date spécifique (solstice d'été)
dotnet run -- --lat 48.8566 --long 2.3522 --date 2024-06-21 --fuseau 1
```

## 📊 Sortie

Le programme affiche:
- Paramètres du site
- Heures de lever/coucher du soleil
- Tableau horaire: élévation solaire, GHI, DNI, DHI
- Énergie totale journalière (kWh/m²)

```
  Heure  │ Élév. │    GHI    │    DNI    │    DHI    │ Soleil
─────────┼───────┼───────────┼───────────┼───────────┼────────
  12:00  │  17,1° │     221,0 │     427,4 │      95,5 │  ☀️
  12:30  │  17,7° │     230,2 │     445,6 │      94,6 │  ☀️
```

## 🔬 Formules

| Composant | Formule |
|-----------|---------|
| **GHI** | `DHI + DNI × cos(θz)` |
| **DNI** | `S₀ × τ^m × correction_altitude` |
| **DHI** | `S₀ × Pdiff × (1 - τ^m) × cos²(θz/2)` |

- `S₀` = 1367 W/m² (constante solaire)
- `τ` = 0.7 (transmissivité, ciel clair)
- `m` = masse d'air (Kasten-Young 1989)
- `θz` = angle zénithal

## 📁 Structure

```
├── IrradianceSolaire.vbproj
├── Program.vb
├── README.md
└── Modules/
    ├── SolarPosition.vb       # Position solaire
    ├── Atmosphere.vb          # Corrections atmosphériques
    ├── IrradianceCalculator.vb # Calculs GHI/DNI/DHI
    └── CommandLineParser.vb   # Arguments CLI
```

## 📄 License

MIT License - voir [LICENSE](LICENSE)
