' ============================================================================
' Module: Atmosphere
' Description: Corrections atmosphériques (masse d'air, transmissivité, 
'              correction d'altitude)
' ============================================================================

Namespace Modules

    Public Module Atmosphere

        ' Constantes atmosphériques
        Private Const SOLAR_CONSTANT As Double = 1367.0  ' W/m² (constante solaire)
        Private Const TRANSMISSIVITY_CLEAR_SKY As Double = 0.7  ' Transmissivité ciel clair
        Private Const ALTITUDE_CORRECTION_FACTOR As Double = 0.000125  ' +12.5% par 1000m

        Private Const DEG_TO_RAD As Double = Math.PI / 180.0

        ''' <summary>
        ''' Retourne la constante solaire (irradiance extraterrestre)
        ''' </summary>
        Public Function GetConstanteSolaire() As Double
            Return SOLAR_CONSTANT
        End Function

        ''' <summary>
        ''' Calcule la masse d'air relative (Air Mass)
        ''' Formule de Kasten-Young (1989) pour angles zénithaux élevés
        ''' AM = 1 / [cos(θz) + 0.50572 × (96.07995 - θz)^(-1.6364)]
        ''' </summary>
        ''' <param name="angleZenithal">Angle zénithal en degrés</param>
        ''' <returns>Masse d'air relative (1.0 au zénith, ~38 à l'horizon)</returns>
        Public Function CalculerMasseAir(angleZenithal As Double) As Double
            ' Si le soleil est sous l'horizon, retourner une valeur très élevée
            If angleZenithal >= 90.0 Then
                Return Double.PositiveInfinity
            End If

            Dim zenRad As Double = angleZenithal * DEG_TO_RAD
            Dim cosZen As Double = Math.Cos(zenRad)

            ' Formule de Kasten-Young pour précision aux grands angles
            Dim correction As Double = 0.50572 * Math.Pow(96.07995 - angleZenithal, -1.6364)
            Dim masseAir As Double = 1.0 / (cosZen + correction)

            Return Math.Max(1.0, masseAir)
        End Function

        ''' <summary>
        ''' Corrige la masse d'air pour l'altitude
        ''' La pression atmosphérique diminue avec l'altitude
        ''' </summary>
        ''' <param name="masseAir">Masse d'air au niveau de la mer</param>
        ''' <param name="altitude">Altitude en mètres</param>
        ''' <returns>Masse d'air corrigée pour l'altitude</returns>
        Public Function CorrigerMasseAirAltitude(masseAir As Double, altitude As Double) As Double
            ' Rapport de pression atmosphérique: P/P0 = exp(-altitude/8500)
            Dim rapportPression As Double = Math.Exp(-altitude / 8500.0)
            Return masseAir * rapportPression
        End Function

        ''' <summary>
        ''' Calcule la transmissivité atmosphérique
        ''' τ^m où τ est la transmissivité de base et m la masse d'air
        ''' </summary>
        ''' <param name="masseAir">Masse d'air relative</param>
        ''' <param name="transmissiviteBase">Transmissivité de base (0.7 ciel clair, 0.4 nuageux)</param>
        ''' <returns>Transmissivité effective (0 à 1)</returns>
        Public Function CalculerTransmissivite(masseAir As Double, 
                                                Optional transmissiviteBase As Double = TRANSMISSIVITY_CLEAR_SKY) As Double
            If Double.IsInfinity(masseAir) OrElse masseAir <= 0 Then
                Return 0.0
            End If

            Return Math.Pow(transmissiviteBase, masseAir)
        End Function

        ''' <summary>
        ''' Calcule le facteur de correction d'altitude pour l'irradiance
        ''' L'irradiance augmente d'environ 12.5% par 1000m d'altitude
        ''' </summary>
        ''' <param name="altitude">Altitude en mètres</param>
        ''' <returns>Facteur multiplicatif (>1 pour altitudes positives)</returns>
        Public Function CalculerCorrectionAltitude(altitude As Double) As Double
            Return 1.0 + (altitude * ALTITUDE_CORRECTION_FACTOR)
        End Function

        ''' <summary>
        ''' Calcule l'irradiance extraterrestre corrigée pour la distance Terre-Soleil
        ''' Varie de ±3.3% au cours de l'année
        ''' </summary>
        ''' <param name="jourAnnee">Jour de l'année (1-365)</param>
        ''' <returns>Irradiance extraterrestre en W/m²</returns>
        Public Function CalculerIrradianceExtraterrestre(jourAnnee As Integer) As Double
            ' Correction de distance Terre-Soleil (excentricité orbitale)
            Dim argument As Double = (360.0 / 365.0) * jourAnnee * DEG_TO_RAD
            Dim correctionDistance As Double = 1.0 + 0.033 * Math.Cos(argument)
            
            Return SOLAR_CONSTANT * correctionDistance
        End Function

        ''' <summary>
        ''' Estime la proportion de rayonnement diffus selon les conditions de ciel
        ''' </summary>
        ''' <param name="cielClair">True si ciel clair, False si nuageux</param>
        ''' <returns>Proportion de rayonnement global qui est diffus (0.1-0.8)</returns>
        Public Function GetProportionDiffuse(cielClair As Boolean) As Double
            If cielClair Then
                Return 0.15  ' 15% diffus en ciel clair
            Else
                Return 0.70  ' 70% diffus en ciel nuageux
            End If
        End Function

    End Module

End Namespace
