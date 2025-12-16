' ============================================================================
' Module: SolarPosition
' Description: Calculs de la position solaire (déclinaison, angle horaire, 
'              angle zénithal, élévation solaire)
' ============================================================================

Namespace Modules

    Public Module SolarPosition

        Private Const DEG_TO_RAD As Double = Math.PI / 180.0
        Private Const RAD_TO_DEG As Double = 180.0 / Math.PI

        ''' <summary>
        ''' Calcule le jour de l'année (1-365/366)
        ''' </summary>
        Public Function CalculerJourAnnee(dateCalcul As Date) As Integer
            Return dateCalcul.DayOfYear
        End Function

        ''' <summary>
        ''' Calcule la déclinaison solaire en degrés
        ''' Formule de Cooper (1969): δ = 23.45 × sin(360/365 × (284 + n))
        ''' </summary>
        ''' <param name="jourAnnee">Jour de l'année (1-365)</param>
        ''' <returns>Déclinaison solaire en degrés (-23.45° à +23.45°)</returns>
        Public Function CalculerDeclinaisonSolaire(jourAnnee As Integer) As Double
            Dim argument As Double = (360.0 / 365.0) * (284.0 + jourAnnee) * DEG_TO_RAD
            Return 23.45 * Math.Sin(argument)
        End Function

        ''' <summary>
        ''' Calcule l'équation du temps en minutes
        ''' Corrige la différence entre temps solaire vrai et temps solaire moyen
        ''' </summary>
        Public Function CalculerEquationTemps(jourAnnee As Integer) As Double
            Dim B As Double = (360.0 / 365.0) * (jourAnnee - 81) * DEG_TO_RAD
            Return 9.87 * Math.Sin(2 * B) - 7.53 * Math.Cos(B) - 1.5 * Math.Sin(B)
        End Function

        ''' <summary>
        ''' Calcule l'heure solaire vraie à partir de l'heure locale
        ''' </summary>
        ''' <param name="heureLocale">Heure locale (0-24)</param>
        ''' <param name="longitude">Longitude du site en degrés</param>
        ''' <param name="jourAnnee">Jour de l'année</param>
        ''' <param name="fuseauHoraire">Décalage UTC en heures (ex: +1 pour Paris)</param>
        ''' <returns>Heure solaire vraie</returns>
        Public Function CalculerHeureSolaireVraie(heureLocale As Double, 
                                                   longitude As Double, 
                                                   jourAnnee As Integer,
                                                   Optional fuseauHoraire As Double = 0) As Double
            ' Méridien standard pour le fuseau horaire
            Dim meridienStandard As Double = fuseauHoraire * 15.0
            
            ' Correction de longitude (4 minutes par degré)
            Dim correctionLongitude As Double = 4.0 * (longitude - meridienStandard)
            
            ' Équation du temps
            Dim equationTemps As Double = CalculerEquationTemps(jourAnnee)
            
            ' Heure solaire vraie
            Return heureLocale + (correctionLongitude + equationTemps) / 60.0
        End Function

        ''' <summary>
        ''' Calcule l'angle horaire en degrés
        ''' ω = 15° × (heure solaire - 12)
        ''' </summary>
        ''' <param name="heureSolaire">Heure solaire vraie (0-24)</param>
        ''' <returns>Angle horaire en degrés (-180° à +180°)</returns>
        Public Function CalculerAngleHoraire(heureSolaire As Double) As Double
            Return 15.0 * (heureSolaire - 12.0)
        End Function

        ''' <summary>
        ''' Calcule l'angle zénithal solaire en degrés
        ''' cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
        ''' </summary>
        ''' <param name="latitude">Latitude du site en degrés</param>
        ''' <param name="declinaison">Déclinaison solaire en degrés</param>
        ''' <param name="angleHoraire">Angle horaire en degrés</param>
        ''' <returns>Angle zénithal en degrés (0° = soleil au zénith)</returns>
        Public Function CalculerAngleZenithal(latitude As Double, 
                                               declinaison As Double, 
                                               angleHoraire As Double) As Double
            Dim latRad As Double = latitude * DEG_TO_RAD
            Dim decRad As Double = declinaison * DEG_TO_RAD
            Dim omegaRad As Double = angleHoraire * DEG_TO_RAD

            Dim cosZenith As Double = Math.Sin(latRad) * Math.Sin(decRad) + 
                                       Math.Cos(latRad) * Math.Cos(decRad) * Math.Cos(omegaRad)
            
            ' Limiter entre -1 et 1 pour éviter les erreurs d'arrondi
            cosZenith = Math.Max(-1.0, Math.Min(1.0, cosZenith))
            
            Return Math.Acos(cosZenith) * RAD_TO_DEG
        End Function

        ''' <summary>
        ''' Calcule l'élévation solaire (complémentaire de l'angle zénithal)
        ''' α = 90° - θz
        ''' </summary>
        ''' <param name="angleZenithal">Angle zénithal en degrés</param>
        ''' <returns>Élévation solaire en degrés (0° = horizon, 90° = zénith)</returns>
        Public Function CalculerElevationSolaire(angleZenithal As Double) As Double
            Return 90.0 - angleZenithal
        End Function

        ''' <summary>
        ''' Calcule l'azimut solaire en degrés (0° = Nord, 90° = Est, 180° = Sud)
        ''' </summary>
        Public Function CalculerAzimutSolaire(latitude As Double,
                                               declinaison As Double,
                                               angleHoraire As Double,
                                               angleZenithal As Double) As Double
            Dim latRad As Double = latitude * DEG_TO_RAD
            Dim decRad As Double = declinaison * DEG_TO_RAD
            Dim zenRad As Double = angleZenithal * DEG_TO_RAD

            Dim cosAzimut As Double = (Math.Sin(decRad) - Math.Sin(latRad) * Math.Cos(zenRad)) /
                                       (Math.Cos(latRad) * Math.Sin(zenRad))
            
            cosAzimut = Math.Max(-1.0, Math.Min(1.0, cosAzimut))
            Dim azimut As Double = Math.Acos(cosAzimut) * RAD_TO_DEG

            ' Ajuster selon l'angle horaire (matin vs après-midi)
            If angleHoraire > 0 Then
                azimut = 360.0 - azimut
            End If

            Return azimut
        End Function

        ''' <summary>
        ''' Calcule les heures de lever et coucher du soleil
        ''' </summary>
        Public Function CalculerHeuresLeverCoucher(latitude As Double, 
                                                    declinaison As Double) As (Lever As Double, Coucher As Double)
            Dim latRad As Double = latitude * DEG_TO_RAD
            Dim decRad As Double = declinaison * DEG_TO_RAD
            
            ' cos(ωs) = -tan(φ)tan(δ)
            Dim cosOmegaS As Double = -Math.Tan(latRad) * Math.Tan(decRad)
            
            ' Vérifier les cas polaires
            If cosOmegaS >= 1.0 Then
                ' Nuit polaire
                Return (0, 0)
            ElseIf cosOmegaS <= -1.0 Then
                ' Jour polaire
                Return (0, 24)
            End If
            
            Dim omegaS As Double = Math.Acos(cosOmegaS) * RAD_TO_DEG
            Dim lever As Double = 12.0 - omegaS / 15.0
            Dim coucher As Double = 12.0 + omegaS / 15.0
            
            Return (lever, coucher)
        End Function

    End Module

End Namespace
