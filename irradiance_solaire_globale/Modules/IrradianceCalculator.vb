' ============================================================================
' Module: IrradianceCalculator
' Description: Calcul des irradiances solaires (DNI, DHI, GHI)
' ============================================================================

Namespace Modules

    Public Module IrradianceCalculator

        Private Const DEG_TO_RAD As Double = Math.PI / 180.0

        ''' <summary>
        ''' Structure contenant les résultats d'irradiance
        ''' </summary>
        Public Structure ResultatIrradiance
            Public DNI As Double       ' Direct Normal Irradiance (W/m²)
            Public DHI As Double       ' Diffuse Horizontal Irradiance (W/m²)
            Public GHI As Double       ' Global Horizontal Irradiance (W/m²)
            Public AngleZenithal As Double  ' Angle zénithal (degrés)
            Public ElevationSolaire As Double  ' Élévation solaire (degrés)
            Public Heure As Double     ' Heure de calcul
            Public SoleilVisible As Boolean  ' Le soleil est-il au-dessus de l'horizon?
        End Structure

        ''' <summary>
        ''' Calcule le DNI (Direct Normal Irradiance)
        ''' DNI = S₀ × τ^m × correction_altitude
        ''' </summary>
        ''' <param name="irradianceExtra">Irradiance extraterrestre (W/m²)</param>
        ''' <param name="transmissivite">Transmissivité effective</param>
        ''' <param name="correctionAltitude">Facteur de correction d'altitude</param>
        ''' <returns>DNI en W/m²</returns>
        Public Function CalculerDNI(irradianceExtra As Double,
                                     transmissivite As Double,
                                     correctionAltitude As Double) As Double
            If transmissivite <= 0 Then
                Return 0.0
            End If

            Return irradianceExtra * transmissivite * correctionAltitude
        End Function

        ''' <summary>
        ''' Calcule le DHI (Diffuse Horizontal Irradiance)
        ''' Modèle simplifié basé sur la proportion diffuse et l'angle zénithal
        ''' </summary>
        ''' <param name="irradianceExtra">Irradiance extraterrestre (W/m²)</param>
        ''' <param name="transmissivite">Transmissivité effective</param>
        ''' <param name="angleZenithal">Angle zénithal en degrés</param>
        ''' <param name="proportionDiffuse">Proportion de rayonnement diffus</param>
        ''' <returns>DHI en W/m²</returns>
        Public Function CalculerDHI(irradianceExtra As Double,
                                     transmissivite As Double,
                                     angleZenithal As Double,
                                     proportionDiffuse As Double) As Double
            If angleZenithal >= 90.0 Then
                Return 0.0
            End If

            ' Rayonnement diffus isotrope avec pondération angulaire
            Dim zenRad As Double = angleZenithal * DEG_TO_RAD
            Dim facteurAngulaire As Double = Math.Cos(zenRad / 2.0) ^ 2

            ' Le diffus est ce qui n'est pas transmis directement
            Dim rayonnementNonTransmis As Double = irradianceExtra * (1.0 - transmissivite)
            
            Return rayonnementNonTransmis * proportionDiffuse * facteurAngulaire
        End Function

        ''' <summary>
        ''' Calcule le GHI (Global Horizontal Irradiance)
        ''' GHI = DHI + DNI × cos(θz)
        ''' </summary>
        ''' <param name="DNI">Direct Normal Irradiance (W/m²)</param>
        ''' <param name="DHI">Diffuse Horizontal Irradiance (W/m²)</param>
        ''' <param name="angleZenithal">Angle zénithal en degrés</param>
        ''' <returns>GHI en W/m²</returns>
        Public Function CalculerGHI(DNI As Double,
                                     DHI As Double,
                                     angleZenithal As Double) As Double
            If angleZenithal >= 90.0 Then
                Return DHI  ' Seul le diffus est présent sous l'horizon
            End If

            Dim zenRad As Double = angleZenithal * DEG_TO_RAD
            Dim composanteDirect As Double = DNI * Math.Cos(zenRad)

            Return DHI + composanteDirect
        End Function

        ''' <summary>
        ''' Calcule toutes les irradiances pour un instant donné
        ''' </summary>
        Public Function CalculerIrradianceComplete(latitude As Double,
                                                    longitude As Double,
                                                    altitude As Double,
                                                    dateCalcul As Date,
                                                    heureLocale As Double,
                                                    Optional fuseauHoraire As Double = 0,
                                                    Optional cielClair As Boolean = True) As ResultatIrradiance
            Dim resultat As New ResultatIrradiance()
            resultat.Heure = heureLocale

            ' 1. Calculs de position solaire
            Dim jourAnnee As Integer = SolarPosition.CalculerJourAnnee(dateCalcul)
            Dim declinaison As Double = SolarPosition.CalculerDeclinaisonSolaire(jourAnnee)
            Dim heureSolaire As Double = SolarPosition.CalculerHeureSolaireVraie(heureLocale, longitude, jourAnnee, fuseauHoraire)
            Dim angleHoraire As Double = SolarPosition.CalculerAngleHoraire(heureSolaire)
            Dim angleZenithal As Double = SolarPosition.CalculerAngleZenithal(latitude, declinaison, angleHoraire)
            Dim elevation As Double = SolarPosition.CalculerElevationSolaire(angleZenithal)

            resultat.AngleZenithal = angleZenithal
            resultat.ElevationSolaire = elevation
            resultat.SoleilVisible = (elevation > 0)

            ' 2. Si le soleil est sous l'horizon, pas d'irradiance
            If Not resultat.SoleilVisible Then
                resultat.DNI = 0.0
                resultat.DHI = 0.0
                resultat.GHI = 0.0
                Return resultat
            End If

            ' 3. Calculs atmosphériques
            Dim irradianceExtra As Double = Atmosphere.CalculerIrradianceExtraterrestre(jourAnnee)
            Dim masseAir As Double = Atmosphere.CalculerMasseAir(angleZenithal)
            Dim masseAirCorrigee As Double = Atmosphere.CorrigerMasseAirAltitude(masseAir, altitude)
            Dim transmissivite As Double = Atmosphere.CalculerTransmissivite(masseAirCorrigee)
            Dim correctionAltitude As Double = Atmosphere.CalculerCorrectionAltitude(altitude)
            Dim proportionDiffuse As Double = Atmosphere.GetProportionDiffuse(cielClair)

            ' 4. Calculs d'irradiance
            resultat.DNI = CalculerDNI(irradianceExtra, transmissivite, correctionAltitude)
            resultat.DHI = CalculerDHI(irradianceExtra, transmissivite, angleZenithal, proportionDiffuse)
            resultat.GHI = CalculerGHI(resultat.DNI, resultat.DHI, angleZenithal)

            Return resultat
        End Function

        ''' <summary>
        ''' Simule l'irradiance sur une journée entière
        ''' </summary>
        ''' <param name="latitude">Latitude en degrés</param>
        ''' <param name="longitude">Longitude en degrés</param>
        ''' <param name="altitude">Altitude en mètres</param>
        ''' <param name="dateCalcul">Date de simulation</param>
        ''' <param name="fuseauHoraire">Décalage UTC en heures</param>
        ''' <param name="intervalleMinutes">Intervalle entre les calculs (défaut 30 min)</param>
        ''' <param name="cielClair">Conditions de ciel</param>
        ''' <returns>Liste des résultats pour chaque pas de temps</returns>
        Public Function SimulerJournee(latitude As Double,
                                        longitude As Double,
                                        altitude As Double,
                                        dateCalcul As Date,
                                        Optional fuseauHoraire As Double = 0,
                                        Optional intervalleMinutes As Integer = 30,
                                        Optional cielClair As Boolean = True) As List(Of ResultatIrradiance)
            Dim resultats As New List(Of ResultatIrradiance)()
            
            ' Simuler de 0h à 23h59
            Dim heureDebut As Double = 0.0
            Dim heureFin As Double = 24.0
            Dim pas As Double = intervalleMinutes / 60.0

            Dim heure As Double = heureDebut
            While heure < heureFin
                Dim resultat = CalculerIrradianceComplete(latitude, longitude, altitude, 
                                                          dateCalcul, heure, fuseauHoraire, cielClair)
                resultats.Add(resultat)
                heure += pas
            End While

            Return resultats
        End Function

        ''' <summary>
        ''' Calcule l'énergie totale reçue sur la journée (kWh/m²)
        ''' </summary>
        Public Function CalculerEnergieTotaleJournee(resultats As List(Of ResultatIrradiance),
                                                      intervalleMinutes As Integer) As (GHI As Double, DNI As Double, DHI As Double)
            Dim totalGHI As Double = 0.0
            Dim totalDNI As Double = 0.0
            Dim totalDHI As Double = 0.0

            For Each r In resultats
                totalGHI += r.GHI
                totalDNI += r.DNI
                totalDHI += r.DHI
            Next

            ' Convertir en kWh/m² (W/m² × heures / 1000)
            Dim heuresParIntervalle As Double = intervalleMinutes / 60.0
            totalGHI = totalGHI * heuresParIntervalle / 1000.0
            totalDNI = totalDNI * heuresParIntervalle / 1000.0
            totalDHI = totalDHI * heuresParIntervalle / 1000.0

            Return (totalGHI, totalDNI, totalDHI)
        End Function

    End Module

End Namespace
