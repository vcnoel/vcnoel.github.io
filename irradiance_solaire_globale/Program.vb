' ============================================================================
' Programme: IrradianceSolaire
' Description: Point d'entrée principal pour le calcul d'irradiance solaire
' Usage: IrradianceSolaire --lat <latitude> --long <longitude> [options]
' ============================================================================

Imports IrradianceSolaire.Modules

Module Program

    Sub Main(args As String())
        ' Afficher le titre
        AfficherTitre()

        ' Parser les arguments
        Dim params = CommandLineParser.Parser(args)

        ' Afficher l'aide si demandé
        If params.AfficherAide Then
            CommandLineParser.AfficherAide()
            Return
        End If

        ' Vérifier la validité des paramètres
        If Not params.EstValide Then
            Console.ForegroundColor = ConsoleColor.Red
            Console.WriteLine(params.MessageErreur)
            Console.ResetColor()
            Console.WriteLine("Utilisez --help pour voir l'aide.")
            Environment.ExitCode = 1
            Return
        End If

        ' Lancer la simulation
        ExecuterSimulation(params)
    End Sub

    ''' <summary>
    ''' Affiche le titre du programme
    ''' </summary>
    Private Sub AfficherTitre()
        Console.WriteLine()
        Console.ForegroundColor = ConsoleColor.Cyan
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════════════╗")
        Console.WriteLine("║     ☀️  SIMULATEUR D'IRRADIANCE SOLAIRE  ☀️                            ║")
        Console.WriteLine("║         Calcul GHI, DNI, DHI sur une journée                          ║")
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════════════╝")
        Console.ResetColor()
        Console.WriteLine()
    End Sub

    ''' <summary>
    ''' Exécute la simulation complète
    ''' </summary>
    Private Sub ExecuterSimulation(params As CommandLineParser.Parametres)
        ' Afficher les paramètres
        AfficherParametres(params)

        ' Calculer les heures de lever/coucher
        Dim jourAnnee = SolarPosition.CalculerJourAnnee(params.DateCalcul)
        Dim declinaison = SolarPosition.CalculerDeclinaisonSolaire(jourAnnee)
        Dim heuresLC = SolarPosition.CalculerHeuresLeverCoucher(params.Latitude, declinaison)
        
        Console.ForegroundColor = ConsoleColor.Yellow
        Console.WriteLine($"🌅 Lever du soleil (solaire):  {FormatHeure(heuresLC.Lever)}")
        Console.WriteLine($"🌇 Coucher du soleil (solaire): {FormatHeure(heuresLC.Coucher)}")
        Console.WriteLine($"📅 Jour de l'année: {jourAnnee}")
        Console.WriteLine($"📐 Déclinaison solaire: {declinaison:F2}°")
        Console.ResetColor()
        Console.WriteLine()

        ' Simuler la journée
        Console.WriteLine("Simulation en cours...")
        Console.WriteLine()

        Dim resultats = IrradianceCalculator.SimulerJournee(
            params.Latitude,
            params.Longitude,
            params.Altitude,
            params.DateCalcul,
            params.FuseauHoraire,
            params.IntervalleMinutes,
            params.CielClair
        )

        ' Afficher les résultats
        AfficherResultats(resultats, params.IntervalleMinutes)

        ' Calculer et afficher les totaux journaliers
        Dim totaux = IrradianceCalculator.CalculerEnergieTotaleJournee(resultats, params.IntervalleMinutes)
        AfficherTotaux(totaux)
    End Sub

    ''' <summary>
    ''' Affiche les paramètres de simulation
    ''' </summary>
    Private Sub AfficherParametres(params As CommandLineParser.Parametres)
        Console.ForegroundColor = ConsoleColor.Green
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════")
        Console.WriteLine("                        PARAMÈTRES DU SITE")
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════")
        Console.ResetColor()
        Console.WriteLine($"  📍 Latitude:      {params.Latitude:F4}°")
        Console.WriteLine($"  📍 Longitude:     {params.Longitude:F4}°")
        Console.WriteLine($"  ⛰️  Altitude:      {params.Altitude:F0} m")
        Console.WriteLine($"  📅 Date:          {params.DateCalcul:yyyy-MM-dd}")
        Console.WriteLine($"  🕐 Fuseau:        UTC{If(params.FuseauHoraire >= 0, "+", "")}{params.FuseauHoraire}")
        Console.WriteLine($"  ⏱️  Intervalle:    {params.IntervalleMinutes} minutes")
        Console.WriteLine($"  ☁️  Conditions:    {If(params.CielClair, "Ciel clair", "Nuageux")}")
        Console.WriteLine()
    End Sub

    ''' <summary>
    ''' Affiche les résultats horaires sous forme de tableau
    ''' </summary>
    Private Sub AfficherResultats(resultats As List(Of IrradianceCalculator.ResultatIrradiance), 
                                   intervalleMinutes As Integer)
        Console.ForegroundColor = ConsoleColor.Green
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════")
        Console.WriteLine("                    IRRADIANCE SOLAIRE (W/m²)")
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════")
        Console.ResetColor()
        Console.WriteLine()
        
        ' En-tête du tableau
        Console.ForegroundColor = ConsoleColor.White
        Console.WriteLine("  Heure  │ Élév. │    GHI    │    DNI    │    DHI    │ Soleil")
        Console.WriteLine("─────────┼───────┼───────────┼───────────┼───────────┼────────")
        Console.ResetColor()

        ' Données (afficher seulement les heures de jour ou proches)
        For Each r In resultats
            ' N'afficher que les heures entre 5h et 21h pour la lisibilité
            If r.Heure >= 5.0 AndAlso r.Heure <= 21.0 Then
                Dim symbole As String = If(r.SoleilVisible, "  ☀️", "  🌙")
                
                If r.SoleilVisible Then
                    Console.ForegroundColor = ConsoleColor.Yellow
                Else
                    Console.ForegroundColor = ConsoleColor.DarkGray
                End If

                Console.WriteLine($"  {FormatHeure(r.Heure)}  │ {r.ElevationSolaire,5:F1}° │ {r.GHI,9:F1} │ {r.DNI,9:F1} │ {r.DHI,9:F1} │{symbole}")
            End If
        Next

        Console.ResetColor()
        Console.WriteLine()
    End Sub

    ''' <summary>
    ''' Affiche les totaux d'énergie journalière
    ''' </summary>
    Private Sub AfficherTotaux(totaux As (GHI As Double, DNI As Double, DHI As Double))
        Console.ForegroundColor = ConsoleColor.Magenta
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════")
        Console.WriteLine("                  ÉNERGIE TOTALE JOURNALIÈRE (kWh/m²)")
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════")
        Console.ResetColor()
        Console.WriteLine()
        Console.WriteLine($"  ⚡ GHI Total (Global):    {totaux.GHI:F2} kWh/m²")
        Console.WriteLine($"  ⚡ DNI Total (Direct):    {totaux.DNI:F2} kWh/m²")
        Console.WriteLine($"  ⚡ DHI Total (Diffus):    {totaux.DHI:F2} kWh/m²")
        Console.WriteLine()
        
        ' Afficher une barre de progression visuelle pour GHI
        Dim maxGHI As Double = 8.0  ' kWh/m² max typique
        Dim pourcentage As Integer = CInt(Math.Min(100, (totaux.GHI / maxGHI) * 100))
        Dim barreRemplie As Integer = pourcentage \ 2
        
        Console.Write("  GHI: [")
        Console.ForegroundColor = ConsoleColor.Yellow
        Console.Write(New String("█"c, barreRemplie))
        Console.ForegroundColor = ConsoleColor.DarkGray
        Console.Write(New String("░"c, 50 - barreRemplie))
        Console.ResetColor()
        Console.WriteLine($"] {pourcentage}%")
        Console.WriteLine()
    End Sub

    ''' <summary>
    ''' Formate une heure décimale en HH:MM
    ''' </summary>
    Private Function FormatHeure(heureDecimale As Double) As String
        Dim heures As Integer = CInt(Math.Floor(heureDecimale))
        Dim minutes As Integer = CInt((heureDecimale - heures) * 60)
        Return $"{heures:D2}:{minutes:D2}"
    End Function

End Module
