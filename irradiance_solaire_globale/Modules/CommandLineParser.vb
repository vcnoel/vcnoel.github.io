' ============================================================================
' Module: CommandLineParser
' Description: Parseur d'arguments en ligne de commande
' ============================================================================

Namespace Modules

    Public Class CommandLineParser

        ''' <summary>
        ''' Structure contenant les paramètres parsés
        ''' </summary>
        Public Structure Parametres
            Public Latitude As Double
            Public Longitude As Double
            Public Altitude As Double
            Public DateCalcul As Date
            Public FuseauHoraire As Double
            Public IntervalleMinutes As Integer
            Public CielClair As Boolean
            Public AfficherAide As Boolean
            Public EstValide As Boolean
            Public MessageErreur As String
        End Structure

        ''' <summary>
        ''' Parse les arguments de la ligne de commande
        ''' </summary>
        Public Shared Function Parser(args As String()) As Parametres
            Dim params As New Parametres()
            
            ' Valeurs par défaut
            params.Latitude = Double.NaN
            params.Longitude = Double.NaN
            params.Altitude = 0.0
            params.DateCalcul = Date.Today
            params.FuseauHoraire = 0.0
            params.IntervalleMinutes = 30
            params.CielClair = True
            params.AfficherAide = False
            params.EstValide = True
            params.MessageErreur = ""

            Try
                Dim i As Integer = 0
                While i < args.Length
                    Select Case args(i).ToLower()
                        Case "--lat", "-lat"
                            If i + 1 < args.Length Then
                                params.Latitude = Double.Parse(args(i + 1), 
                                    System.Globalization.CultureInfo.InvariantCulture)
                                i += 1
                            End If

                        Case "--long", "-long", "--lon", "-lon"
                            If i + 1 < args.Length Then
                                params.Longitude = Double.Parse(args(i + 1), 
                                    System.Globalization.CultureInfo.InvariantCulture)
                                i += 1
                            End If

                        Case "--alt", "-alt", "--altitude"
                            If i + 1 < args.Length Then
                                params.Altitude = Double.Parse(args(i + 1), 
                                    System.Globalization.CultureInfo.InvariantCulture)
                                i += 1
                            End If

                        Case "--date", "-date"
                            If i + 1 < args.Length Then
                                params.DateCalcul = Date.Parse(args(i + 1))
                                i += 1
                            End If

                        Case "--fuseau", "-fuseau", "--tz", "-tz"
                            If i + 1 < args.Length Then
                                params.FuseauHoraire = Double.Parse(args(i + 1), 
                                    System.Globalization.CultureInfo.InvariantCulture)
                                i += 1
                            End If

                        Case "--intervalle", "-intervalle", "--interval"
                            If i + 1 < args.Length Then
                                params.IntervalleMinutes = Integer.Parse(args(i + 1))
                                i += 1
                            End If

                        Case "--nuageux", "-nuageux", "--cloudy"
                            params.CielClair = False

                        Case "--help", "-help", "-h", "/?"
                            params.AfficherAide = True

                    End Select
                    i += 1
                End While

                ' Validation des paramètres requis
                If Not params.AfficherAide Then
                    If Double.IsNaN(params.Latitude) Then
                        params.EstValide = False
                        params.MessageErreur = "Erreur: La latitude est requise (--lat)"
                    ElseIf Double.IsNaN(params.Longitude) Then
                        params.EstValide = False
                        params.MessageErreur = "Erreur: La longitude est requise (--long)"
                    ElseIf params.Latitude < -90 OrElse params.Latitude > 90 Then
                        params.EstValide = False
                        params.MessageErreur = "Erreur: La latitude doit être entre -90 et 90 degrés"
                    ElseIf params.Longitude < -180 OrElse params.Longitude > 180 Then
                        params.EstValide = False
                        params.MessageErreur = "Erreur: La longitude doit être entre -180 et 180 degrés"
                    End If
                End If

            Catch ex As Exception
                params.EstValide = False
                params.MessageErreur = $"Erreur de parsing: {ex.Message}"
            End Try

            Return params
        End Function

        ''' <summary>
        ''' Affiche l'aide d'utilisation
        ''' </summary>
        Public Shared Sub AfficherAide()
            Console.WriteLine()
            Console.WriteLine("╔═════════════════════════════════════════════════════════════════╗")
            Console.WriteLine("║           CALCULATEUR D'IRRADIANCE SOLAIRE                      ║")
            Console.WriteLine("╚═════════════════════════════════════════════════════════════════╝")
            Console.WriteLine()
            Console.WriteLine("Usage: IrradianceSolaire --lat <latitude> --long <longitude> [options]")
            Console.WriteLine()
            Console.WriteLine("Arguments requis:")
            Console.WriteLine("  --lat <valeur>       Latitude du site (-90 à 90 degrés)")
            Console.WriteLine("  --long <valeur>      Longitude du site (-180 à 180 degrés)")
            Console.WriteLine()
            Console.WriteLine("Arguments optionnels:")
            Console.WriteLine("  --alt <valeur>       Altitude en mètres (défaut: 0)")
            Console.WriteLine("  --date <YYYY-MM-DD>  Date de simulation (défaut: aujourd'hui)")
            Console.WriteLine("  --fuseau <valeur>    Fuseau horaire UTC (ex: 1 pour UTC+1)")
            Console.WriteLine("  --intervalle <min>   Intervalle de calcul en minutes (défaut: 30)")
            Console.WriteLine("  --nuageux            Utiliser les conditions de ciel nuageux")
            Console.WriteLine("  --help               Afficher cette aide")
            Console.WriteLine()
            Console.WriteLine("Exemples:")
            Console.WriteLine("  IrradianceSolaire --lat 48.8566 --long 2.3522 --alt 35 --fuseau 1")
            Console.WriteLine("  IrradianceSolaire --lat 0 --long 0 --date 2024-06-21")
            Console.WriteLine("  IrradianceSolaire --lat -16.5 --long -68.15 --alt 3640 --nuageux")
            Console.WriteLine()
        End Sub

    End Class

End Namespace
