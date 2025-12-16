/**
 * Solar Irradiance Demo - Main Application
 */
document.addEventListener('DOMContentLoaded', () => {
    // Set default date to today
    const today = new Date();
    document.getElementById('simDate').valueAsDate = today;

    // Chart instance
    let chart = null;

    // City button handlers
    document.querySelectorAll('.city-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.city-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.getElementById('latitude').value = btn.dataset.lat;
            document.getElementById('longitude').value = btn.dataset.long;
            document.getElementById('altitude').value = btn.dataset.alt;
            document.getElementById('timezone').value = btn.dataset.tz;

            runSimulation();
        });
    });

    // Sky toggle handlers
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Match Lat handler
    document.getElementById('matchLatBtn').addEventListener('click', () => {
        const lat = Math.abs(parseFloat(document.getElementById('latitude').value));
        document.getElementById('tilt').value = Math.round(lat);
        runSimulation();
    });

    // Simulate button
    document.getElementById('simulateBtn').addEventListener('click', runSimulation);

    // Mode State
    let currentMode = 'standard';

    // Mode Toggle Handlers
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;

            // Toggle UI Visibility and Logic
            if (currentMode === 'standard') {
                document.getElementById('standard-mode').classList.remove('hidden');
                document.getElementById('comparison-mode').classList.add('hidden');
                document.getElementById('standard-totals').classList.remove('hidden');
                document.getElementById('comparison-totals').classList.add('hidden');

                // Auto-run for standard mode for convenience
                runSimulation();
            } else {
                document.getElementById('standard-mode').classList.add('hidden');
                document.getElementById('comparison-mode').classList.remove('hidden');
                document.getElementById('standard-totals').classList.add('hidden');
                document.getElementById('comparison-totals').classList.remove('hidden');

                // Clear chart and wait for user input
                clearResults();
            }
        });
    });

    // Run initial simulation (Standard mode default)
    runSimulation();

    function clearResults() {
        // Clear totals
        document.getElementById('totalGTIA').textContent = '-- kWh/m²';
        document.getElementById('totalGTIB').textContent = '-- kWh/m²';
        document.getElementById('totalGTIC').textContent = '-- kWh/m²';
        document.getElementById('maxElevation').textContent = '--°';

        // Clear chart
        const ctx = document.getElementById('irradianceChart').getContext('2d');
        if (chart) chart.destroy();

        // Draw empty chart with grid
        chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Hour' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'W/m²' }, min: 0, max: 1000 }
                }
            }
        });
    }

    function runSimulation() {
        const lat = parseFloat(document.getElementById('latitude').value);
        const lon = parseFloat(document.getElementById('longitude').value);
        const alt = parseFloat(document.getElementById('altitude').value);
        const tz = parseFloat(document.getElementById('timezone').value);
        const date = new Date(document.getElementById('simDate').value);
        const clearSky = document.querySelector('.toggle-btn.active').dataset.sky === 'clear';

        // Common Solar Info
        const dayOfYear = SolarEngine.getDayOfYear(date);
        const declination = SolarEngine.calculateSolarDeclination(dayOfYear);
        const sunTimes = SolarEngine.calculateSunriseSunset(lat, declination);

        document.getElementById('sunriseTime').textContent = SolarEngine.formatHour(sunTimes.sunrise);
        document.getElementById('sunsetTime').textContent = SolarEngine.formatHour(sunTimes.sunset);
        document.getElementById('dayOfYear').textContent = dayOfYear;
        document.getElementById('declination').textContent = declination.toFixed(2) + '°';

        // --- STANDARD MODE ---
        if (currentMode === 'standard') {
            const tilt = parseFloat(document.getElementById('tilt').value);
            const azimuth = parseFloat(document.getElementById('azimuth').value);

            const results = SolarEngine.simulateDay(lat, lon, alt, date, tz, 30, clearSky, tilt, azimuth);
            const totals = SolarEngine.calculateDailyTotals(results, 30);

            // Max Elevation
            const maxElevation = Math.max(...results.map(r => r.elevation));
            document.getElementById('maxElevation').textContent = maxElevation.toFixed(1) + '°';

            // Totals
            const maxTotal = Math.max(totals.ghi, totals.dni, totals.dhi, totals.gti, 8.0);
            document.getElementById('totalGHI').textContent = totals.ghi.toFixed(2) + ' kWh/m²';
            document.getElementById('totalDNI').textContent = totals.dni.toFixed(2) + ' kWh/m²';
            document.getElementById('totalDHI').textContent = totals.dhi.toFixed(2) + ' kWh/m²';
            document.getElementById('totalGTI').textContent = totals.gti.toFixed(2) + ' kWh/m²';

            document.getElementById('ghiBar').style.width = Math.min(100, (totals.ghi / maxTotal) * 100) + '%';
            document.getElementById('dniBar').style.width = Math.min(100, (totals.dni / maxTotal) * 100) + '%';
            document.getElementById('dhiBar').style.width = Math.min(100, (totals.dhi / maxTotal) * 100) + '%';
            document.getElementById('gtiBar').style.width = Math.min(100, (totals.gti / maxTotal) * 100) + '%';

            updateChartStandard(results);
            updateTable(results); // Standard Table
        }
        // --- COMPARISON MODE ---
        else {
            // Scenario A
            const tiltA = parseFloat(document.getElementById('tiltA').value);
            const azimA = parseFloat(document.getElementById('azimuthA').value);
            const resultsA = SolarEngine.simulateDay(lat, lon, alt, date, tz, 30, clearSky, tiltA, azimA);
            const totalsA = SolarEngine.calculateDailyTotals(resultsA, 30);

            // Scenario B
            const tiltB = parseFloat(document.getElementById('tiltB').value);
            const azimB = parseFloat(document.getElementById('azimuthB').value);
            const resultsB = SolarEngine.simulateDay(lat, lon, alt, date, tz, 30, clearSky, tiltB, azimB);
            const totalsB = SolarEngine.calculateDailyTotals(resultsB, 30);

            // Scenario C
            const tiltC = parseFloat(document.getElementById('tiltC').value);
            const azimC = parseFloat(document.getElementById('azimuthC').value);
            const resultsC = SolarEngine.simulateDay(lat, lon, alt, date, tz, 30, clearSky, tiltC, azimC);
            const totalsC = SolarEngine.calculateDailyTotals(resultsC, 30);

            // Update Comparison Totals
            document.getElementById('totalGTIA').textContent = totalsA.gti.toFixed(2) + ' kWh/m²';
            document.getElementById('totalGTIB').textContent = totalsB.gti.toFixed(2) + ' kWh/m²';
            document.getElementById('totalGTIC').textContent = totalsC.gti.toFixed(2) + ' kWh/m²';

            // Max Elevation (same for all)
            const maxElevation = Math.max(...resultsA.map(r => r.elevation));
            document.getElementById('maxElevation').textContent = maxElevation.toFixed(1) + '°';

            updateChartComparison(resultsA, resultsB, resultsC);
            // Table could show Comparison data, but let's default to Scenario A or clear it for now to avoid complexity
            updateTable(resultsA);
        }
    }

    function updateChartStandard(results) {
        const ctx = document.getElementById('irradianceChart').getContext('2d');
        const labels = results.map(r => SolarEngine.formatHour(r.hour));

        if (chart) chart.destroy();

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'GHI', data: results.map(r => r.ghi), borderColor: '#22c55e', backgroundColor: 'rgba(34, 197, 94, 0.1)', fill: false, tension: 0.4, borderDash: [5, 5] },
                    { label: 'DNI', data: results.map(r => r.dni), borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: false, tension: 0.4, hidden: false },
                    { label: 'DHI', data: results.map(r => r.dhi), borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', fill: true, tension: 0.4, hidden: false },
                    { label: 'GTI', data: results.map(r => r.gti), borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.2)', fill: true, tension: 0.4, borderWidth: 3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { display: true, position: 'top', align: 'end' } }, // Show legend
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b6b80', maxTicksLimit: 12 } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b6b80' }, title: { display: true, text: 'W/m²', color: '#6b6b80' } }
                }
            }
        });

        // Manually toggle legend visibility based on mode if needed, Chart.js handles it mostly.
        // For Standard mode, we want typical items.
    }

    function updateChartComparison(resultsA, resultsB, resultsC) {
        const ctx = document.getElementById('irradianceChart').getContext('2d');
        const labels = resultsA.map(r => SolarEngine.formatHour(r.hour));

        if (chart) chart.destroy();

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Scenario A (GTI)',
                        data: resultsA.map(r => r.gti),
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2
                    },
                    {
                        label: 'Scenario B (GTI)',
                        data: resultsB.map(r => r.gti),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2
                    },
                    {
                        label: 'Scenario C (GTI)',
                        data: resultsC.map(r => r.gti),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: true, position: 'top', align: 'end', labels: { color: '#94a3b8' } },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b6b80', maxTicksLimit: 12 } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b6b80' }, title: { display: true, text: 'GTI (W/m²)', color: '#6b6b80' } }
                }
            }
        });
    }

    function updateTable(results) {
        const tbody = document.getElementById('tableBody');
        tbody.innerHTML = '';

        for (const r of results) {
            if (r.hour < 5 || r.hour > 21) continue;
            const tr = document.createElement('tr');
            tr.className = r.sunVisible ? 'sun-up' : 'sun-down';
            tr.innerHTML = `
                <td>${SolarEngine.formatHour(r.hour)}</td>
                <td>${r.elevation.toFixed(1)}°</td>
                <td>${r.ghi.toFixed(1)}</td>
                <td>${r.dni.toFixed(1)}</td>
                <td>${r.dhi.toFixed(1)}</td>
                <td style="font-weight: 600; color: #ef4444">${r.gti.toFixed(1)}</td>
                <td class="status-sun">${r.sunVisible ? '☀️' : '🌙'}</td>
            `;
            tbody.appendChild(tr);
        }
    }
});
