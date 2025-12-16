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

    // Simulate button
    document.getElementById('simulateBtn').addEventListener('click', runSimulation);

    // Run initial simulation
    runSimulation();

    function runSimulation() {
        const lat = parseFloat(document.getElementById('latitude').value);
        const lon = parseFloat(document.getElementById('longitude').value);
        const alt = parseFloat(document.getElementById('altitude').value);
        const tz = parseFloat(document.getElementById('timezone').value);
        const date = new Date(document.getElementById('simDate').value);
        const clearSky = document.querySelector('.toggle-btn.active').dataset.sky === 'clear';

        // Get solar info
        const dayOfYear = SolarEngine.getDayOfYear(date);
        const declination = SolarEngine.calculateSolarDeclination(dayOfYear);
        const sunTimes = SolarEngine.calculateSunriseSunset(lat, declination);

        // Update info display
        document.getElementById('sunriseTime').textContent = SolarEngine.formatHour(sunTimes.sunrise);
        document.getElementById('sunsetTime').textContent = SolarEngine.formatHour(sunTimes.sunset);
        document.getElementById('dayOfYear').textContent = dayOfYear;
        document.getElementById('declination').textContent = declination.toFixed(2) + '°';

        // Run simulation
        const results = SolarEngine.simulateDay(lat, lon, alt, date, tz, 30, clearSky);
        const totals = SolarEngine.calculateDailyTotals(results, 30);

        // Calculate max elevation (solar noon)
        const maxElevation = Math.max(...results.map(r => r.elevation));
        document.getElementById('maxElevation').textContent = maxElevation.toFixed(1) + '°';

        // Update totals
        const maxGHI = 8.0;
        document.getElementById('totalGHI').textContent = totals.ghi.toFixed(2) + ' kWh/m²';
        document.getElementById('totalDNI').textContent = totals.dni.toFixed(2) + ' kWh/m²';
        document.getElementById('totalDHI').textContent = totals.dhi.toFixed(2) + ' kWh/m²';
        document.getElementById('ghiBar').style.width = Math.min(100, (totals.ghi / maxGHI) * 100) + '%';
        document.getElementById('dniBar').style.width = Math.min(100, (totals.dni / maxGHI) * 100) + '%';
        document.getElementById('dhiBar').style.width = Math.min(100, (totals.dhi / maxGHI) * 100) + '%';

        // Update chart
        updateChart(results);

        // Update table
        updateTable(results);
    }

    function updateChart(results) {
        const ctx = document.getElementById('irradianceChart').getContext('2d');
        const labels = results.map(r => SolarEngine.formatHour(r.hour));

        if (chart) chart.destroy();

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'GHI', data: results.map(r => r.ghi), borderColor: '#22c55e', backgroundColor: 'rgba(34, 197, 94, 0.1)', fill: true, tension: 0.4 },
                    { label: 'DNI', data: results.map(r => r.dni), borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: true, tension: 0.4 },
                    { label: 'DHI', data: results.map(r => r.dhi), borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', fill: true, tension: 0.4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b6b80', maxTicksLimit: 12 } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b6b80' }, title: { display: true, text: 'W/m²', color: '#6b6b80' } }
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
                <td class="status-sun">${r.sunVisible ? '☀️' : '🌙'}</td>
            `;
            tbody.appendChild(tr);
        }
    }
});
