/**
 * Solar Irradiance Engine - JavaScript port
 */
const DEG_TO_RAD = Math.PI / 180.0;
const RAD_TO_DEG = 180.0 / Math.PI;
const SOLAR_CONSTANT = 1367.0;

function getDayOfYear(date) {
    const start = new Date(date.getFullYear(), 0, 0);
    return Math.floor((date - start) / (1000 * 60 * 60 * 24));
}

function calculateSolarDeclination(dayOfYear) {
    const arg = (360.0 / 365.0) * (284.0 + dayOfYear) * DEG_TO_RAD;
    return 23.45 * Math.sin(arg);
}

function calculateEquationOfTime(dayOfYear) {
    const B = (360.0 / 365.0) * (dayOfYear - 81) * DEG_TO_RAD;
    return 9.87 * Math.sin(2 * B) - 7.53 * Math.cos(B) - 1.5 * Math.sin(B);
}

function calculateTrueSolarTime(localHour, longitude, dayOfYear, timezone) {
    const standardMeridian = timezone * 15.0;
    const longitudeCorrection = 4.0 * (longitude - standardMeridian);
    const eot = calculateEquationOfTime(dayOfYear);
    return localHour + (longitudeCorrection + eot) / 60.0;
}

function calculateZenithAngle(latitude, declination, hourAngle) {
    const latRad = latitude * DEG_TO_RAD;
    const decRad = declination * DEG_TO_RAD;
    const omegaRad = hourAngle * DEG_TO_RAD;
    let cosZen = Math.sin(latRad) * Math.sin(decRad) +
        Math.cos(latRad) * Math.cos(decRad) * Math.cos(omegaRad);
    cosZen = Math.max(-1.0, Math.min(1.0, cosZen));
    return Math.acos(cosZen) * RAD_TO_DEG;
}

function calculateSunriseSunset(latitude, declination) {
    const latRad = latitude * DEG_TO_RAD;
    const decRad = declination * DEG_TO_RAD;
    const cosOmegaS = -Math.tan(latRad) * Math.tan(decRad);
    if (cosOmegaS >= 1.0) return { sunrise: 0, sunset: 0 };
    if (cosOmegaS <= -1.0) return { sunrise: 0, sunset: 24 };
    const omegaS = Math.acos(cosOmegaS) * RAD_TO_DEG;
    return { sunrise: 12.0 - omegaS / 15.0, sunset: 12.0 + omegaS / 15.0 };
}

function calculateAirMass(zenithAngle) {
    if (zenithAngle >= 90.0) return Infinity;
    const zenRad = zenithAngle * DEG_TO_RAD;
    const correction = 0.50572 * Math.pow(96.07995 - zenithAngle, -1.6364);
    return Math.max(1.0, 1.0 / (Math.cos(zenRad) + correction));
}

function calculateExtraterrestrialIrradiance(dayOfYear) {
    const arg = (360.0 / 365.0) * dayOfYear * DEG_TO_RAD;
    return SOLAR_CONSTANT * (1.0 + 0.033 * Math.cos(arg));
}

function calculateIrradiance(lat, lon, alt, date, hour, tz, clearSky) {
    const dayOfYear = getDayOfYear(date);
    const declination = calculateSolarDeclination(dayOfYear);
    const solarHour = calculateTrueSolarTime(hour, lon, dayOfYear, tz);
    const hourAngle = 15.0 * (solarHour - 12.0);
    const zenithAngle = calculateZenithAngle(lat, declination, hourAngle);
    const elevation = 90.0 - zenithAngle;

    const result = { hour, zenithAngle, elevation, sunVisible: elevation > 0, dni: 0, dhi: 0, ghi: 0 };
    if (!result.sunVisible) return result;

    const extraIrr = calculateExtraterrestrialIrradiance(dayOfYear);
    const airMass = calculateAirMass(zenithAngle);
    const correctedAM = airMass * Math.exp(-alt / 8500.0);
    const baseT = clearSky ? 0.7 : 0.4;
    const transmissivity = Math.pow(baseT, correctedAM);
    const altCorrection = 1.0 + (alt * 0.000125);
    const diffuseProp = clearSky ? 0.15 : 0.70;

    result.dni = extraIrr * transmissivity * altCorrection;
    const zenRad = zenithAngle * DEG_TO_RAD;
    result.dhi = extraIrr * (1 - transmissivity) * diffuseProp * Math.pow(Math.cos(zenRad / 2), 2);
    result.ghi = result.dhi + result.dni * Math.cos(zenRad);
    return result;
}

function simulateDay(lat, lon, alt, date, tz, interval, clearSky) {
    const results = [];
    for (let h = 0; h < 24; h += interval / 60.0) {
        results.push(calculateIrradiance(lat, lon, alt, date, h, tz, clearSky));
    }
    return results;
}

function calculateDailyTotals(results, interval) {
    let ghi = 0, dni = 0, dhi = 0;
    for (const r of results) { ghi += r.ghi; dni += r.dni; dhi += r.dhi; }
    const h = interval / 60.0;
    return { ghi: ghi * h / 1000, dni: dni * h / 1000, dhi: dhi * h / 1000 };
}

function formatHour(h) {
    const hrs = Math.floor(h);
    const mins = Math.floor((h - hrs) * 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
}

window.SolarEngine = {
    getDayOfYear, calculateSolarDeclination, calculateSunriseSunset,
    simulateDay, calculateDailyTotals, formatHour
};
