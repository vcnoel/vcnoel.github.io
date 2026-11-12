// Browser check for the redesigned pages, run from the repository root:
//
//     node tools/browser_check.mjs [--chrome PATH] [--out DIR]
//
// Needs Node 22 or later (built-in WebSocket) and a local Chrome or Edge. The
// script serves the repository on a local port, opens each page in headless
// Chrome at 360, 768 and 1280 px in light and dark themes, and reports console
// errors, uncaught exceptions, failed requests and horizontal overflow. It also
// clicks the publication filter, the theme toggle and the mobile menu, and saves
// full-page screenshots to DIR (default: a folder in the system temp directory).
// Exit status is 1 when any error is found.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdirSync, mkdtempSync, readFileSync, existsSync, statSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, resolve, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const PAGES = ['index.html', 'posts/index.html', 'posts/site-relaunch.html', 'posts/template.html'];
const WIDTHS = [360, 768, 1280];
const THEMES = ['light', 'dark'];

const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };
const OUT = resolve(opt('--out', join(tmpdir(), 'site-shots')));
const CANDIDATES = [
    opt('--chrome', ''),
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
].filter(Boolean);
const CHROME = CANDIDATES.find((p) => existsSync(p));
if (!CHROME) { console.error('No Chrome or Edge found. Pass --chrome PATH.'); process.exit(2); }

const TYPES = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.pdf': 'application/pdf', '.ico': 'image/x-icon', '.txt': 'text/plain',
};

function startServer() {
    const server = createServer((req, res) => {
        let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        if (path.endsWith('/')) path += 'index.html';
        const file = normalize(join(ROOT, path));
        if (!file.startsWith(ROOT + sep) || !existsSync(file) || !statSync(file).isFile()) {
            res.writeHead(404); res.end('not found'); return;
        }
        res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream' });
        res.end(readFileSync(file));
    });
    return new Promise((ok) => server.listen(0, '127.0.0.1', () => ok(server)));
}

function launchChrome(profile) {
    const proc = spawn(CHROME, [
        '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
        '--disable-extensions', `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    return new Promise((ok, fail) => {
        let buf = '';
        const t = setTimeout(() => fail(new Error('Chrome did not start')), 30000);
        proc.stderr.on('data', (d) => {
            buf += d.toString();
            const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
            if (m) { clearTimeout(t); ok({ proc, ws: m[1] }); }
        });
    });
}

class CDP {
    constructor(url) {
        this.ws = new WebSocket(url);
        this.id = 0;
        this.pending = new Map();
        this.handlers = [];
        this.ws.addEventListener('message', (ev) => {
            const msg = JSON.parse(ev.data);
            if (msg.id && this.pending.has(msg.id)) {
                const { ok, fail } = this.pending.get(msg.id);
                this.pending.delete(msg.id);
                if (msg.error) fail(new Error(msg.error.message)); else ok(msg.result);
            } else if (msg.method) {
                this.handlers.forEach((h) => h(msg));
            }
        });
    }
    open() { return new Promise((ok) => this.ws.addEventListener('open', ok, { once: true })); }
    send(method, params = {}, sessionId) {
        const id = ++this.id;
        this.ws.send(JSON.stringify({ id, method, params, sessionId }));
        return new Promise((ok, fail) => this.pending.set(id, { ok, fail }));
    }
    on(fn) { this.handlers.push(fn); }
    waitFor(method, sessionId, timeout = 20000) {
        return new Promise((ok) => {
            const t = setTimeout(() => ok(null), timeout);
            const h = (msg) => {
                if (msg.method === method && msg.sessionId === sessionId) {
                    clearTimeout(t);
                    this.handlers = this.handlers.filter((x) => x !== h);
                    ok(msg);
                }
            };
            this.handlers.push(h);
        });
    }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const OVERFLOW_JS = `(() => {
    const de = document.documentElement, w = de.clientWidth;
    const offenders = [...document.querySelectorAll('body *')].filter((e) => {
        const r = e.getBoundingClientRect();
        if (r.width === 0 || r.right <= w + 1) return false;
        for (let p = e.parentElement; p; p = p.parentElement) {
            const s = getComputedStyle(p);
            if (s.overflowX === 'auto' || s.overflowX === 'hidden' || s.overflowX === 'scroll') return false;
        }
        return true;
    }).slice(0, 6).map((e) => e.tagName.toLowerCase() + '.' + String(e.className || '').trim().replace(/\\s+/g, '.') + ' right=' + Math.round(e.getBoundingClientRect().right));
    return JSON.stringify({ scrollWidth: de.scrollWidth, clientWidth: w, offenders });
})()`;

async function main() {
    mkdirSync(OUT, { recursive: true });
    const server = await startServer();
    const base = `http://127.0.0.1:${server.address().port}`;
    const profile = mkdtempSync(join(tmpdir(), 'site-check-profile-'));
    const { proc, ws } = await launchChrome(profile);
    const cdp = new CDP(ws);
    await cdp.open();

    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId: s } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, s);
    await cdp.send('Runtime.enable', {}, s);
    await cdp.send('Log.enable', {}, s);
    await cdp.send('Network.enable', {}, s);

    let bucket = [];
    cdp.on((msg) => {
        if (msg.sessionId !== s) return;
        const p = msg.params || {};
        if (msg.method === 'Runtime.consoleAPICalled' && (p.type === 'error' || p.type === 'assert')) {
            bucket.push('console.' + p.type + ': ' + (p.args || []).map((a) => a.value ?? a.description ?? '').join(' '));
        } else if (msg.method === 'Runtime.exceptionThrown') {
            bucket.push('exception: ' + (p.exceptionDetails?.exception?.description || p.exceptionDetails?.text));
        } else if (msg.method === 'Log.entryAdded' && p.entry?.level === 'error') {
            bucket.push('log: ' + p.entry.text + (p.entry.url ? ' (' + p.entry.url + ')' : ''));
        } else if (msg.method === 'Network.responseReceived' && p.response?.status >= 400) {
            bucket.push('http ' + p.response.status + ': ' + p.response.url);
        } else if (msg.method === 'Network.loadingFailed' && !p.canceled) {
            bucket.push('request failed: ' + p.errorText + ' (' + p.requestId + ')');
        }
    });

    const evaluate = async (expr) => {
        const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, s);
        return r.result.value;
    };

    const errors = [];
    const report = [];
    for (const page of PAGES) {
        for (const width of WIDTHS) {
            for (const theme of THEMES) {
                bucket = [];
                await cdp.send('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: width < 800 }, s);
                await cdp.send('Emulation.setEmulatedMedia', { features: [
                    { name: 'prefers-color-scheme', value: theme },
                    { name: 'prefers-reduced-motion', value: 'reduce' },
                ] }, s);
                await evaluate("try { localStorage.removeItem('theme') } catch (e) {}");
                const loaded = cdp.waitFor('Page.loadEventFired', s);
                await cdp.send('Page.navigate', { url: `${base}/${page}` }, s);
                await loaded;
                await sleep(1200);
                const ov = JSON.parse(await evaluate(OVERFLOW_JS));
                const eff = await evaluate("document.documentElement.getAttribute('data-theme-effective')");
                const line = `${page} @${width}px ${theme}: scrollWidth ${ov.scrollWidth}/${ov.clientWidth}, theme ${eff}`;
                report.push(line);
                if (ov.scrollWidth > ov.clientWidth) errors.push(`${line}: horizontal overflow ${JSON.stringify(ov.offenders)}`);
                if (eff !== theme) errors.push(`${line}: effective theme is ${eff}, expected ${theme}`);

                if (page === 'index.html' && width === 1280 && theme === 'light') {
                    const res = JSON.parse(await evaluate(`(() => {
                        const all = document.querySelectorAll('#pub-list > li').length;
                        document.querySelector('[data-filter="agents"]').click();
                        const agents = document.querySelectorAll('#pub-list > li:not([hidden])').length;
                        const status = document.getElementById('pub-status').textContent;
                        document.querySelector('[data-filter="all"]').click();
                        const back = document.querySelectorAll('#pub-list > li:not([hidden])').length;
                        document.querySelector('[data-theme-toggle]').click();
                        const toggled = document.documentElement.getAttribute('data-theme-effective');
                        document.querySelector('[data-theme-toggle]').click();
                        const posts = document.querySelectorAll('[data-posts] .post-item').length;
                        return JSON.stringify({ all, agents, status, back, toggled, posts });
                    })()`));
                    report.push('  interactions: ' + JSON.stringify(res));
                    if (!(res.agents > 0 && res.agents < res.all && res.back === res.all)) errors.push('publication filter did not filter: ' + JSON.stringify(res));
                    if (res.toggled !== 'dark') errors.push('theme toggle did not switch to dark');
                    if (res.posts < 1) errors.push('posts list is empty');
                }
                if (width === 360 && theme === 'light') {
                    const menu = JSON.parse(await evaluate(`(() => {
                        const b = document.querySelector('.menu-toggle');
                        const before = getComputedStyle(document.getElementById('site-nav')).display;
                        b.click();
                        const after = getComputedStyle(document.getElementById('site-nav')).display;
                        const expanded = b.getAttribute('aria-expanded');
                        b.click();
                        return JSON.stringify({ before, after, expanded });
                    })()`));
                    report.push('  menu: ' + JSON.stringify(menu));
                    if (!(menu.before === 'none' && menu.after !== 'none' && menu.expanded === 'true')) errors.push(`${page}: mobile menu did not open: ${JSON.stringify(menu)}`);
                }

                await evaluate(`Promise.all([...document.querySelectorAll('img[loading="lazy"]')].map((i) => {
                    i.loading = 'eager';
                    return i.complete ? null : new Promise((r) => { i.onload = () => r(); i.onerror = () => r(); });
                })).then(() => true)`);
                const broken = await evaluate(`[...document.images].filter((i) => !i.naturalWidth).map((i) => i.getAttribute('src'))`);
                for (const b of broken) errors.push(`${page} @${width}px ${theme}: image did not load: ${b}`);
                const metrics = await cdp.send('Page.getLayoutMetrics', {}, s);
                const height = Math.ceil(metrics.cssContentSize.height);
                const TILE = 8000;
                const stem = `${page.replace(/[\\/]/g, '_').replace('.html', '')}-${width}-${theme}`;
                for (let y = 0, part = 0; y < height; y += TILE, part += 1) {
                    const shot = await cdp.send('Page.captureScreenshot', {
                        format: 'png', captureBeyondViewport: true,
                        clip: { x: 0, y, width, height: Math.min(TILE, height - y), scale: 1 },
                    }, s);
                    const name = height > TILE ? `${stem}-part${part}.png` : `${stem}.png`;
                    writeFileSync(join(OUT, name), Buffer.from(shot.data, 'base64'));
                }

                for (const b of bucket) errors.push(`${page} @${width}px ${theme}: ${b}`);
            }
        }
    }

    // Reveal on scroll, with motion allowed: blocks below the fold start pending and are all shown after a scroll.
    {
        bucket = [];
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
        await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, s);
        const loaded = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/index.html` }, s);
        await loaded;
        await sleep(800);
        const before = await evaluate("document.querySelectorAll('.reveal.is-pending').length");
        const after = await evaluate(`(async () => {
            for (let y = 0; y < document.documentElement.scrollHeight; y += 400) {
                window.scrollTo({ top: y, behavior: 'instant' });
                await new Promise((r) => setTimeout(r, 60));
            }
            await new Promise((r) => setTimeout(r, 400));
            const firstVisible = !document.querySelector('.hero').classList.contains('is-pending');
            return JSON.stringify({ pending: document.querySelectorAll('.reveal.is-pending').length, firstVisible });
        })()`);
        const res = JSON.parse(after);
        report.push(`reveal: ${before} pending at load, ${res.pending} after scrolling`);
        if (!(before > 0 && res.pending === 0)) errors.push(`reveal on scroll: ${before} pending at load, ${res.pending} after scrolling`);
        for (const b of bucket) errors.push(`index.html reveal pass: ${b}`);
    }

    await cdp.send('Browser.close').catch(() => {});
    proc.kill();
    server.close();
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* profile still locked */ }

    report.forEach((l) => console.log(l));
    console.log(`Screenshots: ${OUT}`);
    errors.forEach((e) => console.log('ERROR ' + e));
    console.log(errors.length ? `${errors.length} error(s)` : 'OK');
    process.exit(errors.length ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
