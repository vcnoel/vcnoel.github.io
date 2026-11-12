// Browser check for the redesigned pages, run from the repository root:
//
//     node tools/browser_check.mjs [--chrome PATH] [--out DIR] [--widths 360,390,414,768,1280]
//
// Needs Node 22 or later (built-in WebSocket) and a local Chrome or Edge. The
// script serves the repository on a local port and opens each page in headless
// Chrome at every width, in light and dark themes. It reports console errors,
// uncaught exceptions, failed requests, images that did not load and horizontal
// overflow. At widths up to 820 px it also checks that no visible text is under
// 14 px and that buttons, toggles and standalone links are at least 44 px, and
// it operates the menu, the toggles and the filter with touch events. One
// full-page screenshot per page, width and theme goes to DIR (default
// tools/screens, which is not tracked). Exit status is 1 when any error is found.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdirSync, mkdtempSync, readFileSync, existsSync, statSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, resolve, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = resolve(fileURLToPath(new URL('..', import.meta.url)));
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };
// --root serves another folder, for example a built docs site, with the same checks.
const ROOT = resolve(opt('--root', REPO));
const PREFIX = opt('--prefix', '');
const PAGES = opt('--pages', 'index.html,posts/index.html,posts/site-relaunch.html,posts/template.html').split(',');
const WIDTHS = opt('--widths', '360,390,414,768,1280').split(',').map(Number);
const THEMES = ['light', 'dark'];
const OUT = resolve(opt('--out', join(REPO, 'tools', 'screens')));
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
    '.gif': 'image/gif', '.webm': 'video/webm', '.mp4': 'video/mp4',
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

const LAYOUT_JS = `(() => {
    const de = document.documentElement, w = de.clientWidth;
    const visible = (e) => e.getClientRects().length > 0 && (e.checkVisibility ? e.checkVisibility({ visibilityProperty: true, opacityProperty: true }) : true);
    const name = (e) => e.tagName.toLowerCase() + (e.className && e.className.baseVal === undefined ? '.' + String(e.className).trim().replace(/\\s+/g, '.') : '');
    const clipped = (e) => {
        for (let p = e.parentElement; p; p = p.parentElement) {
            const s = getComputedStyle(p);
            if (s.overflowX === 'auto' || s.overflowX === 'hidden' || s.overflowX === 'scroll') return true;
        }
        return false;
    };
    const offenders = [...document.querySelectorAll('body *')].filter((e) => {
        const r = e.getBoundingClientRect();
        return r.width > 0 && r.right > w + 1 && !clipped(e);
    }).slice(0, 6).map((e) => name(e) + ' right=' + Math.round(e.getBoundingClientRect().right));
    const small = new Set();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
        const n = walker.currentNode, e = n.parentElement;
        if (!n.nodeValue.trim() || !e || !visible(e) || e.closest('.visually-hidden, svg, script, style')) continue;
        const fs = parseFloat(getComputedStyle(e).fontSize);
        if (fs < 13.99) small.add(name(e) + ' ' + fs.toFixed(1) + 'px "' + n.nodeValue.trim().slice(0, 24) + '"');
    }
    const taps = [];
    document.querySelectorAll('button, summary, a[href]').forEach((e) => {
        if (!visible(e) || e.closest('.visually-hidden')) return;
        if (e.tagName === 'A' && getComputedStyle(e).display === 'inline') return;
        const r = e.getBoundingClientRect();
        if (r.height < 43.5 || r.width < 43.5) taps.push(name(e) + ' ' + Math.round(r.width) + 'x' + Math.round(r.height));
    });
    const panel = document.querySelector('.code-panel');
    const code = document.querySelector('.code-panel pre[aria-label="Python example"]');
    const strip = document.querySelector('.strip');
    return JSON.stringify({
        scrollWidth: de.scrollWidth, clientWidth: w, height: de.scrollHeight, offenders,
        small: [...small].slice(0, 8), taps: taps.slice(0, 8),
        demo: code ? { codeScrolls: code.scrollWidth > code.clientWidth, codeWidth: Math.round(code.getBoundingClientRect().width),
            panelWidth: Math.round(panel.getBoundingClientRect().width),
            stripWidth: strip ? Math.round(strip.getBoundingClientRect().width) : 0 } : null,
    });
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
    // Tap an element with touch events, as a phone would.
    const tap = async (selector) => {
        const pt = JSON.parse(await evaluate(`(() => {
            const e = document.querySelector(${JSON.stringify(selector)});
            if (!e) return 'null';
            e.scrollIntoView({ block: 'center', behavior: 'instant' });
            const r = e.getBoundingClientRect();
            return JSON.stringify({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
        })()`));
        if (!pt) return false;
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: pt.x, y: pt.y }] }, s);
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }, s);
        await sleep(250);
        return true;
    };

    const errors = [];
    const report = [];
    const shots = [];
    for (const page of PAGES) {
        for (const width of WIDTHS) {
            const mobile = width <= 820;
            for (const theme of THEMES) {
                bucket = [];
                const label = `${page} @${width}px ${theme}`;
                await cdp.send('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile }, s);
                await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: mobile }, s);
                await cdp.send('Emulation.setEmulatedMedia', { features: [
                    { name: 'prefers-color-scheme', value: theme },
                    { name: 'prefers-reduced-motion', value: 'reduce' },
                ] }, s);
                await evaluate("try { localStorage.removeItem('theme') } catch (e) {}");
                const loaded = cdp.waitFor('Page.loadEventFired', s);
                await cdp.send('Page.navigate', { url: `${base}/${page}` }, s);
                await loaded;
                await sleep(1000);
                await evaluate(`Promise.all([...document.querySelectorAll('img[loading="lazy"]')].map((i) => {
                    i.loading = 'eager';
                    return i.complete ? null : new Promise((r) => { i.onload = () => r(); i.onerror = () => r(); });
                })).then(() => true)`);
                const broken = await evaluate(`[...document.images].filter((i) => !i.naturalWidth).map((i) => i.getAttribute('src'))`);
                for (const b of broken) errors.push(`${label}: image did not load: ${b}`);

                const lay = JSON.parse(await evaluate(LAYOUT_JS));
                const eff = await evaluate("document.documentElement.getAttribute('data-theme-effective')");
                report.push(`${label}: height ${lay.height} px, scrollWidth ${lay.scrollWidth}/${lay.clientWidth}`);
                if (lay.scrollWidth > lay.clientWidth) errors.push(`${label}: horizontal overflow ${JSON.stringify(lay.offenders)}`);
                if (eff !== theme) errors.push(`${label}: effective theme is ${eff}, expected ${theme}`);
                if (mobile) {
                    if (lay.small.length) errors.push(`${label}: text under 14 px: ${lay.small.join(' | ')}`);
                    if (lay.taps.length) errors.push(`${label}: tap targets under 44 px: ${lay.taps.join(' | ')}`);
                    if (lay.demo && lay.demo.stripWidth > width) errors.push(`${label}: the film strip is wider than the page`);
                }

                const height = Math.min(lay.height, 16000);
                const shot = await cdp.send('Page.captureScreenshot', {
                    format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width, height, scale: 1 },
                }, s);
                const file = `${PREFIX}${page.replace(/[\\/]/g, '_').replace('.html', '')}-${width}-${theme}.png`;
                writeFileSync(join(OUT, file), Buffer.from(shot.data, 'base64'));
                shots.push(file);

                // Interactions, once per width, after the screenshot.
                if (theme === 'light') {
                    if (mobile && await evaluate("!!document.querySelector('.menu-toggle')")) {
                        await tap('.menu-toggle');
                        const open = await evaluate("getComputedStyle(document.getElementById('site-nav')).display + ' ' + document.querySelector('.menu-toggle').getAttribute('aria-expanded')");
                        if (open !== 'block true') errors.push(`${label}: the menu did not open by touch (${open})`);
                        await tap('.menu-toggle');
                    }
                    if (mobile && await evaluate("!!document.querySelector('label[for=\"__drawer\"].st-drawer-btn')")) {
                        // The docs: the drawer opens by touch and holds the site's links.
                        await tap('label[for="__drawer"].st-drawer-btn');
                        const drawer = await evaluate("document.getElementById('__drawer').checked && document.querySelector('.st-drawer-site').checkVisibility()");
                        if (!drawer) errors.push(`${label}: the docs drawer did not open by touch`);
                        await evaluate("document.getElementById('__drawer').checked = false");
                    }
                    if (mobile && await evaluate("!!document.querySelector('[data-theme-toggle]')")) {
                        await tap('[data-theme-toggle]');
                        const t = await evaluate("document.documentElement.getAttribute('data-theme-effective')");
                        if (t !== 'dark') errors.push(`${label}: the theme toggle did not respond to touch`);
                        await tap('[data-theme-toggle]');
                    }
                    if (page === 'index.html' && ROOT === REPO) {
                        const before = JSON.parse(await evaluate(`JSON.stringify({
                            all: document.querySelectorAll('#pub-list > li').length,
                            shown: document.querySelectorAll('#pub-list > li:not([hidden])').length,
                            news: document.querySelectorAll('#news .news-item').length,
                            newsShown: [...document.querySelectorAll('#news .news-item')].filter((e) => e.checkVisibility()).length,
                        })`));
                        const press = mobile ? tap : async (sel) => { await evaluate(`document.querySelector(${JSON.stringify(sel)}).click()`); };
                        const filterHidden = await evaluate("document.querySelector('[data-filter-for]').hidden");
                        await press('#pub-toggle');
                        const expanded = await evaluate("document.querySelectorAll('#pub-list > li:not([hidden])').length");
                        const filterShown = await evaluate("!document.querySelector('[data-filter-for]').hidden");
                        await press('[data-filter="agents"]');
                        const agents = await evaluate("document.querySelectorAll('#pub-list > li:not([hidden])').length");
                        await press('[data-filter="all"]');
                        await press('#pub-toggle');
                        const collapsed = await evaluate("document.querySelectorAll('#pub-list > li:not([hidden])').length");
                        await press('#news details summary');
                        const newsOpen = await evaluate("[...document.querySelectorAll('#news .news-item')].filter((e) => e.checkVisibility()).length");
                        await press('#news details summary');
                        await press('#details summary');
                        const skillsOpen = await evaluate("document.getElementById('details').open && document.getElementById('skills').checkVisibility()");
                        await press('#details summary');
                        const res = { ...before, filterHidden, expanded, filterShown, agents, collapsed, newsOpen, skillsOpen };
                        report.push('  interactions: ' + JSON.stringify(res));
                        if (!(res.shown === 5 && res.filterHidden && res.expanded === res.all && res.filterShown && res.agents === 4 && res.collapsed === 5)) errors.push(`${label}: publication toggle or filter failed: ${JSON.stringify(res)}`);
                        if (!(res.newsShown === 3 && res.newsOpen === res.news)) errors.push(`${label}: news toggle failed: ${JSON.stringify(res)}`);
                        if (!res.skillsOpen) errors.push(`${label}: details toggle failed`);
                        if (mobile && lay.demo) report.push('  demo: ' + JSON.stringify(lay.demo));
                    }
                }
                for (const b of bucket) errors.push(`${label}: ${b}`);
            }
        }
    }

    // With motion allowed: the film strip lights up once when it scrolls into view, and blocks
    // below the fold are revealed by scrolling.
    if (PAGES.includes('index.html') && ROOT === REPO) {
        bucket = [];
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
        await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }, s);
        await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, s);
        const loaded = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/index.html` }, s);
        await loaded;
        await sleep(300);
        const ready = await evaluate("document.querySelector('[data-strip]').classList.contains('is-ready')");
        const pending = await evaluate("document.querySelectorAll('.reveal.is-pending').length");
        const after = await evaluate(`(async () => {
            for (let y = 0; y < document.documentElement.scrollHeight; y += 400) {
                window.scrollTo({ top: y, behavior: 'instant' });
                await new Promise((r) => setTimeout(r, 60));
            }
            await new Promise((r) => setTimeout(r, 400));
            return document.querySelectorAll('.reveal.is-pending').length;
        })()`);
        await evaluate("document.querySelector('[data-strip]').scrollIntoView({ block: 'center', behavior: 'instant' })");
        await sleep(3500);
        const played = JSON.parse(await evaluate(`(() => {
            const st = document.querySelector('[data-strip]');
            const cells = [...st.querySelectorAll('.strip-cell')].map((c) => getComputedStyle(c).opacity);
            return JSON.stringify({ played: st.classList.contains('is-played'), cells });
        })()`));
        report.push(`motion pass: strip ready ${ready}, ${JSON.stringify(played)}, reveal ${pending} pending then ${after}`);
        if (!ready || !played.played || played.cells.some((o) => o !== '1')) errors.push(`film strip did not play to its final state: ${JSON.stringify(played)}`);
        if (!(pending > 0 && after === 0)) errors.push(`reveal on scroll: ${pending} pending at load, ${after} after scrolling`);
        for (const b of bucket) errors.push(`index.html motion pass: ${b}`);
    }

    await cdp.send('Browser.close').catch(() => {});
    proc.kill();
    server.close();
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* profile still locked */ }

    report.forEach((l) => console.log(l));
    console.log(`Screenshots (${shots.length}): ${OUT}`);
    errors.forEach((e) => console.log('ERROR ' + e));
    console.log(errors.length ? `${errors.length} error(s)` : 'OK');
    process.exit(errors.length ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
