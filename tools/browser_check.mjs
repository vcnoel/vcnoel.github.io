// Browser check for the redesigned pages, run from the repository root:
//
//     node tools/browser_check.mjs [--chrome PATH] [--out DIR] [--widths 360,390,414,768,1280] [--no-shots]
//
// Needs Node 22 or later (built-in WebSocket) and a local Chrome or Edge. The
// script serves the repository on a local port and opens each page in headless
// Chrome at every width, in light and dark themes. It reports console errors,
// uncaught exceptions, failed requests, images that did not load and horizontal
// overflow. At widths up to 820 px it also checks that no visible text is under
// 14 px and that buttons, toggles and standalone links are at least 44 px, and
// it operates the menu, the toggles and the filter with touch events. One
// full-page screenshot per page, width and theme goes to DIR (default
// tools/screens, which is not tracked), unless --no-shots is given. Chrome runs
// with a fresh profile in the temp directory, which is removed once Chrome has
// exited. Exit status is 1 when any error is found.

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
const PAGES = opt('--pages', 'index.html,software/spectral-trust.html,posts/index.html,posts/site-relaunch.html,posts/template.html').split(',');
const WIDTHS = opt('--widths', '360,390,414,768,1280').split(',').map(Number);
const THEMES = ['light', 'dark'];
const OUT = resolve(opt('--out', join(REPO, 'tools', 'screens')));
// --no-shots runs every check without writing the screenshots.
const SHOTS = !args.includes('--no-shots');
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
        '--disable-extensions', '--disable-breakpad', '--disable-crash-reporter',
        `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    return new Promise((ok, fail) => {
        let buf = '';
        const t = setTimeout(() => { proc.kill(); fail(new Error('Chrome did not start')); }, 30000);
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

// Close Chrome, wait for it to exit, then remove its profile folder from the temp directory.
// On Windows the profile's files stay locked for a moment after the process exits, so the removal
// is retried for up to 15 s. Returns false when the folder is still there.
async function closeChrome(cdp, proc, profile) {
    if (proc) {
        const running = () => proc.exitCode === null && proc.signalCode === null;
        const exited = new Promise((ok) => { if (!running()) ok(); else proc.once('exit', ok); });
        if (cdp) await Promise.race([cdp.send('Browser.close').catch(() => {}), sleep(3000)]);
        await Promise.race([exited, sleep(10000)]);
        if (running()) { proc.kill(); await Promise.race([exited, sleep(5000)]); }
    }
    for (let i = 0; i < 30; i++) {
        try { rmSync(profile, { recursive: true, force: true }); } catch { /* still locked */ }
        if (!existsSync(profile)) return true;
        await sleep(500);
    }
    return false;
}
let teardown = null;

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

// The banner: measured at the top of the page, then after a scroll. Each item's text is put in
// the shown item in turn to test that all of them fit at this width.
const BANNER_JS = `(async () => {
    const bar = document.querySelector('[data-announce]');
    if (!bar) return 'null';
    window.scrollTo({ top: 0, behavior: 'instant' });
    await new Promise((r) => requestAnimationFrame(() => r()));
    const header = document.querySelector('.site-header');
    const r = bar.getBoundingClientRect();
    const reserved = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--bar-h'));
    const items = [...bar.querySelectorAll('.announce-item')];
    const shown = items.filter((i) => getComputedStyle(i).visibility === 'visible');
    const clipped = [];
    const copy = shown[0] && shown[0].querySelector('.announce-copy');
    if (copy) {
        const label = copy.querySelector('.announce-label'), text = copy.querySelector('.announce-text');
        const keep = [label && label.textContent, text.textContent];
        for (const it of window.SITE_ANNOUNCEMENTS || []) {
            if (label) label.textContent = it.label || '';
            text.textContent = it.text;
            if (copy.scrollHeight > copy.clientHeight + 1 || copy.scrollWidth > copy.clientWidth + 1 || copy.getBoundingClientRect().bottom > r.bottom + 0.5)
                clipped.push(it.text.slice(0, 40) + ' (' + copy.scrollWidth + 'x' + copy.scrollHeight + ' in ' + copy.clientWidth + 'x' + copy.clientHeight + ')');
        }
        if (label) label.textContent = keep[0];
        text.textContent = keep[1];
    }
    const headerGap = Math.abs(header.getBoundingClientRect().top - r.bottom);
    window.scrollTo({ top: 900, behavior: 'instant' });
    await new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => res())));
    const h = header.getBoundingClientRect();
    const probe = document.elementFromPoint(Math.round(innerWidth / 2), Math.round(h.bottom + 4));
    // a page shorter than the window plus the banner cannot scroll the banner away (the post template)
    const room = document.documentElement.scrollHeight - innerHeight;
    const afterScroll = { headerTop: Math.round(h.top), barBottom: Math.round(bar.getBoundingClientRect().bottom), coveredByBar: !!(probe && probe.closest('[data-announce]')), short: room < r.height };
    window.scrollTo({ top: 0, behavior: 'instant' });
    return JSON.stringify({ position: getComputedStyle(bar).position, height: r.height, reserved, items: items.length, shown: shown.length,
        pause: !!bar.querySelector('.announce-pause'), clipped, headerGap, afterScroll });
})()`;

async function main() {
    mkdirSync(OUT, { recursive: true });
    const server = await startServer();
    const base = `http://127.0.0.1:${server.address().port}`;
    const profile = mkdtempSync(join(tmpdir(), 'site-check-profile-'));
    let proc = null, cdp = null;
    teardown = async () => {
        teardown = null;
        server.close();
        return closeChrome(cdp, proc, profile);
    };
    const launched = await launchChrome(profile);
    proc = launched.proc;
    cdp = new CDP(launched.ws);
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

                // Announcement bar: in the page flow above the header, its height fixed, every item
                // fitting without clipping, and nothing covered once the page scrolls.
                const bar = JSON.parse(await evaluate(BANNER_JS));
                if (!bar) errors.push(`${label}: no announcement bar`);
                else {
                    if (width === 360 || width === 1280) report.push(`  banner: ${JSON.stringify(bar)}`);
                    if (!['static', 'relative'].includes(bar.position)) errors.push(`${label}: the banner is ${bar.position}, it could cover content`);
                    if (Math.abs(bar.height - bar.reserved) > 0.5) errors.push(`${label}: banner height ${bar.height} differs from the reserved ${bar.reserved}`);
                    if (bar.items !== 1 || bar.pause) errors.push(`${label}: with reduced motion the banner should show one item and no pause button`);
                    if (bar.clipped.length) errors.push(`${label}: banner items clipped: ${bar.clipped.join(' | ')}`);
                    if (bar.headerGap > 0.5) errors.push(`${label}: the header does not sit right under the banner (${bar.headerGap})`);
                    if (bar.afterScroll.coveredByBar || (!bar.afterScroll.short && (bar.afterScroll.headerTop !== 0 || bar.afterScroll.barBottom > 0))) errors.push(`${label}: after scrolling the banner covers content: ${JSON.stringify(bar.afterScroll)}`);
                }

                // Post figures: every video must load from one of its sources (WebM or MP4).
                if (width === 1280 && theme === 'light') {
                    const vids = await evaluate(`Promise.all([...document.querySelectorAll('video')].map((v) => new Promise((r) => {
                        const done = (ok) => r((ok ? 'ok ' : 'failed ') + (v.currentSrc || v.querySelector('source')?.getAttribute('src')) + ' ' + v.videoWidth + 'x' + v.videoHeight);
                        v.preload = 'metadata';
                        v.addEventListener('loadedmetadata', () => done(true), { once: true });
                        setTimeout(() => done(v.readyState >= 1), 10000);
                        v.load();
                    }))).then((x) => JSON.stringify(x))`);
                    for (const line of JSON.parse(vids)) {
                        if (!line.startsWith('ok ')) errors.push(`${label}: video ${line}`);
                    }
                    if (JSON.parse(vids).length) report.push(`  videos: ${JSON.parse(vids).length} loaded`);
                }

                // Card icons and spotlight with reduced motion: no animation, the finished picture,
                // and hovering a card shows the whole gradient border at once, without the glow.
                if (page === 'index.html' && ROOT === REPO && width === 1280 && theme === 'light') {
                    const still = JSON.parse(await evaluate(`(() => {
                        const arts = [...document.querySelectorAll('svg.card-art')];
                        const cs = (sel) => getComputedStyle(document.querySelector(sel));
                        return JSON.stringify({ arts: arts.length,
                            animations: arts.reduce((n, a) => n + a.getAnimations({ subtree: true }).length, 0),
                            armed: arts.filter((a) => a.classList.contains('is-anim')).length,
                            final: [cs('.ca-rsn .rs-halo').opacity, cs('.ca-rsn .rs-n5').fill, cs('.ca-agt .ag-tool').opacity,
                                cs('.ca-agt .ag-shield').fillOpacity, cs('.ca-lgl .lg-l3').opacity, cs('.ca-lgl .lg-s3').fill] });
                    })()`));
                    const card = JSON.parse(await evaluate(`(() => { const c = document.querySelectorAll('#research .card')[1]; c.scrollIntoView({ block: 'center', behavior: 'instant' });
                        const r = c.getBoundingClientRect(); return JSON.stringify({ x: r.left + r.width * 0.3, y: r.top + r.height * 0.4 }); })()`));
                    await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: Math.round(card.x), y: Math.round(card.y) }, s);
                    await sleep(150);
                    const hov = JSON.parse(await evaluate(`(() => { const c = document.querySelectorAll('#research .card')[1];
                        const b = getComputedStyle(c, '::before'), a = getComputedStyle(c, '::after');
                        return JSON.stringify({ border: b.opacity, radial: b.backgroundImage.includes('radial'), glow: a.opacity, mx: c.style.getPropertyValue('--mx') }); })()`));
                    await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 1, y: 1 }, s);
                    report.push(`  card icons, reduced motion: ${JSON.stringify(still)}, hover ${JSON.stringify(hov)}`);
                    const want = ['0.95', 'rgb(230, 159, 0)', '0.32', '0.18', '0', 'rgb(230, 159, 0)'];
                    if (still.arts !== 3 || still.animations !== 0 || still.armed !== 0) errors.push(`${label}: with reduced motion the card icons should not animate: ${JSON.stringify(still)}`);
                    if (JSON.stringify(still.final) !== JSON.stringify(want)) errors.push(`${label}: the card icons do not show their finished picture: ${JSON.stringify(still.final)}`);
                    if (hov.border !== '1' || hov.radial || hov.glow !== '0' || hov.mx) errors.push(`${label}: with reduced motion a hovered card should show the whole border only: ${JSON.stringify(hov)}`);
                }

                if (SHOTS) {
                    const height = Math.min(lay.height, 16000);
                    const shot = await cdp.send('Page.captureScreenshot', {
                        format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width, height, scale: 1 },
                    }, s);
                    const file = `${PREFIX}${page.replace(/[\\/]/g, '_').replace('.html', '')}-${width}-${theme}.png`;
                    writeFileSync(join(OUT, file), Buffer.from(shot.data, 'base64'));
                    shots.push(file);
                }

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
                    if (mobile && page === 'index.html' && ROOT === REPO) {
                        // A tap on a tile shows its whole gradient border.
                        await tap('#research .deploy-tile strong');
                        const lit = await evaluate("getComputedStyle(document.querySelector('#research .deploy-tile'), '::before').opacity");
                        if (lit !== '1') errors.push(`${label}: a tap on a tile did not show its gradient border (opacity ${lit})`);
                    }
                    if (page === 'index.html' && ROOT === REPO) {
                        const before = JSON.parse(await evaluate(`JSON.stringify({
                            all: document.querySelectorAll('#pub-list > li').length,
                            shown: document.querySelectorAll('#pub-list > li:not([hidden])').length,
                            news: document.querySelectorAll('#news .news-item').length,
                            newsTop: document.querySelectorAll('#news .wrap > .news-list > .news-item').length,
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
                        if (!(res.newsShown === res.newsTop && res.newsTop < res.news && res.newsOpen === res.news)) errors.push(`${label}: news toggle failed: ${JSON.stringify(res)}`);
                        if (!res.skillsOpen) errors.push(`${label}: details toggle failed`);
                        if (mobile && lay.demo) report.push('  demo: ' + JSON.stringify(lay.demo));
                    }
                }
                for (const b of bucket) errors.push(`${label}: ${b}`);
            }
        }
    }

    // With motion allowed: the attention graph draws in once it is in view, the film strip lights up
    // once, the hero text settles, and blocks below the fold are revealed by scrolling.
    const motionPages = PAGES.filter((pg) => ['index.html', 'software/spectral-trust.html'].includes(pg));
    for (const pg of ROOT === REPO ? motionPages : []) {
        bucket = [];
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
        await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }, s);
        await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, s);
        const loaded = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/${pg}` }, s);
        await loaded;
        await sleep(300);
        const pending = await evaluate("document.querySelectorAll('.reveal.is-pending').length");
        const after = await evaluate(`(async () => {
            for (let y = 0; y < document.documentElement.scrollHeight; y += 400) {
                window.scrollTo({ top: y, behavior: 'instant' });
                await new Promise((r) => setTimeout(r, 60));
            }
            await new Promise((r) => setTimeout(r, 400));
            return document.querySelectorAll('.reveal.is-pending').length;
        })()`);
        const state = JSON.parse(await evaluate(`(async () => {
            const res = {};
            const g = document.querySelector('svg.st-graph');
            if (g) {
                g.scrollIntoView({ block: 'center', behavior: 'instant' });
                await new Promise((r) => setTimeout(r, 4500));
                const line = g.querySelector('.g-edges line:last-child');
                const node = g.querySelector('.g-nodes circle');
                res.graph = { ready: g.classList.contains('is-ready'), played: g.classList.contains('is-played'),
                    dash: getComputedStyle(line).strokeDashoffset,
                    fill: getComputedStyle(node).fill, target: node.style.getPropertyValue('--c') };
            }
            const st = document.querySelector('[data-strip]');
            if (st) {
                st.scrollIntoView({ block: 'center', behavior: 'instant' });
                await new Promise((r) => setTimeout(r, 3500));
                res.strip = { played: st.classList.contains('is-played') || !st.classList.contains('is-ready'),
                    cells: [...st.querySelectorAll('.strip-cell')].map((c) => getComputedStyle(c).opacity) };
            }
            const hero = document.querySelector('.hero .hero-text > *');
            res.hero = hero ? getComputedStyle(hero).opacity : null;
            return JSON.stringify(res);
        })()`));
        report.push(`motion pass ${pg}: ${JSON.stringify(state)}, reveal ${pending} pending then ${after}`);
        const hex = (c) => c.trim().toLowerCase();
        const rgb = (h) => { const n = parseInt(h.slice(1), 16); return `rgb(${n >> 16}, ${(n >> 8) & 255}, ${n & 255})`; };
        if (state.graph && !(state.graph.ready && state.graph.played && parseFloat(state.graph.dash) === 0 && state.graph.fill === rgb(hex(state.graph.target))))
            errors.push(`${pg}: the attention graph did not draw to its final state: ${JSON.stringify(state.graph)}`);
        if (state.strip && (!state.strip.played || state.strip.cells.some((o) => o !== '1')))
            errors.push(`${pg}: the film strip did not reach its final state: ${JSON.stringify(state.strip)}`);
        if (state.hero !== null && state.hero !== '1') errors.push(`${pg}: the hero text did not settle (opacity ${state.hero})`);
        if (after !== 0) errors.push(`${pg}: reveal on scroll left ${after} blocks hidden`);
        if (pg === 'index.html' && !(pending > 0)) errors.push(`${pg}: no block was waiting to be revealed`);
        for (const b of bucket) errors.push(`${pg} motion pass: ${b}`);
    }

    // With motion allowed: the banner moves to the next item after 5.5 s, holds while the pointer or
    // the keyboard focus is on it and while paused, and the page below it never moves (no layout shift).
    if (ROOT === REPO && PAGES.includes('index.html')) {
        bucket = [];
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
        await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }, s);
        await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, s);
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 640, y: 700 }, s);
        const loaded = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/index.html` }, s);
        await loaded;
        await evaluate(`(() => { window.__cls = 0; new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!e.hadRecentInput) window.__cls += e.value; }).observe({ type: 'layout-shift', buffered: true }); return true; })()`);
        const idx = () => evaluate("[...document.querySelectorAll('.announce-item')].findIndex((i) => i.classList.contains('is-active'))");
        const mainTop = () => evaluate("Math.round(document.getElementById('main').getBoundingClientRect().top + scrollY)");
        const res = { items: await evaluate("document.querySelectorAll('.announce-item').length"), top0: await mainTop(), start: await idx() };
        await sleep(6200);
        res.after6s = await idx();
        res.top1 = await mainTop();
        const bb = JSON.parse(await evaluate("JSON.stringify(document.querySelector('[data-announce]').getBoundingClientRect())"));
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: Math.round(bb.x + bb.width / 3), y: Math.round(bb.y + bb.height / 2) }, s);
        const hoverAt = await idx();
        await sleep(6200);
        res.hoverHeld = (await idx()) === hoverAt;
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 640, y: 700 }, s);
        await sleep(6200);
        res.afterLeave = (await idx()) !== hoverAt;
        // keyboard: Tab from the skip link reaches the banner link, and focus holds the item
        await evaluate("document.activeElement && document.activeElement.blur(), document.querySelector('.skip-link').focus(), true");
        await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 }, s);
        await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 }, s);
        res.tabReaches = await evaluate("!!document.activeElement.closest('.announce-item.is-active')");
        const focusAt = await idx();
        await sleep(6200);
        res.focusHeld = (await idx()) === focusAt;
        await evaluate("document.activeElement.blur(), true");
        await evaluate("document.querySelector('.announce-pause').click(), true");
        const pauseAt = await idx();
        await sleep(6200);
        res.pauseHeld = (await idx()) === pauseAt && (await evaluate("document.querySelector('.announce-pause').getAttribute('aria-pressed')")) === 'true';
        res.top2 = await mainTop();
        res.cls = Number((await evaluate('window.__cls')).toFixed(4));
        // At 360 px, with the pause button taking its place, every item still fits in two lines.
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 360, height: 800, deviceScaleFactor: 1, mobile: true }, s);
        const loaded360 = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/index.html` }, s);
        await loaded360;
        await sleep(600);
        res.clipped360 = JSON.parse(await evaluate(`JSON.stringify([...document.querySelectorAll('.announce-copy')]
            .filter((c) => c.scrollHeight > c.clientHeight + 1 || c.getBoundingClientRect().bottom > document.querySelector('[data-announce]').getBoundingClientRect().bottom + 0.5)
            .map((c) => c.textContent.slice(0, 40)))`));
        res.pause360 = await evaluate("(() => { const r = document.querySelector('.announce-pause').getBoundingClientRect(); return Math.round(r.width) + 'x' + Math.round(r.height); })()");
        if (res.clipped360.length) errors.push(`banner: items clipped at 360 px next to the pause button: ${res.clipped360.join(' | ')}`);
        report.push(`banner motion: ${JSON.stringify(res)}`);
        if (!(res.items > 1 && res.after6s !== res.start)) errors.push(`banner: did not move to the next item: ${JSON.stringify(res)}`);
        if (!res.hoverHeld || !res.afterLeave) errors.push(`banner: hover did not hold the item, or leaving did not resume: ${JSON.stringify(res)}`);
        if (!res.tabReaches || !res.focusHeld) errors.push(`banner: keyboard focus did not reach or hold the item: ${JSON.stringify(res)}`);
        if (!res.pauseHeld) errors.push(`banner: the pause button did not stop the rotation`);
        if (res.top0 !== res.top1 || res.top1 !== res.top2) errors.push(`banner: the page moved while the items changed: ${JSON.stringify(res)}`);
        if (res.cls > 0.01) errors.push(`banner: cumulative layout shift ${res.cls}`);
        for (const b of bucket) errors.push(`banner motion pass: ${b}`);
    }

    // With motion allowed: the attention graph first draws after the page has loaded, holds, fades
    // back and draws again while it stays in view, and starts over when it comes back into view.
    // The film strip follows the same cycle. None of it moves the layout.
    if (ROOT === REPO && PAGES.includes('software/spectral-trust.html')) {
        bucket = [];
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
        await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }, s);
        await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, s);
        const loaded = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/software/spectral-trust.html` }, s);
        await loaded;
        const cyc = JSON.parse(await evaluate(`(async () => {
            window.__cls = 0;
            new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!e.hadRecentInput) window.__cls += e.value; }).observe({ type: 'layout-shift', buffered: true });
            const g = document.querySelector('svg.st-graph');
            const state = (el) => el.classList.contains('is-rewinding') ? 'rewinding' : el.classList.contains('is-played') ? 'played' : el.classList.contains('is-ready') ? 'ready' : 'none';
            const wait = (ms) => new Promise((r) => setTimeout(r, ms));
            const seq = [state(g)];
            const t0 = performance.now();
            let firstPlay = null;
            while (performance.now() - t0 < 12500) {
                await wait(150);
                const st = state(g);
                if (st !== seq[seq.length - 1]) seq.push(st);
                if (st === 'played' && firstPlay === null) firstPlay = Math.round(performance.now() - t0);
            }
            window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' });
            await wait(400);
            const away = state(g);
            window.scrollTo({ top: 0, behavior: 'instant' });
            await wait(600);
            const back = state(g);
            const strip = document.querySelector('[data-strip]');
            strip.scrollIntoView({ block: 'center', behavior: 'instant' });
            await wait(600);
            const stripIn = state(strip);
            window.scrollTo({ top: 0, behavior: 'instant' });
            await wait(400);
            const stripAway = state(strip);
            strip.scrollIntoView({ block: 'center', behavior: 'instant' });
            await wait(3500);
            const stripBack = state(strip);
            const cells = [...strip.querySelectorAll('.strip-cell')].map((c) => getComputedStyle(c).opacity);
            return JSON.stringify({ seq, firstPlay, away, back, stripIn, stripAway, stripBack, cells, cls: Number(window.__cls.toFixed(4)) });
        })()`));
        report.push(`figure cycle: ${JSON.stringify(cyc)}`);
        const seqStr = cyc.seq.join(' ');
        if (!seqStr.startsWith('ready played rewinding played')) errors.push(`figure cycle: the graph did not draw, rewind and draw again: ${seqStr}`);
        if (!(cyc.firstPlay >= 250)) errors.push(`figure cycle: the graph played before the page had loaded and painted (${cyc.firstPlay} ms)`);
        if (cyc.away !== 'ready' || cyc.back !== 'played') errors.push(`figure cycle: the graph did not start over on coming back into view (${cyc.away}, ${cyc.back})`);
        if (cyc.stripIn !== 'played' || cyc.stripAway !== 'ready' || cyc.stripBack !== 'played' || cyc.cells.some((o) => o !== '1')) errors.push(`figure cycle: the film strip did not replay: ${JSON.stringify(cyc)}`);
        if (cyc.cls > 0.01) errors.push(`figure cycle: cumulative layout shift ${cyc.cls}`);
        for (const b of bucket) errors.push(`figure cycle pass: ${b}`);
    }

    // With motion allowed: the card icons loop while in view and pause out of view, hovering a card
    // replays its icon only while it holds its finished picture and at most once every 2.5 s, the
    // spotlight follows a mouse, keyboard focus shows the whole border, and nothing moves the layout.
    if (ROOT === REPO && PAGES.includes('index.html')) {
        bucket = [];
        await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
        await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }, s);
        await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, s);
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 1, y: 1 }, s);
        const loaded = cdp.waitFor('Page.loadEventFired', s);
        await cdp.send('Page.navigate', { url: `${base}/index.html` }, s);
        await loaded;
        await sleep(400);
        // layout shifts from here on only: the icons, the replays and the spotlight must cause none
        await evaluate(`(() => { window.__cls = 0; new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!e.hadRecentInput) window.__cls += e.value; }).observe({ type: 'layout-shift' }); return true; })()`);
        const run = JSON.parse(await evaluate(`(async () => {
            const wait = (ms) => new Promise((r) => setTimeout(r, ms));
            const arts = [...document.querySelectorAll('svg.card-art')];
            const time = () => arts.map((a) => { const an = a.getAnimations({ subtree: true }); return an.length ? Math.round(Number(an[0].currentTime)) : null; });
            const res = { armed: arts.filter((a) => a.classList.contains('is-anim')).length, animations: arts.map((a) => a.getAnimations({ subtree: true }).length) };
            document.querySelector('#research .grid-3').scrollIntoView({ block: 'center', behavior: 'instant' });
            await wait(400);
            res.live = arts.map((a) => a.classList.contains('is-live'));
            const t0 = time(); await wait(1000); const t1 = time();
            res.advanced = t0.map((v, i) => t1[i] - v);
            window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' });
            await wait(400);
            res.liveAway = arts.map((a) => a.classList.contains('is-live'));
            const t2 = time(); await wait(800); const t3 = time();
            res.advancedAway = t2.map((v, i) => t3[i] - v);
            document.querySelector('#research .grid-3').scrollIntoView({ block: 'center', behavior: 'instant' });
            await wait(400);
            res.liveBack = arts.map((a) => a.classList.contains('is-live'));
            return JSON.stringify(res);
        })()`));
        // Hover replays: the pointer enters the first card with its icon set to a given moment of the loop.
        const firstCard = JSON.parse(await evaluate(`(() => { const r = document.querySelector('#research .card').getBoundingClientRect(); return JSON.stringify({ x: r.left + r.width / 2, y: r.top + r.height / 2 }); })()`));
        const enterAt = async (ms) => {
            await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 1, y: 1 }, s);
            await evaluate(`(() => { for (const a of document.querySelector('svg.card-art').getAnimations({ subtree: true })) a.currentTime = ${ms}; return true; })()`);
            await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: Math.round(firstCard.x), y: Math.round(firstCard.y) }, s);
            await sleep(50);
            return evaluate(`(() => { const an = document.querySelector('svg.card-art').getAnimations({ subtree: true }); return Math.round(Number(an[0].currentTime)) % 6000; })()`);
        };
        const replay = { hold: await enterAt(4000), tooSoon: await enterAt(4200) };
        await sleep(2600);
        replay.duringBuild = await enterAt(1000);
        replay.holdAgain = await enterAt(4500);
        // Spotlight: the pointer at 30 % and 40 % of the second card.
        const sp = JSON.parse(await evaluate(`(() => { const r = document.querySelectorAll('#research .card')[1].getBoundingClientRect(); return JSON.stringify({ x: r.left, y: r.top, w: r.width, h: r.height }); })()`));
        const px = Math.round(sp.x + sp.w * 0.3), py = Math.round(sp.y + sp.h * 0.4);
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: px, y: py }, s);
        await sleep(500);
        const spot = JSON.parse(await evaluate(`(() => { const c = document.querySelectorAll('#research .card')[1];
            const b = getComputedStyle(c, '::before'), a = getComputedStyle(c, '::after');
            return JSON.stringify({ mx: c.style.getPropertyValue('--mx'), my: c.style.getPropertyValue('--my'), border: b.opacity, radial: b.backgroundImage.includes('radial'), glow: a.opacity }); })()`));
        spot.want = `${px - Math.round(sp.x)}px ${py - Math.round(sp.y)}px`;
        await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 1, y: 1 }, s);
        await sleep(500);
        spot.after = await evaluate("getComputedStyle(document.querySelectorAll('#research .card')[1], '::before').opacity");
        // Keyboard: a Tab press, then focus on a link of the third card, shows its whole border.
        await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 }, s);
        await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 }, s);
        await evaluate("document.querySelectorAll('#research .card')[2].querySelector('a').focus(), true");
        await sleep(500);
        const focus = JSON.parse(await evaluate(`(() => { const c = document.querySelectorAll('#research .card')[2], b = getComputedStyle(c, '::before');
            return JSON.stringify({ visible: !!c.querySelector(':focus-visible'), border: b.opacity, radial: b.backgroundImage.includes('radial') }); })()`));
        await evaluate('document.activeElement.blur(), true');
        const cls = Number((await evaluate('window.__cls')).toFixed(4));
        report.push(`card icons and spotlight: ${JSON.stringify({ ...run, replay, spot, focus, cls })}`);
        if (run.armed !== 3 || run.animations.some((n) => n < 10)) errors.push(`card icons: not armed with motion allowed: ${JSON.stringify(run)}`);
        if (run.live.some((v) => !v) || run.advanced.some((d) => d < 800)) errors.push(`card icons: not running while in view: ${JSON.stringify(run)}`);
        if (run.liveAway.some((v) => v) || run.advancedAway.some((d) => d !== 0)) errors.push(`card icons: still running out of view: ${JSON.stringify(run)}`);
        if (run.liveBack.some((v) => !v)) errors.push(`card icons: did not resume on coming back into view: ${JSON.stringify(run)}`);
        if (!(replay.hold < 300 && replay.tooSoon >= 4200 && replay.duringBuild >= 1000 && replay.duringBuild < 1400 && replay.holdAgain < 300))
            errors.push(`card icons: hover replay should happen only while the icon holds, at most every 2.5 s: ${JSON.stringify(replay)}`);
        const off = (a, b) => Math.abs(parseFloat(a) - parseFloat(b));
        const [wx, wy] = spot.want.split(' ');
        if (!(off(spot.mx, wx) <= 1 && off(spot.my, wy) <= 1) || spot.border !== '1' || !spot.radial || spot.glow !== '1' || spot.after !== '0')
            errors.push(`spotlight: did not follow the pointer or did not fade out: ${JSON.stringify(spot)}`);
        if (!focus.visible || focus.border !== '1' || focus.radial) errors.push(`spotlight: keyboard focus did not show the whole border: ${JSON.stringify(focus)}`);
        if (cls > 0.001) errors.push(`card icons and spotlight: cumulative layout shift ${cls}`);
        for (const b of bucket) errors.push(`card icons pass: ${b}`);
    }

    if (!(await teardown())) errors.push(`the Chrome profile could not be removed: ${profile}`);

    report.forEach((l) => console.log(l));
    console.log(`Screenshots (${shots.length}): ${OUT}`);
    errors.forEach((e) => console.log('ERROR ' + e));
    console.log(errors.length ? `${errors.length} error(s)` : 'OK');
    process.exit(errors.length ? 1 : 0);
}

process.on('SIGINT', async () => {
    if (teardown) await teardown().catch(() => {});
    process.exit(130);
});

main().catch(async (e) => {
    console.error(e);
    if (teardown) await teardown().catch(() => {});
    process.exit(2);
});
