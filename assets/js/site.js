/* Shared script for vcnoel.github.io: announcement bar, theme toggle, mobile menu, section
   highlighting, reveal on scroll, publication filter, animated figures, card icons,
   copy button, posts list.
   Every feature checks for its elements, so the file is safe on every page. */
(function () {
    'use strict';

    var root = document.documentElement;
    var darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

    function storedTheme() {
        try { return localStorage.getItem('theme'); } catch (e) { return null; }
    }

    function effectiveTheme() {
        var t = root.getAttribute('data-theme');
        if (t === 'light' || t === 'dark') return t;
        return darkQuery && darkQuery.matches ? 'dark' : 'light';
    }

    /* Theme */
    function initTheme() {
        var buttons = document.querySelectorAll('[data-theme-toggle]');
        var metaColor = document.querySelector('meta[name="theme-color"]:not([media])');

        function sync() {
            var eff = effectiveTheme();
            root.setAttribute('data-theme-effective', eff);
            buttons.forEach(function (b) {
                b.setAttribute('aria-label', eff === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
            });
            if (metaColor) metaColor.setAttribute('content', eff === 'dark' ? '#0f1115' : '#fbfaf7');
        }

        buttons.forEach(function (b) {
            b.addEventListener('click', function () {
                var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
                root.setAttribute('data-theme', next);
                try { localStorage.setItem('theme', next); } catch (e) { /* storage unavailable */ }
                sync();
            });
        });

        if (darkQuery) {
            var onChange = function () { if (!storedTheme()) sync(); };
            if (darkQuery.addEventListener) darkQuery.addEventListener('change', onChange);
            else if (darkQuery.addListener) darkQuery.addListener(onChange);
        }
        sync();
    }

    /* Announcement bar: the items of assets/js/announcements.js, one at a time. A new item every
       5.5 s, held while the pointer or the keyboard focus is on the bar, and stopped by its pause
       button. With reduced motion only the first item shows and nothing moves. */
    var ANNOUNCE_MS = 5500;

    function siteRoot() {
        var s = document.querySelector('script[src$="assets/js/site.js"]');
        return s ? s.src.replace(/assets\/js\/site\.js$/, '') : '';
    }

    function initAnnounce() {
        var bar = document.querySelector('[data-announce]');
        if (!bar) return;
        var items = Array.isArray(window.SITE_ANNOUNCEMENTS) ? window.SITE_ANNOUNCEMENTS : [];
        if (!items.length) { bar.hidden = true; return; }
        var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduce) items = items.slice(0, 1);
        var root = siteRoot();

        var inner = el('div', 'wrap announce-inner');
        var list = el('ul', 'announce-list');
        list.setAttribute('role', 'list');
        var lis = items.map(function (item, i) {
            var li = el('li', 'announce-item' + (i === 0 ? ' is-active' : ''));
            var a = el('a', 'announce-link');
            var external = /^https?:\/\//.test(item.href || '');
            a.href = external ? item.href : root + (item.href || '');
            if (external) { a.target = '_blank'; a.rel = 'noopener'; }
            var copy = el('span', 'announce-copy');
            if (item.label) {
                copy.appendChild(el('span', 'announce-label', item.label));
                copy.appendChild(document.createTextNode(' '));
            }
            copy.appendChild(el('span', 'announce-text', item.text || ''));
            a.appendChild(copy);
            li.appendChild(a);
            list.appendChild(li);
            return li;
        });
        inner.appendChild(list);
        bar.appendChild(inner);
        if (lis.length < 2) return;

        var btn = el('button', 'announce-pause');
        btn.type = 'button';
        btn.setAttribute('aria-label', 'Pause the news');
        btn.setAttribute('aria-pressed', 'false');
        btn.innerHTML = '<svg class="icon i-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5v14M15 5v14"/></svg>'
            + '<svg class="icon i-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15l12-7.5z"/></svg>';
        inner.appendChild(btn);

        var current = 0, timer = null, hover = false, focus = false, paused = false;
        function show(next) {
            var prev = lis[current];
            prev.classList.remove('is-active');
            prev.classList.add('is-leaving');
            setTimeout(function () { prev.classList.remove('is-leaving'); }, 800);
            current = next;
            lis[current].classList.add('is-active');
        }
        function schedule() {
            clearTimeout(timer);
            if (hover || focus || paused || document.hidden) return;
            timer = setTimeout(function () { show((current + 1) % lis.length); schedule(); }, ANNOUNCE_MS);
        }
        bar.addEventListener('mouseenter', function () { hover = true; schedule(); });
        bar.addEventListener('mouseleave', function () { hover = false; schedule(); });
        bar.addEventListener('focusin', function () { focus = true; schedule(); });
        bar.addEventListener('focusout', function (e) {
            if (!bar.contains(e.relatedTarget)) { focus = false; schedule(); }
        });
        btn.addEventListener('click', function () {
            paused = !paused;
            btn.setAttribute('aria-pressed', String(paused));
            schedule();
        });
        document.addEventListener('visibilitychange', schedule);
        schedule();
    }

    /* Mobile menu */
    function initMenu() {
        var btn = document.querySelector('.menu-toggle');
        if (!btn) return;
        var nav = document.getElementById(btn.getAttribute('aria-controls'));
        if (!nav) return;

        function set(open) {
            btn.setAttribute('aria-expanded', String(open));
            btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
            nav.classList.toggle('is-open', open);
        }

        btn.addEventListener('click', function () {
            set(btn.getAttribute('aria-expanded') !== 'true');
        });
        nav.addEventListener('click', function (e) {
            if (e.target.closest('a')) set(false);
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && btn.getAttribute('aria-expanded') === 'true') {
                set(false);
                btn.focus();
            }
        });
        if (window.matchMedia) {
            var wide = window.matchMedia('(min-width: 960px)');
            var onWide = function () { if (wide.matches) set(false); };
            if (wide.addEventListener) wide.addEventListener('change', onWide);
            else if (wide.addListener) wide.addListener(onWide);
        }
    }

    /* Highlight the nav link of the section in view */
    function initSectionHighlight() {
        if (!('IntersectionObserver' in window)) return;
        var links = {};
        document.querySelectorAll('.site-nav a[href^="#"]').forEach(function (a) {
            links[a.getAttribute('href').slice(1)] = a;
        });
        var sections = Object.keys(links).map(function (id) { return document.getElementById(id); })
            .filter(Boolean);
        if (!sections.length) return;
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                Object.keys(links).forEach(function (id) { links[id].removeAttribute('aria-current'); });
                var link = links[entry.target.id];
                if (link) link.setAttribute('aria-current', 'true');
            });
        }, { rootMargin: '-40% 0px -55% 0px' });
        sections.forEach(function (s) { io.observe(s); });
    }

    /* Reveal blocks below the fold as they scroll into view */
    function initReveal() {
        if (!('IntersectionObserver' in window)) return;
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.remove('is-pending');
                    io.unobserve(entry.target);
                }
            });
        }, { rootMargin: '0px 0px -6% 0px' });
        document.querySelectorAll('.reveal').forEach(function (el) {
            if (el.getBoundingClientRect().top < window.innerHeight * 0.94) return;
            el.classList.add('is-pending');
            io.observe(el);
        });
    }

    /* Publication filter */
    function initPubFilter() {
        var bar = document.querySelector('[data-filter-for]');
        if (!bar) return;
        var list = document.getElementById(bar.getAttribute('data-filter-for'));
        var status = document.getElementById(bar.getAttribute('data-filter-status'));
        if (!list) return;
        var items = Array.prototype.slice.call(list.querySelectorAll('[data-tags]'));
        var buttons = Array.prototype.slice.call(bar.querySelectorAll('[data-filter]'));

        function tagsOf(item) { return item.getAttribute('data-tags').split(/\s+/); }

        buttons.forEach(function (b) {
            var tag = b.getAttribute('data-filter');
            var n = tag === 'all' ? items.length : items.filter(function (i) { return tagsOf(i).indexOf(tag) !== -1; }).length;
            var count = b.querySelector('.count');
            if (count) count.textContent = String(n);
            if (n === 0 && tag !== 'all') b.hidden = true;
        });

        /* With no topic selected, only the featured entries show until the toggle expands the list. */
        var toggle = document.getElementById(bar.getAttribute('data-filter-toggle') || '');
        var featured = items.filter(function (i) { return i.hasAttribute('data-featured'); });
        var canCollapse = !!toggle && featured.length > 0 && featured.length < items.length;
        var expanded = !canCollapse;
        var current = 'all';

        function apply(tag, announce) {
            current = tag;
            var shown = 0;
            items.forEach(function (i) {
                var match = tag === 'all' ? (expanded || i.hasAttribute('data-featured')) : tagsOf(i).indexOf(tag) !== -1;
                i.hidden = !match;
                if (match) shown += 1;
            });
            buttons.forEach(function (b) {
                b.setAttribute('aria-pressed', String(b.getAttribute('data-filter') === tag));
            });
            if (canCollapse) {
                bar.hidden = !expanded;
                toggle.setAttribute('aria-expanded', String(expanded));
                var label = toggle.querySelector('.label');
                if (label) label.textContent = expanded ? 'Selected publications only' : 'All ' + items.length + ' publications';
            }
            if (status && announce) {
                var name = '';
                buttons.forEach(function (b) {
                    if (b.getAttribute('data-filter') === tag) name = b.getAttribute('data-label') || '';
                });
                status.textContent = tag === 'all'
                    ? 'Showing ' + shown + ' of ' + items.length + ' publications.'
                    : 'Showing ' + shown + ' of ' + items.length + ' publications tagged ' + name + '.';
            }
        }

        buttons.forEach(function (b) {
            b.addEventListener('click', function () { apply(b.getAttribute('data-filter'), true); });
        });
        if (canCollapse) {
            toggle.hidden = false;
            toggle.addEventListener('click', function () {
                expanded = !expanded;
                apply(expanded ? current : 'all', true);
            });
        }
        /* A link to a hidden entry (from the news list) expands the list first. */
        function revealTarget() {
            var id = window.location.hash.slice(1);
            var target = id ? document.getElementById(id) : null;
            if (target && target.hidden && items.indexOf(target) !== -1) {
                expanded = true;
                apply('all', false);
                target.scrollIntoView();
            }
        }
        window.addEventListener('hashchange', revealTarget);
        apply('all', false);
        revealTarget();
    }

    /* Animated figures (the attention graph and the film strip). Each one waits in its start
       state (is-ready), draws in (is-played) once the page has loaded and painted and the figure is
       in view, holds the finished figure, fades back to the start (is-rewinding) and draws again,
       for as long as it stays in view. It starts over each time it comes back into view. Without
       motion, or without this script, the finished figure is shown and nothing moves. */
    var REPLAY = { holdMs: 7000, rewindMs: 1000 };

    function replayInView(fig, drawMs, threshold) {
        var visible = false, timer = null;
        function play() {
            clearTimeout(timer);
            // two frames, so that the start state is painted before the transition starts
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    if (!visible) return;
                    fig.classList.remove('is-rewinding');
                    fig.classList.add('is-played');
                    timer = setTimeout(rewind, drawMs + REPLAY.holdMs);
                });
            });
        }
        function rewind() {
            if (!visible) return;
            fig.classList.add('is-rewinding');
            fig.classList.remove('is-played');
            timer = setTimeout(play, REPLAY.rewindMs + 100);
        }
        function reset() {
            clearTimeout(timer);
            fig.classList.remove('is-played', 'is-rewinding');
        }
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!visible && entry.intersectionRatio >= threshold) { visible = true; play(); }
                else if (visible && !entry.isIntersecting) { visible = false; reset(); }
            });
        }, { threshold: [0, threshold] });
        io.observe(fig);
    }

    function initFigures() {
        if (!('IntersectionObserver' in window)) return;
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        var figs = [];
        document.querySelectorAll('svg.st-graph').forEach(function (g) { figs.push([g, 2800, 0.3]); });
        document.querySelectorAll('[data-strip]').forEach(function (st) { figs.push([st, 2500, 0.35]); });
        if (!figs.length) return;
        figs.forEach(function (f) { f[0].classList.add('is-ready'); });
        function start() {
            setTimeout(function () { figs.forEach(function (f) { replayInView(f[0], f[1], f[2]); }); }, 300);
        }
        if (document.readyState === 'complete') start();
        else window.addEventListener('load', start);
    }

    /* The card icons (drawn by tools/make_card_icons.py) loop every 6 s while in view and pause
       out of view. Each loop ends on the finished picture, held from 3.6 s. Hovering a card while
       its icon holds plays the icon again, at most once every 2.5 s. With reduced motion, or
       without this script, the finished picture shows and nothing moves. */
    var CARD_ART = { loopMs: 6000, holdFromMs: 3600, replayGapMs: 2500 };

    function initCardArt() {
        var arts = Array.prototype.slice.call(document.querySelectorAll('svg.card-art'));
        if (!arts.length || !('IntersectionObserver' in window)) return;
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) { entry.target.classList.toggle('is-live', entry.isIntersecting); });
        }, { threshold: 0 });
        arts.forEach(function (art) {
            art.classList.add('is-anim');
            io.observe(art);
            var card = art.closest('.card');
            if (!card || !art.getAnimations) return;
            var last = -Infinity;
            card.addEventListener('pointerenter', function (e) {
                if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
                var now = performance.now();
                if (now - last < CARD_ART.replayGapMs) return;
                var anims = art.getAnimations({ subtree: true });
                if (!anims.length) return;
                var t = Number(anims[0].currentTime) || 0;
                var delay = anims[0].effect ? anims[0].effect.getTiming().delay || 0 : 0;
                var phase = t < delay ? CARD_ART.loopMs : (t - delay) % CARD_ART.loopMs;
                if (phase < CARD_ART.holdFromMs) return;
                last = now;
                anims.forEach(function (a) { a.currentTime = delay; });
            });
        });
    }

    /* Looping figures in posts: play only while visible, and never under reduced motion. */
    function initVideos() {
        var videos = Array.prototype.slice.call(document.querySelectorAll('video[data-loop]'));
        if (!videos.length) return;
        var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduce || !('IntersectionObserver' in window)) {
            // the poster is the finished figure; no control bar over its axes
            videos.forEach(function (v) { v.removeAttribute('autoplay'); v.pause(); });
            return;
        }
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                var v = entry.target;
                if (entry.isIntersecting) {
                    var p = v.play();
                    if (p && p.catch) p.catch(function () { v.controls = true; });
                } else {
                    v.pause();
                }
            });
        }, { threshold: 0.2 });
        videos.forEach(function (v) { io.observe(v); });
    }

    /* Copy to clipboard */
    function initCopy() {
        document.querySelectorAll('[data-copy]').forEach(function (btn) {
            var label = btn.querySelector('.copy-label');
            var status = document.getElementById(btn.getAttribute('data-copy-status'));
            var original = label ? label.textContent : '';
            var timer = null;

            function done(ok) {
                if (label) label.textContent = ok ? 'Copied' : 'Select and copy';
                if (status) status.textContent = ok ? 'Command copied to the clipboard.' : 'Copy failed. Select the command to copy it.';
                clearTimeout(timer);
                timer = setTimeout(function () {
                    if (label) label.textContent = original;
                    if (status) status.textContent = '';
                }, 2000);
            }

            function fallback(text) {
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.position = 'absolute';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                var ok = false;
                try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
                document.body.removeChild(ta);
                return ok;
            }

            btn.addEventListener('click', function () {
                var text = btn.getAttribute('data-copy');
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallback(text)); });
                } else {
                    done(fallback(text));
                }
            });
        });
    }

    /* Posts list from posts.json. The static items in the HTML stay if the fetch fails. */
    function formatDate(s) {
        var d = new Date(s + 'T00:00:00Z');
        if (isNaN(d.getTime())) return s;
        return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
    }

    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }

    function renderPost(post, strip, headingTag) {
        var li = el('li', 'post-item');
        var head = el('div', 'post-head');
        var title = el(headingTag, 'post-title');
        var a = el('a', null, post.title);
        var url = post.url || '#';
        if (strip && url.indexOf(strip) === 0) url = url.slice(strip.length);
        a.href = url;
        title.appendChild(a);
        if (post.type === 'repost' && post.source) {
            title.appendChild(document.createTextNode(' '));
            var src = el('span', 'post-source', 'via ' + post.source);
            title.appendChild(src);
        }
        head.appendChild(title);
        if (post.date) {
            var t = el('time', 'post-date', formatDate(post.date));
            t.setAttribute('datetime', post.date);
            head.appendChild(t);
        }
        li.appendChild(head);
        if (post.excerpt) li.appendChild(el('p', 'post-excerpt', post.excerpt));
        if (post.tags && post.tags.length) {
            var ul = el('ul', 'tag-list');
            ul.setAttribute('role', 'list');
            post.tags.forEach(function (tag) { ul.appendChild(el('li', 'tag', tag)); });
            li.appendChild(ul);
        }
        return li;
    }

    function initPosts() {
        document.querySelectorAll('[data-posts]').forEach(function (list) {
            if (!window.fetch) return;
            var limit = parseInt(list.getAttribute('data-limit') || '0', 10);
            var strip = list.getAttribute('data-strip') || '';
            var headingTag = list.getAttribute('data-heading') === 'h2' ? 'h2' : 'h3';
            var filterRow = document.getElementById(list.getAttribute('data-filter') || '');
            fetch(list.getAttribute('data-posts'))
                .then(function (res) { if (!res.ok) throw new Error(String(res.status)); return res.json(); })
                .then(function (posts) {
                    if (!Array.isArray(posts) || !posts.length) return;
                    posts.sort(function (a, b) { return String(b.date || '').localeCompare(String(a.date || '')); });
                    var active = null;

                    function render() {
                        var shown = posts.filter(function (p) { return !active || (p.tags || []).indexOf(active) !== -1; });
                        if (limit > 0) shown = shown.slice(0, limit);
                        while (list.firstChild) list.removeChild(list.firstChild);
                        shown.forEach(function (p) { list.appendChild(renderPost(p, strip, headingTag)); });
                    }

                    if (filterRow) {
                        var tags = [];
                        posts.forEach(function (p) { (p.tags || []).forEach(function (t) { if (tags.indexOf(t) === -1) tags.push(t); }); });
                        if (tags.length > 1) {
                            var make = function (label, tag) {
                                var b = el('button', 'filter-btn', label);
                                b.type = 'button';
                                b.setAttribute('aria-pressed', String(tag === null));
                                b.addEventListener('click', function () {
                                    active = tag;
                                    filterRow.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', 'false'); });
                                    b.setAttribute('aria-pressed', 'true');
                                    render();
                                });
                                return b;
                            };
                            filterRow.appendChild(make('All', null));
                            tags.forEach(function (t) { filterRow.appendChild(make(t, t)); });
                            filterRow.hidden = false;
                        }
                    }
                    render();
                })
                .catch(function () { /* keep the static list */ });
        });
    }

    function init() {
        initAnnounce();
        initTheme();
        initMenu();
        initSectionHighlight();
        initReveal();
        initPubFilter();
        initFigures();
        initCardArt();
        initVideos();
        initCopy();
        initPosts();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
