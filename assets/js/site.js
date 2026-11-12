/* Shared script for vcnoel.github.io: theme toggle, mobile menu, section
   highlighting, reveal on scroll, publication filter, copy button, posts list.
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
            var wide = window.matchMedia('(min-width: 1180px)');
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

        function apply(tag, announce) {
            var shown = 0;
            items.forEach(function (i) {
                var match = tag === 'all' || tagsOf(i).indexOf(tag) !== -1;
                i.hidden = !match;
                if (match) shown += 1;
            });
            buttons.forEach(function (b) {
                b.setAttribute('aria-pressed', String(b.getAttribute('data-filter') === tag));
            });
            if (status && announce) {
                var label = '';
                buttons.forEach(function (b) {
                    if (b.getAttribute('data-filter') === tag) label = b.getAttribute('data-label') || '';
                });
                status.textContent = tag === 'all'
                    ? 'Showing all ' + items.length + ' publications.'
                    : 'Showing ' + shown + ' of ' + items.length + ' publications tagged ' + label + '.';
            }
        }

        buttons.forEach(function (b) {
            b.addEventListener('click', function () { apply(b.getAttribute('data-filter'), true); });
        });
        apply('all', false);
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
        initTheme();
        initMenu();
        initSectionHighlight();
        initReveal();
        initPubFilter();
        initCopy();
        initPosts();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
