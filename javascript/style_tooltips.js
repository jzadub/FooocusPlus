// Style tooltips: hovering a style card shows what the style adds to the
// prompt and negative prompt. Data comes from window.fooocusStyleInfo,
// injected by modules/ui_gradio_extensions.py.

(function () {
    let tip = null;

    function addCss() {
        if (document.getElementById('style-tooltip-css')) return;
        const css = document.createElement('style');
        css.id = 'style-tooltip-css';
        css.textContent = `
#style-tooltip {
    position: fixed; z-index: 10000; max-width: 420px; pointer-events: none;
    padding: 10px 12px; border-radius: 8px; font-size: 13px; line-height: 1.45;
    background: var(--background-fill-primary, #1f2937);
    color: var(--body-text-color, #e5e7eb);
    border: 1px solid var(--border-color-primary, #4b5563);
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    display: none; white-space: normal; word-wrap: break-word;
}
#style-tooltip .st-name { font-weight: 600; font-size: 14px; margin-bottom: 6px; }
#style-tooltip .st-label { font-weight: 600; opacity: 0.75; margin-top: 6px; }
#style-tooltip .st-you { color: var(--color-accent, #f97316); font-weight: 600; }
#style-tooltip .st-none { opacity: 0.6; font-style: italic; }
`;
        document.head.appendChild(css);
    }

    function ensureTip() {
        if (!tip) {
            addCss();
            tip = document.createElement('div');
            tip.id = 'style-tooltip';
            document.body.appendChild(tip);
        }
        return tip;
    }

    function esc(s) {
        return String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    }

    function styleNameOf(label) {
        const input = label.querySelector('input[type="checkbox"]');
        const span = label.querySelector('span');
        if (!input || !span) return null;
        // Same lookup as the style thumbnails: the checkbox value is the
        // untranslated style name; fall back to the visible text.
        let name = input.value && input.value !== 'on' ? input.value : span.textContent.trim();
        if ((!input.value || input.value === 'on') && typeof localization !== 'undefined') {
            name = Object.keys(localization).find(key => localization[key] === name) || name;
        }
        return name;
    }

    function buildHtml(name) {
        const data = window.fooocusStyleInfo || {};
        const info = data[name];
        if (!info) return null;
        let html = `<div class="st-name">${esc(name)}</div>`;
        if (info.d) {
            return html + `<div>${esc(info.d)}</div>`;
        }
        const p = (info.p || '').trim();
        const n = (info.n || '').trim();
        html += '<div class="st-label">Adds to prompt</div>';
        if (!p) {
            html += '<div class="st-none">nothing</div>';
        } else if (p.includes('{prompt}')) {
            html += '<div>' + p.split('{prompt}').map(esc).join('<span class="st-you">[your prompt]</span>') + '</div>';
        } else {
            html += `<div>${esc(p)}</div><div class="st-none">(added after your prompt)</div>`;
        }
        html += '<div class="st-label">Adds to negative</div>';
        html += n ? `<div>${esc(n)}</div>` : '<div class="st-none">nothing</div>';
        return html;
    }

    function place(e) {
        const t = ensureTip();
        const pad = 16;
        let x = e.clientX + pad, y = e.clientY + pad;
        const r = t.getBoundingClientRect();
        if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
        if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
        t.style.left = Math.max(8, x) + 'px';
        t.style.top = Math.max(8, y) + 'px';
    }

    function show(label, e) {
        const name = styleNameOf(label);
        const html = name ? buildHtml(name) : null;
        const t = ensureTip();
        if (!html) { t.style.display = 'none'; return; }
        t.innerHTML = html;
        t.style.display = 'block';
        place(e);
    }

    function hide() {
        if (tip) tip.style.display = 'none';
    }

    function bind() {
        gradioApp().querySelectorAll('.style_selections label').forEach(label => {
            if (label.dataset.styleTipBound) return;
            label.dataset.styleTipBound = '1';
            // name is looked up on each hover, so reused/re-sorted labels stay correct
            label.addEventListener('mouseenter', e => show(label, e));
            label.addEventListener('mousemove', e => { if (tip && tip.style.display === 'block') place(e); });
            label.addEventListener('mouseleave', hide);
        });
    }

    onUiLoaded(bind);
    onAfterUiUpdate(bind);
    window.addEventListener('scroll', hide, true);
})();
