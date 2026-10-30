/* Shared constants and helpers used by charts.js, explorer.js and app.js. */
(function () {
  const VARS = ["Consequentialist", "Emotional", "Relational"];
  const VAR = {
    Base: { key: "base", short: "Base", color: "#5b6478", ink: "#454e60" },
    Consequentialist: { key: "c", short: "C", color: "#00AEB3", ink: "#007a7e" },
    Emotional: { key: "e", short: "E", color: "#F58137", ink: "#b4521a" },
    Relational: { key: "r", short: "R", color: "#ED017D", ink: "#b3005f" },
  };

  const fmt = {
    p: d3.format(".2f"),
    p3: d3.format(".3f"),
    signed: (v) => (v > 0 ? "+" : v < 0 ? "−" : "±") + d3.format(".2f")(Math.abs(v)),
    signed3: (v) => (v > 0 ? "+" : v < 0 ? "−" : "±") + d3.format(".3f")(Math.abs(v)),
    pp: (v) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.round(Math.abs(v) * 100) + " pp",
  };

  /* ---------- tooltip ---------- */
  const tipEl = document.getElementById("tooltip");
  const tip = {
    show(html, evt) {
      tipEl.innerHTML = html;
      tipEl.classList.add("on");
      tip.move(evt);
    },
    move(evt) {
      if (!evt) return;
      const pad = 14;
      const r = tipEl.getBoundingClientRect();
      let x = evt.clientX + pad;
      let y = evt.clientY + pad;
      if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
      if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
      tipEl.style.left = Math.max(8, x) + "px";
      tipEl.style.top = Math.max(8, y) + "px";
    },
    hide() { tipEl.classList.remove("on"); },
  };
  const row = (k, v) => `<div class="tt-r"><span>${k}</span><b>${v}</b></div>`;

  /* ---------- word-level diff: which words of `b` are new relative to `a` ---------- */
  function tokenize(s) { return s.match(/\s+|[^\s]+/g) || []; }
  const norm = (w) => w.toLowerCase().replace(/[.,;:!?"“”']/g, "");
  function diffWords(a, b) {
    const A = tokenize(a).filter((t) => !/^\s+$/.test(t));
    const Bt = tokenize(b);
    const B = Bt.filter((t) => !/^\s+$/.test(t));
    const n = A.length, m = B.length;
    const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--)
      for (let j = m - 1; j >= 0; j--)
        L[i][j] = norm(A[i]) === norm(B[j]) ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const keep = new Array(m).fill(false);
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (norm(A[i]) === norm(B[j])) { keep[j] = true; i++; j++; }
      else if (L[i + 1][j] >= L[i][j + 1]) i++;
      else j++;
    }
    // Rebuild with whitespace, grouping consecutive new words into one run.
    const out = [];
    let wi = 0;
    for (const t of Bt) {
      if (/^\s+$/.test(t)) { out.push({ t, ws: true }); continue; }
      out.push({ t, isNew: !keep[wi] });
      wi++;
    }
    return out;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* Render `b` as HTML with new words wrapped in <mark class=cls>. Small kept words between two
     new runs are absorbed so highlights read as phrases, not confetti. */
  function highlightDiff(a, b, cls) {
    const toks = diffWords(a, b);
    const words = toks.filter((t) => !t.ws);
    for (let k = 1; k < words.length - 1; k++)
      if (!words[k].isNew && words[k - 1].isNew && words[k + 1].isNew && words[k].t.length <= 4) words[k].isNew = true;
    let html = "", open = false;
    toks.forEach((t, idx) => {
      if (t.ws) {
        const next = toks.slice(idx + 1).find((x) => !x.ws);
        if (open && !(next && next.isNew)) { html += "</mark>"; open = false; }
        html += t.t;
        return;
      }
      if (t.isNew && !open) { html += `<mark class="${cls}">`; open = true; }
      if (!t.isNew && open) { html += "</mark>"; open = false; }
      html += esc(t.t);
    });
    if (open) html += "</mark>";
    return html;
  }

  /* Animate a number in an element. */
  function countTo(el, to, format, dur = 700) {
    const from = +el.dataset.val || 0;
    el.dataset.val = to;
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      el.textContent = format(from + (to - from) * e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* Gaussian KDE evaluated on `xs`. */
  function kde(values, xs, bw) {
    const k = (u) => Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
    return xs.map((x) => [x, d3.mean(values, (v) => k((x - v) / bw)) / bw]);
  }

  /* Run fn when element first scrolls into view. */
  function onVisible(el, fn, margin = "0px 0px -12% 0px") {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { io.disconnect(); fn(); } });
    }, { rootMargin: margin });
    io.observe(el);
  }

  /* Re-run render on container width change (debounced). */
  function onResize(el, fn) {
    let w = el.clientWidth, t;
    new ResizeObserver(() => {
      if (Math.abs(el.clientWidth - w) < 2) return;
      w = el.clientWidth;
      clearTimeout(t);
      t = setTimeout(fn, 120);
    }).observe(el);
  }

  /* Logos have very different aspect ratios; scaling height by 1/sqrt(ratio) makes them read as equally large. */
  function logoSize(p, base = 16) {
    const h = base * Math.sqrt(4.5 / p.ratio);
    return { w: h * p.ratio, h };
  }
  const logoImg = (p, base) => { const s = logoSize(p, base); return `<img src="${p.logo}" alt="${p.name}" width="${s.w.toFixed(1)}" height="${s.h.toFixed(1)}">`; };

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const DUR = reduceMotion ? 0 : 750;

  window.CMC = { VARS, VAR, fmt, tip, row, highlightDiff, esc, countTo, kde, onVisible, onResize, logoSize, logoImg, DUR, reduceMotion };
})();
