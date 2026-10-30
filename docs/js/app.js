/* Page bootstrap: navigation, scroll reveals, hero animation, dataset/metric visuals, lazy chart init. */
(function () {
  const { VARS, VAR, fmt, highlightDiff, esc, countTo, onVisible, reduceMotion } = window.CMC;
  const C = window.CMCCharts;
  const $ = (id) => document.getElementById(id);
  const HERO_MS = 7000;

  /* ---------------- nav: solid after the hero, scroll-spy ---------------- */
  const nav = $("nav");
  function onScroll() { nav.classList.toggle("solid", window.scrollY > window.innerHeight - 70); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const links = [...document.querySelectorAll(".nav-links a")];
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((a) => { const s = document.querySelector(a.getAttribute("href")); if (s) spy.observe(s); });

  /* ---------------- reveal on scroll ---------------- */
  const rio = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); rio.unobserve(e.target); } });
  }, { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".reveal").forEach((el) => rio.observe(el));

  /* ---------------- formulas (display) and inline math in the text ---------------- */
  document.querySelectorAll(".formula[data-tex]").forEach((el) => {
    try { katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false, trust: true, strict: false }); }
    catch (err) { el.textContent = el.dataset.tex; }
  });
  document.querySelectorAll(".tex").forEach((el) => {
    try { katex.render(el.textContent, el, { displayMode: false, throwOnError: false, strict: false }); }
    catch (err) { /* keep plain text */ }
  });

  /* ---------------- EMNLP flag: one vector SVG whose edges follow a travelling sine wave, so it stays crisp ---------------- */
  (function flag() {
    const rb = document.querySelector(".ribbon");
    if (!rb) return;
    const label = rb.textContent.trim();
    const W = rb.offsetWidth || 320, S = (rb.offsetHeight || 51) / 3, A = S * 0.09, PAD = A + 2, H = 3 * S + 2 * PAD;
    const k = (2 * Math.PI) / (W / 2.6);
    const colors = ["#ce2939", "#ffffff", "#477050"];
    const ns = "http://www.w3.org/2000/svg";
    rb.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="margin-top:${-PAD}px" aria-hidden="true">
      <defs>
        <linearGradient id="flag-shade" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="${(2 * Math.PI) / k}" y2="0" spreadMethod="repeat">
          <stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset=".25" stop-color="#fff" stop-opacity=".08"/>
          <stop offset=".5" stop-color="#000" stop-opacity="0"/><stop offset=".75" stop-color="#000" stop-opacity=".1"/><stop offset="1" stop-color="#000" stop-opacity="0"/>
        </linearGradient>
        <path id="flag-mid"/>
      </defs>
      ${colors.map((c) => `<path class="fs" fill="${c}"/>`).join("")}
      <path class="fshade" fill="url(#flag-shade)"/>
      <text dominant-baseline="central" text-anchor="middle"><textPath href="#flag-mid" startOffset="50%">${label}</textPath></text>
    </svg>`;
    const svg = rb.querySelector("svg");
    const stripes = svg.querySelectorAll(".fs"), shade = svg.querySelector(".fshade"), mid = svg.querySelector("#flag-mid");
    const grad = svg.querySelector("#flag-shade");
    const xs = d3.range(0, W + 0.1, 4);
    const edge = (y, t, rev) => (rev ? [...xs].reverse() : xs).map((x, i) => `${i ? "L" : (rev ? "L" : "M")}${x.toFixed(1)},${(y + A * Math.sin(k * x - t)).toFixed(2)}`).join("");
    function draw(t) {
      stripes.forEach((p, i) => p.setAttribute("d", edge(PAD + i * S, t, false) + edge(PAD + (i + 1) * S, t, true) + "Z"));
      shade.setAttribute("d", edge(PAD, t, false) + edge(PAD + 3 * S, t, true) + "Z");
      mid.setAttribute("d", edge(PAD + 1.5 * S, t, false));
      grad.setAttribute("gradientTransform", `translate(${(t / k).toFixed(2)},0)`);
    }
    if (reduceMotion) return draw(0);
    let t0 = null;
    const loop = (now) => { if (t0 === null) t0 = now; draw(((now - t0) / 1000) * 0.9); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  })();

  /* ---------------- contact: the address is assembled only on click, so it never appears in the page source ---------------- */
  const CONTACT = [[97, 100, 114, 105, 97, 110, 115, 97, 117, 116, 101, 114, 48, 55, 46, 97, 115], [103, 109, 97, 105, 108, 46, 99, 111, 109]];
  document.querySelectorAll(".js-contact").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    const [u, d] = CONTACT.map((c) => String.fromCharCode(...c));
    window.location.href = "mailto:" + u + "@" + d + "?subject=" + encodeURIComponent("Contextual MoralChoice");
  }));

  /* ---------------- citation copy ---------------- */
  $("copy-bib").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("bibtex").textContent); $("copy-bib").textContent = "Copied"; }
    catch { $("copy-bib").textContent = "Select and copy"; }
    setTimeout(() => ($("copy-bib").textContent = "Copy"), 1800);
  });

  fetch("data/site.json").then((r) => r.json()).then(start).catch((err) => {
    console.error(err);
    document.querySelectorAll(".chart").forEach((el) => {
      el.innerHTML = '<p class="caption">Could not load data. Serve this folder over HTTP (e.g. <code>python3 -m http.server</code>).</p>';
    });
  });

  function start(data) {
    hero(data);
    datasetExample(data);
    metricVisuals();
    results(data);
    window.CMCExplorer.init(data);
  }

  /* ---------------- hero: one dilemma cycling through its variations ---------------- */
  function hero(data) {
    const ex = data.explorer;
    const scen = ex.scenarios.find((s) => s.id === "H_008");
    const avg = ex.avg[scen.id];
    const versions = ["Base", ...VARS];
    const card = $("hero-card");
    card.style.setProperty("--hero-dur", HERO_MS + "ms");
    let idx = 0, timer = null;
    $("hero-actions").innerHTML = `<div>${esc(scen.action1)}</div><div>${esc(scen.action2)}</div>`;
    $("hero-p").textContent = `P(“${scen.action2}”)`;
    $("hero-tabs").innerHTML = versions.map((v) =>
      `<button role="tab" data-v="${v}">${{ Base: "Base", Consequentialist: "Conseq.", Emotional: "Emo.", Relational: "Rel." }[v]}<span class="bar"></span></button>`).join("");

    // Fix the text box to the tallest version so the card never resizes while cycling.
    const textEl = $("hero-text");
    function fixHeight() {
      textEl.style.height = "auto";
      const keep = textEl.innerHTML;
      let h = 0;
      versions.forEach((v) => {
        textEl.innerHTML = v === "Base" ? esc(scen.contexts.Base) : highlightDiff(scen.contexts.Base, scen.contexts[v], VAR[v].key);
        h = Math.max(h, textEl.scrollHeight);
      });
      textEl.innerHTML = keep;
      textEl.style.height = h + "px";
    }
    fixHeight();
    window.addEventListener("resize", () => { clearTimeout(fixHeight.t); fixHeight.t = setTimeout(fixHeight, 150); });

    function show(i) {
      idx = i;
      const v = versions[i];
      document.querySelectorAll("#hero-tabs button").forEach((b, k) => {
        b.classList.remove("on");
        void b.offsetWidth; // restart progress animation
        b.classList.toggle("on", k === i);
        b.setAttribute("aria-selected", k === i);
      });
      const t = $("hero-text");
      t.classList.add("fade");
      setTimeout(() => {
        t.innerHTML = v === "Base" ? esc(scen.contexts.Base) : highlightDiff(scen.contexts.Base, scen.contexts[v], VAR[v].key);
        t.classList.remove("fade");
      }, 300);
      $("hero-base").style.width = avg.Base * 100 + "%";
      $("hero-fill").style.width = avg[v] * 100 + "%";
      $("hero-fill").style.background = v === "Base" ? VAR.Base.color : VAR[v].color;
      countTo($("hero-val"), avg[v], fmt.p, 800);
      $("hero-delta").innerHTML = v === "Base" ? "Base version"
        : `CPS<sup>(<b style="color:${VAR[v].ink}">${VAR[v].short}</b>)</sup> = <b style="color:${VAR[v].ink}">${fmt.signed(avg[v] - avg.Base)}</b>`;
    }
    function schedule() {
      clearInterval(timer);
      if (reduceMotion) return;
      timer = setInterval(() => { if (!card.classList.contains("paused")) show((idx + 1) % versions.length); }, HERO_MS);
    }
    $("hero-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      show(versions.indexOf(b.dataset.v)); schedule();
    });
    card.addEventListener("mouseenter", () => card.classList.add("paused"));
    card.addEventListener("mouseleave", () => card.classList.remove("paused"));
    setTimeout(() => { show(0); schedule(); }, 300);
  }

  /* ---------------- dataset: side-by-side versions with highlighted edits ---------------- */
  function datasetExample(data) {
    const ex = data.explorer;
    const sel = $("ex-select"), grid = $("ex-grid");
    sel.innerHTML = ex.scenarios.map((s) => `<option value="${s.id}">${esc(s.title)} (${esc(s.rule)})</option>`).join("");
    function render(id) {
      const s = ex.scenarios.find((x) => x.id === id);
      grid.innerHTML = ["Base", ...VARS].map((v) => `
        <div class="ex-col ${v === "Base" ? "" : VAR[v].key}">
          <h4>${v === "Base" ? "Base" : `<i class="dot ${VAR[v].key}"></i>${v}`}</h4>
          <p>${v === "Base" ? esc(s.contexts.Base) : highlightDiff(s.contexts.Base, s.contexts[v], VAR[v].key)}</p>
        </div>`).join("");
      $("ex-actions").innerHTML = `<div class="act"><b>Rule-adherent action</b>${esc(s.action1)}</div><div class="act violate"><b>Rule-violating action</b>${esc(s.action2)}</div>`;
    }
    sel.value = "G_113";
    render("G_113");
    sel.addEventListener("change", () => {
      grid.classList.add("swap");
      setTimeout(() => { render(sel.value); requestAnimationFrame(() => grid.classList.remove("swap")); }, 250);
    });
  }

  /* ---------------- metrics: illustrative CPS bars ---------------- */
  function metricVisuals() {
    onVisible($("cps-illu"), () => C.cpsIllustration($("cps-illu")));
  }

  /* ---------------- results ---------------- */
  function results(data) {
    let base, cps;
    onVisible($("chart-base"), () => { base = C.baseViolins($("chart-base"), data); }, "0px 0px -5% 0px");
    onVisible($("chart-cps"), () => { cps = C.cpsForest($("chart-cps"), data); }, "0px 0px -5% 0px");
    onVisible($("chart-scatter"), () => C.obedienceScatter($("chart-scatter"), $("scatter-legend"), data));
    onVisible($("chart-slope"), () => C.humanSlope($("chart-slope"), data));
    C.alignTable($("align-table"), data);

    const steer = C.steering($("chart-steer-curve"), $("chart-steer-dist"), $("alpha"), $("alpha-out"), data);
    onVisible($("chart-steer-curve"), () => steer.renderCurve());
    onVisible($("chart-steer-dist"), () => steer.renderDist());

    document.querySelectorAll(".seg").forEach((seg) => {
      seg.addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        const t = seg.dataset.target;
        if (t === "base" && base) base.sort(b.dataset.sort);
        if (t === "cps" && cps) cps.sort(b.dataset.sort);
        if (t === "cps-var" && cps) cps.variation(b.dataset.var);
      });
    });
  }
})();
