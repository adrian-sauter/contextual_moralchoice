/* Model explorer: pick a dilemma, one or more variations and several models; compare base vs. variation MAL and the CPS. */
(function () {
  const { VARS, VAR, fmt, tip, row, highlightDiff, esc, logoImg, onResize, DUR } = window.CMC;

  const VKEY = { Base: "B", Consequentialist: "C", Emotional: "E", Relational: "R" };
  const VIDX = { Consequentialist: 1, Emotional: 2, Relational: 3 };
  const VDESC = { Consequentialist: "consequence made explicit", Emotional: "vivid affective framing", Relational: "stranger → close ingroup" };
  const DEFAULT_MODELS = ["meta_llama-3.1-70b-instruct", "qwen_qwen3-8b", "deepseek-ai_DeepSeek-V3.1", "claude_claude-sonnet-4-5-20250929", "openai_gpt-4.1"];

  function init(site) {
    const $ = (id) => document.getElementById(id);
    const models = site.models;
    const featured = site.explorer.scenarios;
    const titles = new Map(featured.map((f) => [f.id, f.title]));
    const state = { scen: "G_113", vars: new Set(["Consequentialist"]), models: new Set(DEFAULT_MODELS) };
    let data = null, scenBy = null;

    /* ---------------- collapse / expand ---------------- */
    const shell = $("ex-shell");
    $("ex-toggle").addEventListener("click", () => {
      // content may only be clipped while collapsing/collapsed; otherwise the sticky control bar would not stick
      shell.classList.add("animating");
      clearTimeout(shell._t); shell._t = setTimeout(() => shell.classList.remove("animating"), 520);
      const closed = shell.classList.toggle("closed");
      $("ex-toggle").setAttribute("aria-expanded", !closed);
      shell.querySelector(".ex-toggle-state .lbl").textContent = closed ? "Show" : "Hide";
    });

    /* ---------------- featured dilemmas ---------------- */
    $("scen-picker").innerHTML = featured.map((s) => `<button class="sp-btn" data-id="${s.id}" title="${esc(s.rule)}">${esc(s.title)}</button>`).join("");
    $("scen-picker").addEventListener("click", (e) => {
      const b = e.target.closest(".sp-btn"); if (!b) return;
      setScenario(b.dataset.id);
    });

    /* ---------------- variations (multi-select, at least one) ---------------- */
    $("var-picker").innerHTML = VARS.map((v) =>
      `<button class="vp-btn" data-v="${v}" aria-pressed="false"><span class="vt"><span class="ck" aria-hidden="true"></span>${v}</span><span class="vd">${VDESC[v]}</span></button>`).join("");
    $("var-picker").addEventListener("click", (e) => {
      const b = e.target.closest(".vp-btn"); if (!b || b.disabled) return;
      const v = b.dataset.v;
      if (state.vars.has(v)) { if (state.vars.size > 1) state.vars.delete(v); } else state.vars.add(v);
      update("vars");
    });

    /* ---------------- models ---------------- */
    $("model-picker").innerHTML = site.providers.map((p) => `
      <div class="mp-prov">
        <div class="mp-logo">${logoImg(p, 15)}</div>
        <div class="mp-models">${models.filter((m) => m.provider === p.id).map((m) =>
          `<button class="mp-btn" data-id="${m.id}" style="--mc:${m.color}" aria-pressed="false">${m.name}</button>`).join("")}</div>
      </div>`).join("");
    $("model-picker").addEventListener("click", (e) => {
      const b = e.target.closest(".mp-btn"); if (!b) return;
      state.models.has(b.dataset.id) ? state.models.delete(b.dataset.id) : state.models.add(b.dataset.id);
      update(false);
    });
    $("mp-all").addEventListener("click", () => { models.forEach((m) => state.models.add(m.id)); update(false); });
    $("mp-none").addEventListener("click", () => { state.models.clear(); update(false); });

    const available = (s) => VARS.filter((v) => s.ctx[VKEY[v]]);
    function setScenario(id) {
      state.scen = id;
      const av = available(scenBy.get(id));
      [...state.vars].forEach((v) => { if (!av.includes(v)) state.vars.delete(v); });
      if (!state.vars.size) state.vars.add(av[0]);
      update(true);
    }

    /* ---------------- scenario combobox ---------------- */
    const combo = $("scen-combo"), panel = $("combo-panel"), list = $("combo-list"), search = $("combo-search"), ruleSel = $("combo-rule");
    let activeIdx = -1;
    function openCombo(open) {
      panel.hidden = !open;
      combo.classList.toggle("open", open);
      $("combo-btn").setAttribute("aria-expanded", open);
      if (open) { renderList(); search.focus(); }
    }
    $("combo-btn").addEventListener("click", () => openCombo(panel.hidden));
    document.addEventListener("click", (e) => { if (!combo.contains(e.target)) openCombo(false); });
    search.addEventListener("input", () => { activeIdx = -1; renderList(); });
    ruleSel.addEventListener("change", () => { activeIdx = -1; renderList(); });
    search.addEventListener("keydown", (e) => {
      const items = [...list.querySelectorAll(".combo-item")];
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        activeIdx = Math.max(0, Math.min(items.length - 1, activeIdx + (e.key === "ArrowDown" ? 1 : -1)));
        items.forEach((it, i) => it.classList.toggle("active", i === activeIdx));
        items[activeIdx]?.scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && items[activeIdx]) {
        e.preventDefault(); items[activeIdx].click();
      } else if (e.key === "Escape") openCombo(false);
    });
    list.addEventListener("click", (e) => {
      const it = e.target.closest(".combo-item"); if (!it) return;
      setScenario(it.dataset.id); openCombo(false);
    });
    function renderList() {
      const q = search.value.trim().toLowerCase();
      const rule = ruleSel.value;
      const hits = data.scenarios.filter((s) => (!rule || s.rule === rule) &&
        (!q || [s.a1, s.a2, s.ctx.B, s.rule, titles.get(s.id) || ""].join(" ").toLowerCase().includes(q)));
      const groups = d3.group(hits, (s) => s.rule);
      list.innerHTML = [...groups].map(([r, ss]) => `<div class="combo-group">${esc(r)}</div>` + ss.map((s) => `
        <button type="button" class="combo-item${s.id === state.scen ? " sel" : ""}" data-id="${s.id}" role="option" aria-selected="${s.id === state.scen}">
          <span>${titles.has(s.id) ? `<span class="ti">${esc(titles.get(s.id))}</span>` : ""}${esc(s.a2)}</span>
          <span class="vs" title="Available variations">${VARS.map((v) => `<i class="${s.ctx[VKEY[v]] ? VAR[v].key : ""}"></i>`).join("")}</span>
        </button>`).join("")).join("") || '<p class="combo-count">No scenarios match.</p>';
      $("combo-count").textContent = `${hits.length} of ${data.scenarios.length} scenarios · dots show the available variations (C, E, R)`;
    }

    /* ---------------- dumbbell chart ---------------- */
    const chartEl = $("ex-dumbbell");
    let svg, gRows, gAxis, W;
    function setupChart() {
      chartEl.innerHTML = "";
      W = chartEl.clientWidth;
      svg = d3.select(chartEl).append("svg").attr("width", W).attr("height", 0);
      gAxis = svg.append("g");
      gRows = svg.append("g");
    }

    function drawChart(animate) {
      const s = scenBy.get(state.scen);
      const vs = VARS.filter((v) => state.vars.has(v));
      const narrow = W < 560;
      const cpsW = narrow ? 50 : 62;
      const m = { t: 8, r: 8 + cpsW * vs.length, b: 44, l: narrow ? 112 : 160 };
      const rowH = vs.length > 1 ? 34 : 30;

      const vals = models.map((md) => data.mal[md.id][s.id]).filter(Boolean);
      const mean = (i) => d3.mean(vals, (a) => a[i]);
      const rows = [{ id: "__avg", name: "Average of 22 LLMs", mal: [0, 1, 2, 3].map(mean) }].concat(
        models.filter((md) => state.models.has(md.id) && data.mal[md.id][s.id])
          .map((md) => ({ id: md.id, name: md.name, color: md.color, mal: data.mal[md.id][s.id] })));
      const plotBottom = m.t + rows.length * rowH;
      // set the height directly (a second transition on the same node would cancel a height transition)
      svg.attr("height", plotBottom + m.b);
      const x = d3.scaleLinear([0, 1], [m.l, W - m.r]);

      gAxis.selectAll("*").remove();
      gAxis.append("g").attr("class", "grid").selectAll("line").data([0, 0.25, 0.75, 1]).join("line")
        .attr("x1", x).attr("x2", x).attr("y1", m.t).attr("y2", plotBottom);
      gAxis.append("line").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", m.t).attr("y2", plotBottom)
        .attr("stroke", "#141a26").attr("stroke-opacity", .5).attr("stroke-dasharray", "5 4");
      gAxis.append("g").attr("class", "axis").attr("transform", `translate(0,${plotBottom + 4})`).call(d3.axisBottom(x).ticks(4).tickSize(3));
      gAxis.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", plotBottom + 38).attr("text-anchor", "middle")
        .text("MAL, P(rule-violating action)");
      vs.forEach((v, j) => gAxis.append("text").attr("class", "axis-title").attr("x", W - m.r + 8 + cpsW * (j + 1) - 4).attr("y", plotBottom + 20)
        .attr("text-anchor", "end").attr("fill", VAR[v].ink).text(`CPS ${VAR[v].short}`));

      const t = svg.transition().duration(animate ? DUR : 450).ease(d3.easeCubicInOut);
      const sel = gRows.selectAll("g.db-row").data(rows, (d) => d.id);
      const ent = sel.enter().append("g").attr("class", "db-row").style("opacity", 0)
        .attr("transform", (d, i) => `translate(0,${m.t + i * rowH + rowH / 2})`);
      ent.append("rect").attr("class", "row-bg").attr("rx", 4);
      ent.append("text").attr("class", (d) => "db-label" + (d.id === "__avg" ? " avg" : "")).attr("dy", "0.35em").attr("text-anchor", "end");
      ent.append("g").attr("class", "db-lines");
      ent.append("circle").attr("class", "db-b").attr("r", 6.5).attr("fill", "#5b6478").attr("stroke", "#fff").attr("stroke-width", 1.5).attr("cx", (d) => x(d.mal[0]));
      ent.append("g").attr("class", "db-vs");
      ent.append("g").attr("class", "db-cps");
      ent.append("rect").attr("class", "row-hit");

      const all = ent.merge(sel);
      all.select(".row-bg").attr("x", 0).attr("width", W).attr("y", -rowH / 2).attr("height", rowH);
      all.select(".row-hit").attr("x", 0).attr("width", W).attr("y", -rowH / 2).attr("height", rowH)
        .on("mousemove", (e, d) => {
          all.select(".row-bg").classed("on", (o) => o.id === d.id);
          tip.show(`<div class="tt-t">${d.color ? `<i style="background:${d.color}"></i>` : ""}${d.name}</div>` + row("MAL, base", fmt.p(d.mal[0])) +
            vs.map((v) => row(`MAL, ${v.toLowerCase()}`, `${fmt.p(d.mal[VIDX[v]])} (CPS ${fmt.signed(d.mal[VIDX[v]] - d.mal[0])})`)).join(""), e);
        })
        .on("mouseleave", () => { all.select(".row-bg").classed("on", false); tip.hide(); });
      all.select(".db-label").attr("x", m.l - 14).text((d) => d.name);
      all.transition(t).style("opacity", 1).attr("transform", (d, i) => `translate(0,${m.t + i * rowH + rowH / 2})`);
      all.select(".db-b").transition(t).attr("cx", (d) => x(d.mal[0]));

      // one line + dot per selected variation; small vertical offsets keep overlapping dots readable
      const off = (j) => (vs.length > 1 ? (j - (vs.length - 1) / 2) * 7 : 0);
      all.each(function (d) {
        const g = d3.select(this);
        const items = vs.map((v, j) => ({ v, j, b: d.mal[0], val: d.mal[VIDX[v]] }));
        const ln = g.select(".db-lines").selectAll("line").data(items, (o) => o.v);
        ln.enter().append("line").attr("stroke-width", 2.5).attr("stroke-linecap", "round").attr("stroke-opacity", .45)
          .attr("x1", (o) => x(o.b)).attr("x2", (o) => x(o.b)).attr("y1", (o) => off(o.j)).attr("y2", (o) => off(o.j))
          .merge(ln).attr("stroke", (o) => VAR[o.v].color)
          .transition(t).attr("x1", (o) => x(o.b)).attr("x2", (o) => x(o.val)).attr("y1", (o) => off(o.j)).attr("y2", (o) => off(o.j));
        ln.exit().remove();
        const dt = g.select(".db-vs").selectAll("circle").data(items, (o) => o.v);
        dt.enter().append("circle").attr("r", vs.length > 1 ? 6 : 7.5).attr("stroke", "#fff").attr("stroke-width", 1.5)
          .attr("cx", (o) => x(o.b)).attr("cy", (o) => off(o.j))
          .merge(dt).attr("fill", (o) => (d.id === "__avg" ? VAR[o.v].ink : VAR[o.v].color)).attr("r", vs.length > 1 ? 6 : 7.5)
          .transition(t).attr("cx", (o) => x(o.val)).attr("cy", (o) => off(o.j));
        dt.exit().remove();
        const tx = g.select(".db-cps").selectAll("text").data(items, (o) => o.v);
        tx.enter().append("text").attr("class", "db-cps").attr("dy", "0.35em").attr("text-anchor", "end")
          .merge(tx).attr("x", (o) => W - m.r + 8 + cpsW * (o.j + 1) - 4)
          .attr("fill", (o) => (o.val - o.b > 0.005 ? VAR[o.v].ink : o.val - o.b < -0.005 ? "#3d5a99" : "#434b5c"))
          .text((o) => fmt.signed(o.val - o.b));
        tx.exit().remove();
      });
      sel.exit().transition().duration(250).style("opacity", 0).remove();

      $("ex-legend").innerHTML = `<span><i style="background:#5b6478"></i>Base version</span>` +
        vs.map((v) => `<span><i style="background:${VAR[v].color}"></i>${v} variation</span>`).join("");
    }

    /* ---------------- scenario text ---------------- */
    const textHTML = (s, v) => (v === "Base" ? `<h4>Base</h4><p>${esc(s.ctx.B)}</p>`
      : `<h4><i class="dot ${VAR[v].key}"></i>${v}</h4><p>${highlightDiff(s.ctx.B, s.ctx[VKEY[v]], VAR[v].key)}</p>`);

    // Variation toggles only add/remove the affected column (fade in/out); the rest stays in place.
    function updateTexts() {
      const s = scenBy.get(state.scen);
      const box = $("ex-texts");
      const want = ["Base", ...VARS.filter((v) => state.vars.has(v))];
      [...box.children].forEach((el) => {
        if (!want.includes(el.dataset.v) && !el.classList.contains("leaving")) {
          el.classList.add("leaving");
          setTimeout(() => { el.remove(); box.style.setProperty("--n", box.querySelectorAll(".ex-text:not(.leaving)").length); }, 260);
        }
      });
      want.forEach((v, i) => {
        if (box.querySelector(`.ex-text[data-v="${v}"]:not(.leaving)`)) return;
        const el = document.createElement("div");
        el.className = "ex-text entering " + (v === "Base" ? "" : VAR[v].key);
        el.dataset.v = v;
        el.innerHTML = textHTML(s, v);
        const after = want.slice(0, i).reverse().map((w) => box.querySelector(`.ex-text[data-v="${w}"]:not(.leaving)`)).find(Boolean);
        after ? after.after(el) : box.prepend(el);
        requestAnimationFrame(() => requestAnimationFrame(() => el.classList.remove("entering")));
      });
      box.style.setProperty("--n", want.length);
    }

    function fillScenario(animate) {
      const s = scenBy.get(state.scen);
      const vs = VARS.filter((v) => state.vars.has(v));
      const render = () => {
        const box = $("ex-texts");
        box.style.setProperty("--n", vs.length + 1);
        box.innerHTML = ["Base", ...vs].map((v) => `<div class="ex-text ${v === "Base" ? "" : VAR[v].key}" data-v="${v}">${textHTML(s, v)}</div>`).join("");
        $("ex-scen-actions").innerHTML = `<div class="act"><b>Rule-adherent action</b>${esc(s.a1)}</div>` +
          `<div class="act violate"><b>Rule-violating action a*</b>${esc(s.a2)}</div>`;
        $("combo-cur").innerHTML = `<b>${esc(titles.get(s.id) || s.rule)}:</b> ${esc(s.a2)}`;
      };
      const wrap = $("ex-scenario");
      if (!animate) return render();
      wrap.classList.add("swap");
      setTimeout(() => { render(); requestAnimationFrame(() => wrap.classList.remove("swap")); }, 220);
    }

    /* ---------------- update ---------------- */
    function update(what) {
      if (!data) return;
      const s = scenBy.get(state.scen);
      document.querySelectorAll(".sp-btn").forEach((b) => b.classList.toggle("on", b.dataset.id === state.scen));
      document.querySelectorAll(".vp-btn").forEach((b) => {
        const v = b.dataset.v;
        b.disabled = !s.ctx[VKEY[v]];
        b.title = b.disabled ? "This scenario has no " + v.toLowerCase() + " variation" : "";
        b.classList.toggle("on", state.vars.has(v));
        b.setAttribute("aria-pressed", state.vars.has(v));
      });
      document.querySelectorAll(".mp-btn").forEach((b) => {
        const on = state.models.has(b.dataset.id);
        b.classList.toggle("on", on); b.setAttribute("aria-pressed", on);
      });
      if (what === "vars") updateTexts();
      else if (what) fillScenario(true);
      drawChart(true);
    }

    fetch("data/explorer.json").then((r) => r.json()).then((d) => {
      data = d;
      scenBy = new Map(d.scenarios.map((s) => [s.id, s]));
      ruleSel.innerHTML = `<option value="">All rules</option>` + [...new Set(d.scenarios.map((s) => s.rule))].map((r) => `<option>${esc(r)}</option>`).join("");
      setupChart();
      fillScenario(false);
      update(false);
      onResize(chartEl, () => { setupChart(); drawChart(false); });
    });
  }

  window.CMCExplorer = { init };
})();
