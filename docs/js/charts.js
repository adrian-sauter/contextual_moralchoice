/* Result charts. Each chart is created once with the site data and re-renders on resize. */
(function () {
  const { VARS, VAR, fmt, tip, row, kde, onResize, logoSize, logoImg, DUR } = window.CMC;

  /* Positions along one axis for a list of models: grouped by provider (with gaps) or sorted by value. */
  function modelLayout(models, providers, mode, value, step, groupGap) {
    const pos = new Map();
    const groups = [];
    if (mode === "provider") {
      let cur = 0;
      providers.forEach((p) => {
        const ms = models.filter((m) => m.provider === p.id);
        const p0 = cur;
        ms.forEach((m) => { pos.set(m.id, cur); cur += step; });
        groups.push({ ...p, p0, p1: cur });
        cur += groupGap;
      });
      return { pos, groups, length: cur - groupGap };
    }
    [...models].sort((a, b) => value(a) - value(b)).forEach((m, i) => pos.set(m.id, i * step));
    return { pos, groups: [], length: models.length * step };
  }

  /* ======================================================================
     Base MAL violins (models on x, MAL on y), styled after the paper figure
     ====================================================================== */
  function baseViolins(el, data) {
    const state = { sort: "provider" };
    const stats = new Map(data.models.map((m) => {
      const v = data.baseMAL[m.id].slice().sort(d3.ascending);
      return [m.id, {
        median: d3.quantileSorted(v, 0.5), q1: d3.quantileSorted(v, 0.25), q3: d3.quantileSorted(v, 0.75),
        min: v[0], max: v[v.length - 1], mean: d3.mean(v),
        below: d3.mean(v, (x) => (x < 0.5 ? 1 : 0)), strong: d3.mean(v, (x) => (x < 0.25 ? 1 : 0)),
      }];
    }));
    // KDE cut to the data range, as in the paper (seaborn cut=0)
    const dens = new Map(data.models.map((m) => {
      const st = stats.get(m.id);
      const ys = d3.range(st.min, st.max + 1e-9, Math.max(0.005, (st.max - st.min) / 80));
      if (ys[ys.length - 1] < st.max) ys.push(st.max);
      return [m.id, kde(data.baseMAL[m.id], ys, 0.06)];
    }));
    let first = true, prev = null;

    function render() {
      el.innerHTML = "";
      const W = el.clientWidth;
      const narrow = W < 640;
      const grouped = state.sort === "provider";
      const m = { t: grouped && !narrow ? 46 : 10, r: 6, b: narrow ? 96 : 118, l: 52 };
      const gap = narrow ? 8 : 16;
      const groupsN = grouped ? data.providers.length - 1 : 0;
      const step = (W - m.l - m.r - groupsN * gap) / data.models.length;
      const lay = modelLayout(data.models, data.providers, state.sort, (d) => stats.get(d.id).median, step, gap);
      const ph = narrow ? 210 : 250;
      const H = ph + m.t + m.b;
      const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
      const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
      const y = d3.scaleLinear([0, 1], [ph, 0]);

      g.append("rect").attr("x", 0).attr("width", lay.length).attr("y", y(0.25)).attr("height", y(0) - y(0.25)).attr("fill", "rgba(206,41,57,.07)");
      g.append("line").attr("x1", 0).attr("x2", lay.length).attr("y1", y(0.5)).attr("y2", y(0.5))
        .attr("stroke", "#141a26").attr("stroke-opacity", .7).attr("stroke-width", 1.3).attr("stroke-dasharray", "6 4");
      g.append("g").attr("class", "axis axis-lg").attr("transform", "translate(-6,0)").call(d3.axisLeft(y).tickValues([0, 0.25, 0.5, 0.75, 1]).tickFormat(d3.format(".2f")).tickSize(4));
      svg.append("text").attr("class", "axis-title").attr("transform", `translate(12,${m.t + ph / 2}) rotate(-90)`).attr("text-anchor", "middle")
        .text("MAL, P(rule-violating action)");

      if (grouped) {
        lay.groups.slice(1).forEach((gr) => g.append("line").attr("x1", gr.p0 - gap / 2).attr("x2", gr.p0 - gap / 2).attr("y1", -4).attr("y2", ph + 4)
          .attr("stroke", "#b9b5aa").attr("stroke-width", 1.2));
        if (!narrow) {
          lay.groups.forEach((gr) => {
            const bg = g.append("g").attr("class", "bracket").style("opacity", 0);
            const x0 = gr.p0 + 2, x1 = gr.p1 - 2;
            bg.append("path").attr("d", `M${x0},-6V-12H${x1}V-6`).attr("fill", "none").attr("stroke", "#141a26").attr("stroke-width", 1.4);
            const ls = logoSize(gr, 17);
            bg.append("image").attr("href", gr.logo).attr("x", (x0 + x1) / 2 - ls.w / 2).attr("y", -18 - ls.h - 4).attr("width", ls.w).attr("height", ls.h);
            bg.transition().delay(prev ? DUR : 0).duration(300).style("opacity", 1);
          });
        }
      }

      const hw = Math.min(step * 0.44, 26);
      const cols = g.selectAll("g.vcol").data(data.models, (d) => d.id).join("g").attr("class", "vcol")
        .attr("transform", (d) => `translate(${(prev ? prev.get(d.id) : lay.pos.get(d.id)) + step / 2},0)`);
      cols.transition().duration(prev ? DUR : 0).ease(d3.easeCubicInOut).attr("transform", (d) => `translate(${lay.pos.get(d.id) + step / 2},0)`);
      cols.append("rect").attr("class", "row-bg").attr("x", -step / 2).attr("width", step).attr("y", -4).attr("height", ph + 8).attr("rx", 3);
      const body = cols.append("g").attr("transform", first && DUR ? "scale(0,1)" : null);
      body.transition("grow").duration(DUR).delay((d, i) => (first ? i * 25 : 0)).attr("transform", "scale(1,1)");
      body.append("path")
        .attr("fill", (d) => d.color).attr("stroke", "#141a26").attr("stroke-width", 1.2)
        .attr("d", (d) => {
          const dd = dens.get(d.id);
          const w = d3.scaleLinear([0, d3.max(dd, (p) => p[1])], [0, hw]);
          return d3.area().y((p) => y(p[0])).x0((p) => -w(p[1])).x1((p) => w(p[1])).curve(d3.curveBasis)(dd);
        });
      body.append("rect").attr("x", -2.5).attr("width", 5).attr("y", (d) => y(stats.get(d.id).q3)).attr("height", (d) => y(stats.get(d.id).q1) - y(stats.get(d.id).q3)).attr("fill", "#141a26");
      body.append("circle").attr("cy", (d) => y(stats.get(d.id).median)).attr("r", 3).attr("fill", "#fff").attr("stroke", "#141a26").attr("stroke-width", .6);
      cols.append("text").attr("class", "row-label").attr("transform", `translate(4,${ph + 12}) rotate(-45)`).attr("text-anchor", "end").attr("dy", "0.35em")
        .text((d) => d.name);
      cols.append("rect").attr("class", "row-hit").attr("x", -step / 2).attr("width", step).attr("y", 0).attr("height", ph + 70)
        .on("mousemove", function (e, d) {
          d3.select(this.parentNode).select(".row-bg").classed("on", true);
          const s = stats.get(d.id);
          tip.show(`<div class="tt-t"><i style="background:${d.color}"></i>${d.name}</div>` +
            row("median MAL", fmt.p(s.median)) + row("interquartile range", `${fmt.p(s.q1)} to ${fmt.p(s.q3)}`) +
            row("scenarios with MAL &lt; 0.5", Math.round(s.below * 100) + "%") + row("scenarios with MAL &lt; 0.25", Math.round(s.strong * 100) + "%"), e);
        })
        .on("mouseleave", function () { d3.select(this.parentNode).select(".row-bg").classed("on", false); tip.hide(); });

      first = false;
      prev = lay.pos;
    }

    render();
    onResize(el, () => { prev = null; render(); });
    return { sort(mode) { state.sort = mode; render(); } };
  }

  /* ======================================================================
     CPS forest plot (rotated provider logos, compact rows)
     ====================================================================== */
  function cpsForest(el, data) {
    const state = { sort: "provider", mobileVar: "Consequentialist" };
    const byKey = new Map(data.cps.map((d) => [d.model + "|" + d.variation, d]));
    const avgBy = new Map(data.cpsAvg.map((d) => [d.variation, d]));
    const meanCps = (m) => d3.mean(VARS, (v) => byKey.get(m.id + "|" + v)?.mean ?? 0);
    const lo = d3.min(data.cps, (d) => d.lo), hi = d3.max(data.cps, (d) => d.hi);
    let first = true, prevY = null;

    function render() {
      el.innerHTML = "";
      const W = el.clientWidth;
      const narrow = W < 760;
      const panels = narrow ? [state.mobileVar] : VARS;
      const rowH = 20, gap = 7;
      const logoW = state.sort === "provider" ? 30 : 0, labelW = W < 640 ? 118 : 134;
      const m = { t: 26, r: 10, b: 50, l: logoW + labelW };
      const lay = modelLayout(data.models, data.providers, state.sort, (d) => -meanCps(d), rowH, gap);
      const avgY = lay.length + 10;
      const plotH = avgY + rowH;
      const H = plotH + m.t + m.b;
      const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
      const g = svg.append("g").attr("transform", `translate(0,${m.t})`);
      const pw = (W - m.l - m.r - (panels.length - 1) * 20) / panels.length;
      const x = d3.scaleLinear().domain([Math.min(-0.02, lo), hi]).nice().range([0, pw]);

      // rotated provider logos
      lay.groups.forEach((gr) => {
        const ls = logoSize(gr, 14);
        const k = Math.min(1, (gr.p1 - gr.p0 - 4) / ls.w);
        const lw = ls.w * k, lh = ls.h * k;
        const cy = (gr.p0 + gr.p1) / 2;
        g.append("image").attr("href", gr.logo)
          .attr("x", -lw / 2).attr("y", -lh / 2).attr("width", lw).attr("height", lh)
          .attr("transform", `translate(${logoW / 2},${cy}) rotate(-90)`)
          .style("opacity", 0).transition().delay(prevY ? DUR : 0).duration(300).style("opacity", 1);
        g.append("line").attr("class", "provider-sep").attr("x1", logoW).attr("x2", W - m.r).attr("y1", gr.p1 + gap / 2).attr("y2", gr.p1 + gap / 2);
      });

      const rowsData = [...data.models, { id: "__avg", name: "Average" }];
      const yOf = (d) => (d.id === "__avg" ? avgY : lay.pos.get(d.id));
      const bg = g.selectAll("rect.row-bg").data(rowsData, (d) => d.id).join("rect").attr("class", "row-bg")
        .attr("x", logoW).attr("width", W - logoW).attr("y", (d) => yOf(d)).attr("height", rowH).attr("rx", 3);
      g.selectAll("text.row-label").data(rowsData, (d) => d.id).join("text")
        .attr("class", (d) => "row-label" + (d.id === "__avg" ? " avg" : "")).attr("x", m.l - 10).attr("text-anchor", "end").attr("dy", "0.35em")
        .attr("y", (d) => yOf(d) + rowH / 2).text((d) => d.name);

      panels.forEach((v, pi) => {
        const pg = g.append("g").attr("transform", `translate(${m.l + pi * (pw + 20)},0)`);
        const col = VAR[v].color;
        pg.append("text").attr("class", "panel-title").attr("x", pw / 2).attr("y", -12).attr("text-anchor", "middle").attr("fill", VAR[v].ink).text(v);
        pg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(4)).join("line").attr("x1", x).attr("x2", x).attr("y1", -4).attr("y2", plotH);
        pg.append("line").attr("class", "zero-line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", -4).attr("y2", plotH);
        pg.append("g").attr("class", "axis axis-lg").attr("transform", `translate(0,${plotH + 4})`).call(d3.axisBottom(x).ticks(4).tickSize(3));

        const items = rowsData.map((d) => ({ d, s: d.id === "__avg" ? avgBy.get(v) : byKey.get(d.id + "|" + v) })).filter((o) => o.s);
        const it = pg.selectAll("g.ci").data(items, (o) => o.d.id).join("g")
          .attr("transform", (o) => `translate(0,${(prevY && o.d.id !== "__avg" ? prevY.get(o.d.id) : yOf(o.d)) + rowH / 2})`);
        it.transition().duration(DUR).ease(d3.easeCubicInOut).attr("transform", (o) => `translate(0,${yOf(o.d) + rowH / 2})`);
        const lines = it.append("line").attr("stroke", col).attr("stroke-width", (o) => (o.d.id === "__avg" ? 3 : 2.2)).attr("stroke-linecap", "round")
          .attr("stroke-opacity", .55).attr("x1", x(0)).attr("x2", x(0));
        const dots = it.append("circle").attr("r", (o) => (o.d.id === "__avg" ? 7 : 5.6)).attr("fill", (o) => (o.d.id === "__avg" ? VAR[v].ink : col))
          .attr("stroke", "#fff").attr("stroke-width", 1.5).attr("cx", first ? x(0) : (o) => x(o.s.mean));
        const t = (s) => (first ? s.transition().duration(DUR + 200).delay((o, i) => 150 + i * 15).ease(d3.easeCubicOut) : s);
        t(lines).attr("x1", (o) => x(o.s.lo)).attr("x2", (o) => x(o.s.hi));
        t(dots).attr("cx", (o) => x(o.s.mean));
      });
      svg.append("text").attr("class", "axis-title-lg").attr("x", m.l + (W - m.l - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle")
        .text("Contextual Preference Shift (95% CI)");

      g.selectAll("rect.row-hit").data(rowsData, (d) => d.id).join("rect").attr("class", "row-hit")
        .attr("x", logoW).attr("width", W - logoW).attr("y", (d) => yOf(d)).attr("height", rowH)
        .on("mousemove", (e, d) => {
          bg.classed("on", (b) => b.id === d.id);
          const rowsHtml = VARS.map((v) => {
            const s = d.id === "__avg" ? avgBy.get(v) : byKey.get(d.id + "|" + v);
            return s ? row(`<i style="background:${VAR[v].color}"></i> ${v}`, `${fmt.signed3(s.mean)} <span style="opacity:.6;font-weight:400">[${fmt.p3(s.lo)}, ${fmt.p3(s.hi)}]</span>`) : "";
          }).join("");
          tip.show(`<div class="tt-t">${d.id === "__avg" ? "Average" : `<i style="background:${d.color}"></i>${d.name}`}</div>${rowsHtml}`, e);
        })
        .on("mouseleave", () => { bg.classed("on", false); tip.hide(); });

      prevY = lay.pos;
      first = false;
    }

    render();
    onResize(el, () => { prevY = null; render(); });
    return {
      sort(mode) { state.sort = mode; render(); },
      variation(v) { state.mobileVar = v; prevY = null; render(); },
    };
  }

  /* ======================================================================
     Base MAL vs variation MAL scatter
     ====================================================================== */
  function obedienceScatter(el, legendEl, data) {
    const modelBy = new Map(data.models.map((m) => [m.id, m]));
    const pts = data.scatter.filter((d) => modelBy.has(d.model));
    const all = pts.flatMap((d) => [d.base, d.var]);
    const dom = [Math.floor((d3.min(all) - 0.02) * 20) / 20, Math.ceil((d3.max(all) + 0.02) * 20) / 20];
    let focus = null, pinned = null, first = true, api;

    function regression(ps) {
      const xm = d3.mean(ps, (p) => p.base), ym = d3.mean(ps, (p) => p.var);
      const b = d3.sum(ps, (p) => (p.base - xm) * (p.var - ym)) / d3.sum(ps, (p) => (p.base - xm) ** 2);
      return { b, a: ym - b * xm };
    }

    function render() {
      el.innerHTML = "";
      const W = el.clientWidth;
      const cols = W >= 760 ? 3 : 1;
      const gapX = 24;
      const m = { t: 26, r: 10, b: 40, l: 44 };
      const pw = Math.min(cols === 1 ? 380 : 9999, (W - cols * (m.l + m.r) - (cols - 1) * gapX) / cols);
      const ph = Math.min(pw, 270);
      const H = Math.ceil(3 / cols) * (ph + m.t + m.b);
      const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
      const x = d3.scaleLinear(dom, [0, pw]);
      const y = d3.scaleLinear(dom, [ph, 0]);

      const panels = VARS.map((v, i) => {
        const cx = (i % cols) * (pw + m.l + m.r + gapX) + m.l;
        const cy = Math.floor(i / cols) * (ph + m.t + m.b) + m.t;
        const pg = svg.append("g").attr("transform", `translate(${cx},${cy})`);
        const col = VAR[v].color;
        const ps = pts.filter((p) => p.variation === v);
        pg.append("text").attr("class", "panel-title").attr("x", pw / 2).attr("y", -10).attr("text-anchor", "middle").attr("fill", VAR[v].ink).text(v);
        pg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line").attr("x1", x).attr("x2", x).attr("y1", 0).attr("y2", ph);
        pg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("y1", y).attr("y2", y).attr("x1", 0).attr("x2", pw);
        pg.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`).call(d3.axisBottom(x).ticks(5).tickSize(3));
        pg.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSize(3));
        pg.append("text").attr("class", "axis-title").attr("x", pw / 2).attr("y", ph + 34).attr("text-anchor", "middle").text("Base MAL");
        pg.append("text").attr("class", "axis-title").attr("transform", `translate(${-34},${ph / 2}) rotate(-90)`).attr("text-anchor", "middle").text("Variation MAL");
        pg.append("line").attr("x1", x(dom[0])).attr("y1", y(dom[0])).attr("x2", x(dom[1])).attr("y2", y(dom[1]))
          .attr("stroke", "#141a26").attr("stroke-opacity", .35).attr("stroke-dasharray", "4 4");

        const r = regression(ps);
        const rx = d3.extent(ps, (p) => p.base);
        pg.append("line").attr("stroke", col).attr("stroke-width", 2).attr("stroke-linecap", "round")
          .attr("x1", x(rx[0])).attr("y1", y(r.a + r.b * rx[0])).attr("x2", x(rx[0])).attr("y2", y(r.a + r.b * rx[0]))
          .transition().delay(first ? DUR + 300 : 0).duration(first ? DUR : 0).attr("x2", x(rx[1])).attr("y2", y(r.a + r.b * rx[1]));

        const dots = pg.append("g").selectAll("circle").data(ps, (p) => p.model).join("circle")
          .attr("r", 5.5).attr("fill", (p) => modelBy.get(p.model).color).attr("stroke", "#fff").attr("stroke-width", 1.5)
          .attr("cx", (p) => x(p.base)).attr("cy", (p) => y(first ? p.base : p.var));
        if (first) dots.transition().delay((p, i) => 200 + i * 30).duration(DUR + 200).ease(d3.easeCubicOut).attr("cy", (p) => y(p.var));

        const del = d3.Delaunay.from(ps, (p) => x(p.base), (p) => y(p.var));
        pg.append("rect").attr("width", pw).attr("height", ph).attr("fill", "transparent")
          .on("mousemove", (e) => {
            const [mx, my] = d3.pointer(e);
            const p = ps[del.find(mx, my)];
            if (Math.hypot(x(p.base) - mx, y(p.var) - my) > 28) { setFocus(pinned); tip.hide(); return; }
            setFocus(p.model);
            const md = modelBy.get(p.model);
            tip.show(`<div class="tt-t"><i style="background:${md.color}"></i>${md.name}</div>` +
              row("base MAL", fmt.p(p.base)) + row(`${v} MAL`, fmt.p(p.var)) + row("mean CPS", fmt.signed(p.var - p.base)), e);
          })
          .on("mouseleave", () => { setFocus(pinned); tip.hide(); })
          .on("click", () => { pinned = focus === pinned ? null : focus; setFocus(pinned); });
        return { dots };
      });

      function setFocus(id) {
        focus = id;
        panels.forEach((p) => p.dots.attr("opacity", (d) => (!id || d.model === id ? 1 : 0.18)).attr("r", (d) => (d.model === id ? 7.5 : 5.5))
          .filter((d) => d.model === id).raise());
        legendEl.querySelectorAll(".ml-item").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
      }
      api = { setFocus };
      first = false;
      setFocus(pinned);
    }

    legendEl.innerHTML = data.providers.map((p) => `<div class="ml-group"><div class="lg">${logoImg(p, 15)}</div>` +
      data.models.filter((m) => m.provider === p.id).map((m) => `<button class="ml-item" data-id="${m.id}"><i style="background:${m.color}"></i>${m.name}</button>`).join("") +
      "</div>").join("");
    legendEl.querySelectorAll(".ml-item").forEach((b) => {
      b.addEventListener("mouseenter", () => api.setFocus(b.dataset.id));
      b.addEventListener("mouseleave", () => api.setFocus(pinned));
      b.addEventListener("click", () => { pinned = pinned === b.dataset.id ? null : b.dataset.id; api.setFocus(pinned); });
    });

    render();
    onResize(el, render);
  }

  /* ======================================================================
     Humans vs LLMs: per-variation strip of the 22 model means, LLM average and the human value
     ====================================================================== */
  function humanSlope(el, data) {
    const modelBy = new Map(data.models.map((m) => [m.id, m]));
    const avg = new Map(data.cpsAvg.map((d) => [d.variation, d.mean]));
    const per = d3.group(data.cps.filter((d) => modelBy.has(d.model)), (d) => d.variation);
    const rank = (f) => new Map([...VARS].sort((a, b) => f(b) - f(a)).map((v, i) => [v, i + 1]));
    const rh = rank((v) => data.human.cps[v]), rl = rank((v) => avg.get(v));
    const allX = data.cps.map((d) => d.mean).concat(Object.values(data.human.cps));
    const jit = d3.randomLcg(7);
    const jitter = new Map(data.cps.map((d) => [d.model + d.variation, jit() - 0.5]));

    function render(animate) {
      el.innerHTML = "";
      const W = el.clientWidth;
      const narrow = W < 460;
      const m = { t: 30, r: narrow ? 10 : 96, b: 40, l: narrow ? 74 : 112 };
      const rowH = 66;
      const H = m.t + VARS.length * rowH + m.b;
      const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
      const x = d3.scaleLinear([Math.min(-0.01, d3.min(allX)), d3.max(allX) + 0.01], [m.l, W - m.r]).nice();
      const lg = svg.append("g").attr("transform", `translate(${m.l},10)`);
      const items = [["Humans (N = 132)", "h"], ["LLM average", "a"], ["Individual LLMs", "i"]];
      let lx = 0;
      items.forEach(([t, k]) => {
        const gi = lg.append("g").attr("transform", `translate(${lx},0)`);
        if (k === "h") gi.append("path").attr("d", d3.symbol(d3.symbolStar, 75)()).attr("fill", "#fff").attr("stroke", "#141a26").attr("stroke-width", 1.6);
        if (k === "a") gi.append("circle").attr("r", 6.5).attr("fill", "#5b6478").attr("stroke", "#fff").attr("stroke-width", 1.5);
        if (k === "i") gi.append("circle").attr("r", 3.5).attr("fill", "#8b93a5").attr("fill-opacity", .45).attr("stroke", "#8b93a5");
        const tx = gi.append("text").attr("x", k === "h" ? 13 : 11).attr("dy", "0.35em").attr("class", "row-label").text(t);
        lx += 22 + tx.node().getComputedTextLength() + 14;
      });
      svg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line").attr("x1", x).attr("x2", x).attr("y1", m.t).attr("y2", H - m.b);
      svg.append("line").attr("class", "zero-line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t).attr("y2", H - m.b);
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b + 2})`).call(d3.axisBottom(x).ticks(5).tickSize(3));
      svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text("Mean CPS");

      VARS.forEach((v, i) => {
        const cy = m.t + i * rowH + rowH / 2;
        const col = VAR[v].color;
        const g = svg.append("g");
        if (i > 0) g.append("line").attr("class", "provider-sep").attr("x1", 0).attr("x2", W).attr("y1", cy - rowH / 2).attr("y2", cy - rowH / 2);
        g.append("text").attr("class", "row-label").attr("x", 0).attr("y", cy).attr("dy", "0.35em").attr("fill", VAR[v].ink).style("font-weight", 600)
          .text(narrow ? { Consequentialist: "Conseq.", Emotional: "Emotional", Relational: "Relational" }[v] : v);
        const dots = g.selectAll("circle.ind").data(per.get(v)).join("circle").attr("class", "ind")
          .attr("cx", (d) => x(animate ? 0 : d.mean)).attr("cy", (d) => cy + jitter.get(d.model + d.variation) * 26).attr("r", 4)
          .attr("fill", col).attr("fill-opacity", .35).attr("stroke", col).attr("stroke-opacity", .6)
          .on("mousemove", (e, d) => tip.show(`<div class="tt-t"><i style="background:${modelBy.get(d.model).color}"></i>${modelBy.get(d.model).name}</div>` + row(`${v} CPS`, fmt.signed3(d.mean)), e))
          .on("mouseleave", () => tip.hide());
        if (animate) dots.transition().delay((d, k) => 100 + k * 15).duration(DUR).attr("cx", (d) => x(d.mean));
        const a = avg.get(v), h = data.human.cps[v];
        g.append("line").attr("x1", x(a)).attr("x2", x(a)).attr("y1", cy).attr("y2", cy).attr("stroke", "#141a26").attr("stroke-width", 1.5).attr("stroke-dasharray", "3 3")
          .transition().delay(animate ? DUR : 0).duration(animate ? 600 : 0).attr("x2", x(h));
        g.append("circle").attr("cx", x(a)).attr("cy", cy).attr("r", 8.5).attr("fill", VAR[v].ink).attr("stroke", "#fff").attr("stroke-width", 2)
          .on("mousemove", (e) => tip.show(`<div class="tt-t">LLM average, ${v}</div>` + row("mean CPS", fmt.p3(a)) + row("rank among variations", rl.get(v)), e))
          .on("mouseleave", () => tip.hide());
        g.append("path").attr("d", d3.symbol(d3.symbolStar, 150)()).attr("transform", `translate(${x(h)},${cy})`).attr("fill", col).attr("stroke", "#141a26").attr("stroke-width", 1.5)
          .style("opacity", animate ? 0 : 1).on("mousemove", (e) => tip.show(`<div class="tt-t">Humans, ${v}</div>` + row("mean CPS", fmt.p3(h)) + row("rank among variations", rh.get(v)), e))
          .on("mouseleave", () => tip.hide())
          .transition().delay(animate ? DUR + 500 : 0).duration(400).style("opacity", 1);
        if (!narrow) {
          const t = g.append("text").attr("x", W - m.r + 12).attr("y", cy).attr("dy", "-0.25em").attr("class", "row-label");
          t.append("tspan").attr("fill", "#737b8c").text("Humans ");
          t.append("tspan").attr("font-weight", 600).text(`#${rh.get(v)}`);
          const t2 = g.append("text").attr("x", W - m.r + 12).attr("y", cy).attr("dy", "1.1em").attr("class", "row-label");
          t2.append("tspan").attr("fill", "#737b8c").text("LLMs ");
          t2.append("tspan").attr("font-weight", 600).text(`#${rl.get(v)}`);
        }
      });
    }
    render(true);
    onResize(el, () => render(false));
  }

  function alignTable(tableEl, data) {
    const modelBy = new Map(data.models.map((m) => [m.id, m]));
    const provBy = new Map(data.providers.map((p) => [p.id, p]));
    const rows = data.human.alignment;
    const best = { agr: d3.max(rows, (r) => r.agr), C: d3.max(rows, (r) => r.C), E: d3.max(rows, (r) => r.E), R: d3.max(rows, (r) => r.R) };
    const cell = (v, key, colVar) => {
      const c = d3.color(VAR[colVar].color); c.opacity = 0.08 + (Math.max(0, v) / 0.65) * 0.45;
      return `<td><span class="cell${v === best[key] ? " best" : ""}" style="background:${v > 0 ? c : "transparent"};${v < 0 ? "color:#b3005f" : ""}">${v < 0 ? "−" + Math.abs(v).toFixed(3) : v.toFixed(3)}</span></td>`;
    };
    tableEl.innerHTML = `<thead><tr><th></th><th>Model</th><th>Agr.</th>
      <th style="color:${VAR.Consequentialist.ink}">ρ<sup>(C)</sup></th><th style="color:${VAR.Emotional.ink}">ρ<sup>(E)</sup></th><th style="color:${VAR.Relational.ink}">ρ<sup>(R)</sup></th></tr></thead><tbody>` +
      rows.map((r) => {
        const md = modelBy.get(r.model), pv = provBy.get(md.provider);
        return `<tr><td class="lg">${logoImg(pv, 13)}</td><td>${md.name}</td>` +
          `<td><span class="cell${r.agr === best.agr ? " best" : ""}">${r.agr.toFixed(2)}</span></td>` +
          cell(r.C, "C", "Consequentialist") + cell(r.E, "E", "Emotional") + cell(r.R, "R", "Relational") + "</tr>";
      }).join("") + "</tbody>";
  }

  /* ======================================================================
     Steering: mean CPS curve + ridgeline distributions per alpha
     ====================================================================== */
  function steering(curveEl, distEl, slider, out, data) {
    const curve = d3.group(data.steering.curve, (d) => d.variation);
    const alphas = d3.range(-5, 6);
    const dist = data.steering.dist;
    const state = { alpha: 0 };
    const listeners = [];
    const fmtA = (a) => (a > 0 ? "+" + a : String(a));

    function setAlpha(a) {
      a = Math.max(-5, Math.min(5, Math.round(a)));
      if (a === state.alpha) return;
      state.alpha = a;
      slider.value = a;
      out.textContent = fmtA(a);
      listeners.forEach((f) => f());
    }
    slider.addEventListener("input", () => setAlpha(+slider.value));

    /* --- curve --- */
    let curveFirst = true;
    function renderCurve() {
      curveEl.innerHTML = "";
      const W = curveEl.clientWidth;
      const m = { t: 44, r: W < 600 ? 14 : 104, b: 40, l: 46 };
      const H = 350;
      const svg = d3.select(curveEl).append("svg").attr("width", W).attr("height", H);
      const x = d3.scaleLinear([-5, 5], [m.l, W - m.r]);
      const y = d3.scaleLinear(d3.extent(data.steering.curve.flatMap((d) => [d.lo, d.hi])), [H - m.b, m.t]).nice();
      // legend (centered) + mute/amplify hints
      const lg = svg.append("g");
      let lx = 0;
      VARS.forEach((v) => {
        const gi = lg.append("g").attr("transform", `translate(${lx},0)`);
        gi.append("line").attr("x1", 0).attr("x2", 16).attr("stroke", VAR[v].color).attr("stroke-width", 2.5);
        gi.append("circle").attr("cx", 8).attr("r", 3.5).attr("fill", VAR[v].color);
        const t = gi.append("text").attr("x", 21).attr("dy", "0.35em").attr("class", "row-label").text(v);
        lx += 21 + t.node().getComputedTextLength() + 16;
      });
      lg.attr("transform", `translate(${Math.max(4, (W - lx + 16) / 2)},10)`);
      svg.append("text").attr("x", x(-5)).attr("y", m.t - 8).attr("class", "axis-title").attr("fill", "#737b8c").text("← mute");
      svg.append("text").attr("x", x(5)).attr("y", m.t - 8).attr("class", "axis-title").attr("text-anchor", "end").attr("fill", "#737b8c").text("amplify →");
      svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y).attr("y2", y);
      svg.append("line").attr("class", "zero-line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(10).tickSize(3).tickFormat((d) => (W < 360 && d % 2 ? "" : fmtA(d))));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(3));
      svg.append("text").attr("class", "axis-title-lg").attr("x", (m.l + W - m.r) / 2).attr("y", H - 4).attr("text-anchor", "middle").text("Steering coefficient α");
      svg.append("text").attr("class", "axis-title").attr("transform", `translate(10,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("CPS");

      const guide = svg.append("line").attr("y1", m.t).attr("y2", H - m.b).attr("stroke", "#141a26").attr("stroke-opacity", .4);
      const clip = svg.append("clipPath").attr("id", "steer-clip").append("rect").attr("x", m.l - 8).attr("y", 0).attr("height", H)
        .attr("width", curveFirst && DUR ? 0 : W);
      if (curveFirst && DUR) clip.transition().duration(1400).ease(d3.easeCubicInOut).attr("width", W);
      const series = svg.append("g").attr("clip-path", "url(#steer-clip)");
      const ends = [];
      VARS.forEach((v) => {
        const ds = curve.get(v).slice().sort((a, b) => a.alpha - b.alpha);
        const col = VAR[v].color;
        series.append("path").datum(ds).attr("fill", col).attr("fill-opacity", .1)
          .attr("d", d3.area().x((d) => x(d.alpha)).y0((d) => y(d.lo)).y1((d) => y(d.hi)).curve(d3.curveMonotoneX));
        series.append("path").datum(ds).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2)
          .attr("d", d3.line().x((d) => x(d.alpha)).y((d) => y(d.mean)).curve(d3.curveMonotoneX));
        series.append("g").selectAll("circle").data(ds).join("circle")
          .attr("cx", (d) => x(d.alpha)).attr("cy", (d) => y(d.mean)).attr("r", 3.5).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.2);
        ends.push({ v, y: y(ds[ds.length - 1].mean) });
      });
      if (m.r > 40) {
        ends.sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
        ends.forEach((e) => svg.append("text").attr("x", x(5) + 12).attr("y", e.y).attr("dy", "0.35em").attr("class", "row-label")
          .attr("fill", VAR[e.v].ink).style("font-weight", 600).text(e.v));
      }
      const sel = svg.append("g");
      function update() {
        guide.transition().duration(250).attr("x1", x(state.alpha)).attr("x2", x(state.alpha));
        const pts = VARS.map((v) => curve.get(v).find((c) => c.alpha === state.alpha));
        sel.selectAll("circle").data(pts).join("circle").attr("r", 6).attr("stroke", "#fff").attr("stroke-width", 2)
          .attr("fill", (d) => VAR[d.variation].color)
          .transition().duration(250).attr("cx", (d) => x(d.alpha)).attr("cy", (d) => y(d.mean));
      }
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.r - m.l).attr("height", H - m.b - m.t).attr("fill", "transparent")
        .style("cursor", "ew-resize")
        .on("mousemove", (e) => {
          setAlpha(x.invert(d3.pointer(e)[0]));
          tip.show(`<div class="tt-t">α = ${fmtA(state.alpha)}</div>` + VARS.map((v) => {
            const d = curve.get(v).find((c) => c.alpha === state.alpha);
            return row(`<i style="background:${VAR[v].color}"></i> ${v}`, `${fmt.signed3(d.mean)} <span style="opacity:.6;font-weight:400">[${fmt.p3(d.lo)}, ${fmt.p3(d.hi)}]</span>`);
          }).join(""), e);
        })
        .on("mouseleave", () => tip.hide());
      curveFirst = false;
      update();
      renderCurve.update = update;
    }
    listeners.push(() => renderCurve.update && renderCurve.update());

    /* --- ridgelines --- */
    let distFirst = true;
    function renderDist() {
      distEl.innerHTML = "";
      const W = distEl.clientWidth;
      const cols = W >= 520 ? 3 : 1;
      const labelW = 44, gapX = 18;
      const m = { t: 20, b: 36 };
      const pw = (W - labelW - (cols - 1) * gapX) / cols;
      const rowStep = 24, overlap = 2.1;
      const ph = rowStep * (alphas.length - 1) + rowStep * overlap;
      const H = Math.ceil(3 / cols) * (ph + m.t + m.b);
      const svg = d3.select(distEl).append("svg").attr("width", W).attr("height", H);
      const x = d3.scaleLinear([-0.5, 0.75], [0, pw]);
      const xs = d3.range(-0.55, 0.8, 0.01);
      const base = (a) => rowStep * overlap + (a + 5) * rowStep; // baseline of the ridge for alpha a (top: -5)
      const dens = new Map();
      VARS.forEach((v) => alphas.forEach((a) => dens.set(v + a, kde(dist[v][a] || [], xs, 0.06))));
      const maxD = d3.max([...dens.values()].flat(), (p) => p[1]);
      const hY = d3.scaleLinear([0, maxD], [0, rowStep * overlap]);
      const ridges = [];

      VARS.forEach((v, i) => {
        const cx = labelW + (i % cols) * (pw + gapX);
        const cy = Math.floor(i / cols) * (ph + m.t + m.b) + m.t;
        const pg = svg.append("g").attr("transform", `translate(${cx},${cy})`);
        const col = VAR[v].color;
        pg.append("text").attr("class", "panel-title").attr("x", pw / 2).attr("y", -6).attr("text-anchor", "middle").attr("fill", VAR[v].ink).text(v);
        if (i % cols === 0) {
          pg.selectAll("text.alab").data(alphas).join("text").attr("class", "row-label alab").attr("x", -8).attr("y", (a) => base(a)).attr("dy", "-0.2em")
            .attr("text-anchor", "end").text((a) => `α = ${fmtA(a)}`);
        }
        const rg = pg.selectAll("g.ridge").data(alphas).join("g").attr("class", "ridge");
        const area = (a) => d3.area().defined((p) => p[1] > maxD * 0.006).x((p) => x(p[0])).y0(base(a)).y1((p) => base(a) - hY(p[1])).curve(d3.curveBasis)(dens.get(v + a));
        const top = (a) => d3.line().defined((p) => p[1] > maxD * 0.006).x((p) => x(p[0])).y((p) => base(a) - hY(p[1])).curve(d3.curveBasis)(dens.get(v + a));
        rg.append("path").attr("class", "rfill")
          .attr("fill", col).attr("fill-opacity", (a) => (a <= 0 ? 0.95 : 0.38)).attr("stroke", "#fff").attr("stroke-width", 1.2)
          .attr("d", area)
          .attr("transform", (a) => (distFirst && DUR ? `translate(0,${base(a)}) scale(1,0) translate(0,${-base(a)})` : null))
          .transition("grow").duration(DUR).delay((a, k) => (distFirst ? k * 50 : 0)).attr("transform", "translate(0,0) scale(1,1) translate(0,0)");
        rg.append("path").attr("class", "rtop").attr("d", top).attr("fill", "none").attr("stroke", "#141a26").attr("stroke-width", 1.6).attr("opacity", 0);
        rg.append("rect").attr("x", 0).attr("width", pw).attr("y", (a) => base(a) - rowStep).attr("height", rowStep).attr("fill", "transparent")
          .style("cursor", "pointer")
          .on("mousemove", (e, a) => {
            const vals = dist[v][a] || [];
            tip.show(`<div class="tt-t"><i style="background:${col}"></i>${v} vector, α = ${fmtA(a)}</div>` +
              row("mean CPS", fmt.signed3(d3.mean(vals))) + row("median CPS", fmt.signed3(d3.median(vals))) + row("scenarios", vals.length), e);
          })
          .on("mouseleave", () => tip.hide())
          .on("click", (e, a) => setAlpha(a));
        ridges.push(rg);

        // zero line (dashed) and mean trace (dotted), as in the paper figure
        pg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", ph).attr("stroke", "#141a26").attr("stroke-width", 1.2).attr("stroke-dasharray", "5 3");
        pg.append("g").attr("pointer-events", "none").selectAll("line").data(alphas).join("line")
          .attr("x1", (a) => x(d3.mean(dist[v][a] || [0]))).attr("x2", (a) => x(d3.mean(dist[v][a] || [0])))
          .attr("y1", (a) => base(a) - rowStep).attr("y2", (a) => base(a))
          .attr("stroke", "#141a26").attr("stroke-width", 1.4).attr("stroke-dasharray", "1.5 2.5");
        pg.append("g").attr("class", "axis").attr("transform", `translate(0,${ph + 2})`).call(d3.axisBottom(x).tickValues([-0.4, 0, 0.4]).tickSize(3));
        pg.append("text").attr("class", "axis-title").attr("x", pw / 2).attr("y", ph + 32).attr("text-anchor", "middle").text("Scenario CPS");
      });
      function update() {
        ridges.forEach((rg) => rg.select(".rtop").transition().duration(300).attr("opacity", (a) => (a === state.alpha ? 1 : 0)));
        distEl.querySelectorAll(".alab").forEach((t) => { t.style.fontWeight = t.__data__ === state.alpha ? 600 : 400; });
      }
      distFirst = false;
      update();
      renderDist.update = update;
    }
    listeners.push(() => renderDist.update && renderDist.update());

    return {
      renderCurve() { renderCurve(); onResize(curveEl, renderCurve); },
      renderDist() { renderDist(); onResize(distEl, renderDist); },
    };
  }

  /* ======================================================================
     Metrics: illustrative CPS, one scenario moving along the 0 to 1 MAL scale
     ====================================================================== */
  function cpsIllustration(el) {
    const base = 0.42;
    const rows = [{ k: "Consequentialist", d: 0.09 }, { k: "Emotional", d: 0.07 }, { k: "Relational", d: 0.06 }];
    const W = el.clientWidth;
    const narrow = W < 420;
    const m = { t: 44, r: 14, b: 30, l: narrow ? 64 : 126 };
    const rowH = 34;
    const H = m.t + (rows.length + 1) * rowH + m.b;
    const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
    const x = d3.scaleLinear([0, 1], [m.l, W - m.r]);
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", "malgrad");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#e3e0d7");
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#5b6478");
    rows.forEach((r) => defs.append("marker").attr("id", "ah-" + VAR[r.k].key).attr("viewBox", "0 0 10 10").attr("refX", 7).attr("refY", 5)
      .attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 10).attr("markerHeight", 10).attr("orient", "auto").append("path").attr("d", "M0,0L10,5L0,10z").attr("fill", VAR[r.k].color));

    svg.append("rect").attr("x", x(0)).attr("width", x(1) - x(0)).attr("y", 18).attr("height", 6).attr("rx", 3).attr("fill", "url(#malgrad)");
    svg.append("text").attr("class", "axis-title").attr("x", x(0)).attr("y", 12).text("0: always follows the rule");
    svg.append("text").attr("class", "axis-title").attr("x", x(1)).attr("y", 12).attr("text-anchor", "end").text("1: always breaks it");
    svg.append("line").attr("x1", x(base)).attr("x2", x(base)).attr("y1", 30).attr("y2", H - m.b).attr("stroke", "#5b6478").attr("stroke-dasharray", "3 3");
    svg.append("line").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 30).attr("y2", H - m.b).attr("stroke", "#d2cec4");
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b + 2})`).call(d3.axisBottom(x).ticks(5).tickSize(3));

    const cy0 = m.t + rowH / 2;
    svg.append("text").attr("class", "row-label").attr("x", m.l - 12).attr("y", cy0).attr("dy", "0.35em").attr("text-anchor", "end").style("font-weight", 600).text("Base");
    svg.append("circle").attr("cx", x(base)).attr("cy", cy0).attr("r", 7).attr("fill", "#5b6478").attr("stroke", "#fff").attr("stroke-width", 2);
    svg.append("text").attr("class", "row-label").attr("x", x(base) - 12).attr("y", cy0).attr("dy", "0.35em").attr("text-anchor", "end").text("MAL = 0.42");

    rows.forEach((r, i) => {
      const cy = m.t + (i + 1.5) * rowH;
      const col = VAR[r.k].color;
      svg.append("text").attr("class", "row-label").attr("x", m.l - 12).attr("y", cy).attr("dy", "0.35em").attr("text-anchor", "end").attr("fill", VAR[r.k].ink).style("font-weight", 600)
        .text(narrow ? VAR[r.k].short : r.k);
      svg.append("circle").attr("cx", x(base)).attr("cy", cy).attr("r", 4).attr("fill", "#5b6478").attr("fill-opacity", .5);
      svg.append("line").attr("x1", x(base)).attr("x2", x(base)).attr("y1", cy).attr("y2", cy).attr("stroke", col).attr("stroke-width", 3.5)
        .attr("marker-end", `url(#ah-${VAR[r.k].key})`)
        .transition().delay(400 + i * 350).duration(DUR).ease(d3.easeCubicOut).attr("x2", x(base + r.d));
      svg.append("text").attr("class", "panel-title").style("font-size", "13.5px").attr("x", x(base + r.d) + 12).attr("y", cy).attr("dy", "0.35em").attr("fill", VAR[r.k].ink)
        .attr("opacity", 0).text(`CPS = +${r.d.toFixed(2)}`)
        .transition().delay(400 + i * 350 + DUR).duration(300).attr("opacity", 1);
    });
  }

  window.CMCCharts = { baseViolins, cpsForest, obedienceScatter, humanSlope, alignTable, steering, cpsIllustration };
})();
