# Contextual MoralChoice: Project Page

Static website for *On the Context Sensitivity of LLM Moral Judgment* (Sauter & Schirmer, EMNLP 2026).
Plain HTML/CSS/JS (D3 + KaTeX from cdnjs), no build step.

## View locally

The page loads JSON with `fetch`, so serve it over HTTP rather than opening `index.html` directly:

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Deploy (GitHub Pages)

Push this folder to a repo and enable Pages (Settings → Pages → deploy from branch, root `/`).
`.nojekyll` is included so files are served as-is.

## Regenerating the data

All numbers come from the main repo (`../contextual_moralchoice`):

```bash
pip install pandas
python scripts/build_data.py --repo ../contextual_moralchoice
```

This writes `data/site.json` (base MAL, CPS statistics, base-vs-variation scatter, steering curves and
distributions, human survey numbers, and the featured scenarios used in the hero and dataset viewer) and
`data/explorer.json` (every scenario's texts and each model's MAL per version, loaded by the explorer).
The featured scenarios are set in `FEATURED`; the human-survey numbers are hard-coded in `HUMAN` (from the
paper; the raw survey data is not in the repo). Provider logos are trimmed PNG renders of the SVGs in
`assets/logos/`.

## Layout

```
index.html          page content
css/style.css       styles (variation colours: C #00AEB3, E #F58137, R #ED017D)
js/common.js        shared helpers (tooltip, word diff, KDE, formatting)
js/charts.js        result charts (D3)
js/explorer.js      model explorer
js/app.js           bootstrap: nav, reveals, hero, dataset/metric visuals
assets/logos/       provider logos
```
