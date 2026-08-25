# Neural Networks — Course Portfolio

Portfolio for Artificial Neural Networks and Deep Learning (Insper, 2026.2).

The website is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages through GitHub Actions.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Then open the local address shown by MkDocs.

## Structure

- `docs/` — website content and assets
- `activities/` — course exercises
- `projects/` — larger project work
- `.github/workflows/` — GitHub Pages deployment
- `mkdocs.yml` — site configuration and navigation
