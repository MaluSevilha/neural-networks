# Neural Networks — Course Portfolio

Portfolio for Artificial Neural Networks and Deep Learning (Insper, 2026.2).

The website is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages through GitHub Actions.

## Local development

### Setup
First create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then download the requirements to properly reproduce the activities results:
```bash
pip install -r requirements.txt
```

### Use

Firstly, activate the environment:
```bash
source .venv/bin/activate
```

Then, you can deploy locally the site using:
```bash
mkdocs serve
```

To see the site, open the local address shown in the terminal.

TO reproduce the activities, look for the `README.md`

## Structure

- `docs/` — website content and assets
- `activities/` — course exercises
- `projects/` — larger project work
- `.github/workflows/` — GitHub Pages deployment
- `mkdocs.yml` — site configuration and navigation
