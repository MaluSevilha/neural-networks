# Data activity: reproducibility

Run the activity from the repository root:

```bash
python3 docs/exercises/perceptron/code/perceptron_activity.py
```

The script uses a single random number generator throughout the report that guarantees reproductibility of results:

```python
rng = np.random.default_rng(42)
```

With this, you can reproduce the results present in `docs/activities/data/perceptron/output.txt` in your terminal. To save a new version of your local run into the document use:

```bash
python3 docs/exercises/perceptron/code/perceptron_activity.py > docs/exercises/perceptron/code/output.txt
```