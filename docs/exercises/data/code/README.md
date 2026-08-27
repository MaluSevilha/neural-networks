# Data activity: reproducibility

Run the activity from the repository root:

```bash
python3 docs/activities/data/code/data_activity.py
```

The script uses a single random number generator throughout the report that guarantees reproductibility of results:

```python
rng = np.random.default_rng(42)
```

For Exercise 3, place the Kaggle `train.csv` file at:

```text
docs/activities/data/data/train.csv
```

Download it at: https://www.kaggle.com/competitions/spaceship-titanic/.

With this, you can reproduce the results present in `docs/activities/data/code/output.txt` in your terminal. To save a new version of your local run into the document use:

```bash
python3 docs/activities/data/code/data_activity.py > docs/activities/data/code/output.txt
```