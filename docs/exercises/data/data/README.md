# Data

`train.csv` (Kaggle **Spaceship Titanic**) is intentionally **not committed** to this repository (see `.gitignore`).

To reproduce Exercise 3 locally:

1. Download `train.csv` from the [Spaceship Titanic competition on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data).
2. Place it at `docs/exercises/data/data/train.csv`.
3. Run `python3 docs/exercises/data/code/data_activity.py` from the repository root.
4. To save the results from the run returned on the terminal, use: `python3 docs/exercises/data/code/data_activity.py > docs/exercises/data/output.txt`

The script will regenerate every figure in `../figures/` and print the exact statistics reported in the "Results Summary" table of the exercise page.
