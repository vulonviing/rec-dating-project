# Rec-Dating Project

This project studies the `rec-dating` dataset as a role-based bipartite network.

The workflow is organized as a clean four-stage notebook pipeline so that someone can reproduce the project from top to bottom without guessing which file comes next.

## Core Idea

- `rater` nodes send ratings
- `profile` nodes receive ratings
- edge weight is the observed score from `1` to `10`

This framing lets us separate outgoing activity from received attention and apply network measures such as HITS in a role-consistent way.

## Main Questions

1. How strongly do popularity and prestige align on the profile side?
2. How concentrated is received attention?
3. Do the profiles dominating overall interaction also dominate high-rating buckets?
4. Which profile-side features are most aligned with elite interaction and elite high-rating status?

## Project Structure

- `data/`: raw dataset
- `src/rec_dating_project/`: reusable project code
- `scripts/`: analysis scripts used by the notebooks
- `notebooks/`: the main reproducible notebook workflow
- `paper/`: LaTeX paper draft and bibliography
- `outputs/data/`: generated tables reused across notebooks
- `outputs/figures/`: generated figures reused in notebooks and paper

## Environment Setup

The project was last checked with `Python 3.11.9`.

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The raw dataset is expected at:

```text
data/rec-dating/rec-dating.edges
```

Download it manually from the Network Repository page and place it at the path above:

```text
https://networkrepository.com/rec-dating.php
```

## Recommended Notebook Workflow

Run the notebooks in this order:

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_rec_dating_exploration.ipynb`
3. `notebooks/03_applications.ipynb`
4. `notebooks/04_final_plots_for_paper.ipynb`

What each notebook does:

- `01`: inspects the raw file, explains the role-based modeling choice, and builds the cached dataset summary
- `02`: explores popularity, prestige, inequality, and descriptive plots
- `03`: applies the framework to bucket concentration and feature alignment
- `04`: gathers the final paper-facing plots and reference values

All notebooks are written in English and include both:

- technical interpretation
- plain-language interpretation

## How To Run The Notebooks

### Option A: Interactive

Launch JupyterLab from the project root:

```bash
jupyter lab
```

Then open the notebooks and run them in the numbered order above.

### Option B: Fully Reproducible Terminal Execution

If you want to execute everything from the terminal:

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex python3 -m nbconvert --to notebook --execute notebooks/01_data_preparation.ipynb --inplace --ExecutePreprocessor.timeout=1200
MPLCONFIGDIR=/tmp/matplotlib-codex python3 -m nbconvert --to notebook --execute notebooks/02_rec_dating_exploration.ipynb --inplace --ExecutePreprocessor.timeout=1200
MPLCONFIGDIR=/tmp/matplotlib-codex python3 -m nbconvert --to notebook --execute notebooks/03_applications.ipynb --inplace --ExecutePreprocessor.timeout=1200
MPLCONFIGDIR=/tmp/matplotlib-codex python3 -m nbconvert --to notebook --execute notebooks/04_final_plots_for_paper.ipynb --inplace --ExecutePreprocessor.timeout=1200
```

The `MPLCONFIGDIR` prefix helps on headless or restricted environments where Matplotlib cannot write to its default cache directory.

## Generated Outputs And Rebuild Behavior

- The notebooks reuse cached files in `outputs/data/` and `outputs/figures/` when they already exist.
- Missing artifacts are rebuilt automatically by the relevant scripts.
- If you want a full refresh, set `FORCE_REBUILD = True` in the setup cell of the notebook you are running.

## Scripts Used By The Notebook Pipeline

The final notebook workflow relies on these scripts:

- `scripts/01_dataset_overview.py`
- `scripts/02_full_project_analysis.py`
- `scripts/03_profile_rating_extremes.py`
- `scripts/04_profile_feature_alignment.py`
- `scripts/05_degree_distribution_fit.py`

If the notebook templates ever need to be regenerated, run:

```bash
python3 scripts/06_rebuild_notebooks.py
```

## Paper Reproduction

After `04_final_plots_for_paper.ipynb` has been executed, the paper figures should be available under `outputs/figures/`.

To compile the paper from the project root:

```bash
latexmk -pdf -cd paper/main.tex
```

## Notes

- The dataset is large, so cached artifacts are used intentionally to keep the notebook workflow responsive.
- The paper-facing notebook is the final check that the written paper and the generated figures still agree.
