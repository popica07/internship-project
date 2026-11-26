# Notebook builders

These scripts generate the study notebooks in `notebooks/`. The notebooks only read saved outputs; the experiments themselves are run by the `src/` modules.

Run them from the project root:

    PYTHONPATH=scripts/notebook_builders .venv/bin/python scripts/notebook_builders/d4.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/D4_interpretable_models.ipynb

`common.py` holds the shared setup cell and the writer.
