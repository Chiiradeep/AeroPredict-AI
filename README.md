# AeroPredict AI — Interactive Jet Engine Health Explorer

AeroPredict AI is an educational machine-learning application for exploring **remaining useful life (RUL)** estimates on NASA's **simulated C-MAPSS turbofan engine FD001** dataset.

It addresses a predictive-maintenance question: *Can sensor histories help estimate how many operating cycles remain before a simulated engine reaches the end of its recorded life?*

## Explore the web app

The **Overview** explains the project to first-time visitors. Select **Explore Engine Demo** to replay a previously held-out validation engine, examine its sensor history, and see the precomputed Random Forest / XGBoost RUL estimates. Use the other sidebar workspaces to explore fleet patterns, sensor trends, engineered features, SHAP insights, and the official test benchmark. **No CSV upload is required or supported.**

> **Research / educational prototype only.** This is simulated historical data, not live aircraft telemetry. Health alert categories are illustrative and are **not** maintenance instructions or a clinically/industrially certified risk assessment.

## Evaluation

A frozen model selected using training-engine validation was evaluated once on NASA's official FD001 test set: **one final-observation prediction per 100 test engines**. For the project run used in the demo, the reported XGBoost scores on *true uncapped RUL* were approximately **MAE 12.49 cycles, RMSE 17.31 cycles, R² 0.826**. These are dataset-specific results; refer to the included `outputs/m7_official_test_metrics.csv` for exact values.

**Limitations:** Models were trained on a 125-cycle capped RUL target. As a result, predictions cannot distinguish longer remaining lives well. Official test metrics are not proof of operational aircraft reliability. SHAP describes the model's statistical behavior, not sensor causation.

## Technical stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Plotly · Streamlit.

## Run locally

```bash
python -m venv .venv
# Activate the virtual environment using your shell's command.
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The GitHub-ready export contains the **preloaded public NASA data**, trained model artifacts and **already computed reports** required to show the interactive demo without running expensive training in the cloud. Models are loaded only from your own locally generated artifacts; do not load untrusted `.joblib` files.

## Project architecture

- `app.py` — Obsidian 2.0 dashboard, visitor landing page and navigation
- `src/` — data loading, preprocessing, feature engineering, model training, explanation, monitoring and evaluation
- `data/raw/` — FD001 source text files, needed for dashboard exploration
- `data/processed/feature_manifest.json` — generated feature description
- `models/` — locally trained saved models and summaries
- `outputs/` — validation, SHAP, monitoring replay and final evaluation reports
- `.streamlit/config.toml` — Obsidian 2.0 color theme

## Reproducibility

Training / optimization scripts and source code are included for study. The app itself uses saved offline monitoring replay rows so a visitor can explore predictions quickly and consistently; it does not stream real sensors or train a new model per visitor. NASA's official test labels were held out until the final evaluation.

## Data source

[NASA C-MAPSS simulated jet engine data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Website

**Live demo:** _Add your Streamlit Community Cloud URL here after deployment._
