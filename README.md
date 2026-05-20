# Capstone Project: Bank Failure and Growth Prediction Pipeline

## 1. Project Overview

**What is this project about?**
This project predicts bank financial distress (regulatory enforcement, failure, or critical undercapitalization) and financial growth (sustained profitability improvement) one year ahead. By combining historical FDIC Call Report data spanning 1959–2024, macroeconomic indicators, and regulatory enforcement records for approximately 24,000 U.S. commercial banks, the project identifies both symmetric and asymmetric drivers of banking distress and growth. The final goal is to deliver robust, interpretable machine learning models that provide actionable insights for regulators, risk managers, and financial analysts.

**What does the project do?**
It executes an end-to-end, 18-stage analytical pipeline that automates:
1. **Data Engineering:** Ingesting, cleaning, and merging raw multi-source financial, regulatory, and macroeconomic datasets into an analysis-ready panel.
2. **Feature Engineering:** Constructing CAMELS-aligned financial ratios, year-over-year dynamics, three-year rolling volatility measures, intra-year quarterly statistics, and loan composition features.
3. **Exploratory Analysis:** Quantifying feature–target discrimination via Cohen's d effect sizes, stratified subgroup analysis, and pre-crisis trajectory profiling.
4. **Feature Selection:** Two-stage filtering using Cohen's d thresholds and iterative VIF elimination to remove uninformative and collinear features.
5. **Machine Learning Modeling:** Training pooled baselines (logistic regression, LASSO), panel-aware econometric models (Mixed Effects Logit, Generalized Estimating Equations (GEE)) and hyperparameter-tuned tree ensembles (XGBoost, LightGBM, Random Forest) on a leakage-aware temporal split.
6. **Evaluation**: Threshold-free (PR-AUC, ROC-AUC, Brier) and threshold-dependent (F2-optimal precision, recall, sensitivity, specificity) metrics on a held-out 2018–2023 test window.
7. **Interpretability**: SHAP-based feature attribution for tree models, standardized coefficient analysis for linear models, and a cross-model consensus heatmap comparing normalized importance rankings across all six specifications.
8. **Symmetry Analysis**: A novel test that compares binned SHAP curves across the crisis and growth models using Spearman rank correlation — revealing whether a feature that drives bank failure symmetrically drives bank growth when reversed, or whether the relationship is fundamentally asymmetric.
---

## 2. Repository Structure

The repository is organized into the following primary directories to ensure reproducibility and logical separation of concerns:

```text
capstone_project/
|
├── code/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   ├── preprocessing.py
│   │   └── splits.py
│   ├── 01_data_exploration.Rmd
│   ├── 01_data_exploration.html
│   ├── 02_data_cleaning.Rmd
│   ├── 02_data_cleaning.html
│   ├── 03_data_preparation.Rmd
│   ├── 03_data_preparation.html
│   ├── 04_feature_target_eda.Rmd
│   ├── 04_feature_target_eda.html
│   ├── 05_splits.py
│   ├── 06_baselines_logit_lasso.py
│   ├── 07_mixed_effects_logit.py
│   ├── 08_gee.py
│   ├── 09_train_xgboost.py
│   ├── 10_train_lightgbm.py
│   ├── 11_train_rf.py
│   ├── 12_calibration.py
│   ├── 13_evaluation.py
│   ├── 14_shap_analysis.py
│   ├── 15_symmetry_test.py
│   ├── 16_robustness.py
│   ├── 17_global_interpretability.py
│   ├── 18_symmetry_mirror_chart.py
│   └── install_deps.R
|
├── data/
│   ├── processed_data/
│   │   ├── bank_panel.rds
│   │   ├── df.rds
│   │   ├── df_clean.rds
│   │   ├── df_model.csv
│   │   ├── df_model.rds
│   │   ├── df_model_crisis.csv
│   │   ├── df_model_growth.csv
│   │   ├── selected_features.txt
│   │   ├── selected_features_crisis.txt
│   │   ├── selected_features_growth.txt
│   │   ├── splits.csv
│   │   ├── splits_crisis.csv
│   │   └── splits_growth.csv
│   └── raw_data/
│       ├── EDOOrders.csv
│       ├── call-reports-balance-sheets-Jan2026.dta
│       ├── call-reports-income-statements-Jan2026.dta
│       ├── failures.csv
│       ├── interest_rates.csv
│       ├── macrodata.csv
│       ├── occ_new.csv
│       └── occ_old.xlsx
|
├── models/
│   └── ... (trained model files, e.g., joblib, json)
|
├── paper/
│   └── (files from Overleaf project)
|
├── results/
│   ├── figures/
│   │   └── ... (high-resolution EDA and interpretability PNGs)
│   ├── logs/
│   │   └── ... (optuna hyperparameter tuning SQLite DBs)
│   ├── metrics/
│   │   └── ... (evaluation CSVs, JSONs, and coefficients)
│   └── tables/
│       └── ... (generated LaTeX/CSV tables)
|
├── README.md
├── capstone_project.Rproj
├── historical_call_data_dictionary.xlsx
├── requirements.txt
├── reset_project.sh
└── run_all.sh
```

- **`code/`**: The core directory containing all R Markdown (data processing/EDA) and Python scripts (modeling/evaluation) for the 18 stages of the pipeline.
- **`data/`**: Divided into `raw_data` (initial inputs) and `processed_data` (cleaned datasets ready for modeling).
- **`results/`**: The primary output directory. Auto-generated by the pipeline, containing:
  - `figures/`: High-resolution charts, density plots, and interpretability graphs.
  - `tables/`: Summary statistics, metric comparisons, and data tables.
  - `metrics/`: Pickled SHAP values, performance JSONs, and evaluation logs.
  - `logs/`: Optuna SQLite databases and training logs for hyperparameter tuning.
- **`paper/`**: Drafts and academic write-ups for the final Capstone report.
- **`models/`**: Serialized, trained models for downstream inference.

---

## 3. Data Architecture

### 3.1 Raw Data (`data/raw_data/`)
Note: Since the datasets are large in size, they are not included in the repository. They can be downloaded from the following Google Drive folder: https://drive.google.com/drive/folders/1yogDXmFlKMXfxrQrRgF8dYyenvz8gvOq?usp=share_link

This directory must contain the initial, unmodified datasets before running the pipeline:
- `call-reports-balance-sheets-Jan2026.dta`: FDIC Call Report balance sheets.
- `call-reports-income-statements-Jan2026.dta`: FDIC Call Report income statements.
- `failures.csv`: Historical record of bank failures.
- `interest_rates.csv` / `macrodata.csv`: Temporal macroeconomic indicators.
- `EDOOrders.csv` / `occ_new.csv` / `occ_old.xlsx`: Enforcement action and regulatory datasets.

Data Sources:
- FDIC Call Reports: Federal Reserve Bank of New York, https://www.newyorkfed.org/research/banking_research/balance-sheets-income-statements
- Failures: Federal Deposit Insurance Corporation, https://banks.data.fdic.gov/explore/failures
- Interest Rates: Federal Reserve Economic Data, https://fred.stlouisfed.org/series/FEDFUNDS
- Macroeconomic Data: World Bank Open Data, https://databank.worldbank.org/source/world-development-indicators/Type/TABLE/preview/on#
- OCC Actions: Office of the Comptroller of the Currency, https://apps.occ.gov/EASearch
- FDIC Orders: Federal Deposit Insurance Corporation, https://orders.fdic.gov/s/searchform

### 3.2 Processed Data (`data/processed_data/`)
These files are automatically generated by the early R Markdown stages of the pipeline:
- `df_clean.rds`: The initial cleaned and merged panel dataset.
- `df_model.rds` / `df_model.csv` / `df_model_crisis.csv` / `df_model_growth.csv`: The finalized modeling datasets containing engineered features and lagged targets.
- `splits*.csv`: Row indices/IDs defining the temporal Train, Validation, and Test splits.
- `selected_features*.txt`: Lists of final predictor variables selected for the failure and growth models.

---

## 4. Pipeline Stages and File Purposes

The core logic resides in the `code/` directory, mapped sequentially from `01` to `18`. 

### Stage A: Data Engineering & EDA (R Markdown)
- **`01_data_exploration.Rmd`**: Conducts initial exploratory analysis on raw datasets. Identifies missingness patterns, distributional anomalies, and basic summary statistics. *(Outputs: `01_data_exploration.html`)*
- **`02_data_cleaning.Rmd`**: Handles missing values, standardizes column data types, aligns time-series indices, and merges the disparate raw data sources. *(Outputs: `df_clean.rds`, `02_data_cleaning.html`)*
- **`03_data_preparation.Rmd`**: Prepares the final target variables (Failure, Growth), constructs complex derived features, creates lagged variables to prevent data leakage, and outputs final modeling datasets. *(Outputs: `df_model.rds`, `df_model.csv`, `03_data_preparation.html`)*
- **`04_feature_target_eda.Rmd`**: Conducts in-depth bivariate EDA against the target variables. Calculates Cohen's *d* effect sizes to filter and visualize the most statistically significant features via density plots and boxplots. *(Outputs: HTML reports and charts in `results/figures/`, two final dataframes with selected features for failure and growth models (`df_model_crisis.csv`, `df_model_growth.csv`)) The 2 dataframes are the input to the modeling stage.

### Stage B: Modeling Framework (Python)
- **`05_splits.py`**: Executes temporal data splitting to generate non-overlapping Training, Validation, and Test sets to prevent lookahead bias. *(Outputs: `splits.csv`)*
- **`06_baselines_logit_lasso.py`**: Trains linear baselines (Logistic Regression, Lasso) to establish a performance floor.
- **`07_mixed_effects_logit.py`**: Trains Mixed Effects Logistic Regression to account for unobserved heterogeneity across time or banking groups.
- **`08_gee.py`**: Implements Generalized Estimating Equations (GEE) to handle longitudinal panel data correlations.
- **`09_train_xgboost.py`**: Trains and optimizes an XGBoost model using Optuna for hyperparameter tuning. *(Logs to `results/logs/optuna_xgb.db`)*
- **`10_train_lightgbm.py`**: Trains and optimizes a LightGBM model, optimized for speed and handling sparse features.
- **`11_train_rf.py`**: Trains a Random Forest ensemble model.

### Stage C: Evaluation, Interpretability, and Symmetry (Python)
- **`12_calibration.py`**: Checks model probability calibration (e.g., Platt Scaling, Isotonic Regression) and computes Brier scores.
- **`13_evaluation.py`**: Computes standard classification metrics (AUC-ROC, PR-AUC, F1-Score, Precision, Recall) across all trained models on the test set.
- **`14_shap_analysis.py`**: Computes SHAP values for the advanced tree-based models to derive feature importances and local explanations.
- **`15_symmetry_test.py`**: Conducts rigorous statistical tests to determine if the directional impact of features is symmetric across failure and growth.
- **`16_robustness.py`**: Performs robustness checks to ensure model stability across different asset size classes and temporal holdouts.
- **`17_global_interpretability.py`**: Synthesizes pipeline findings into global feature importance rankings.
- **`18_symmetry_mirror_chart.py`**: Generates the capstone's flagship visualizations comparing the magnitude and direction of feature importances on bank failure versus growth.

---

## 5. Reproducing Paper Figures and Tables

All figures and tables included in the final Capstone paper are automatically generated by the pipeline. If you need to reproduce a specific asset from the paper, refer to the following mapping:

### Figures
- **EDA & Distributions**: All density plots, boxplots, and target subgroup comparisons are generated by `code/04_feature_target_eda.Rmd`. These are saved as PNGs in `results/figures/`.
- **SHAP Interpretability Plots**: Global summary plots and local feature dependency plots for the tree-based models are generated by `code/14_shap_analysis.py`.
- **Cross-Model Importance Heatmap**: The global consensus ranking visualization comparing all six specifications is generated by `code/17_global_interpretability.py`.
- **Symmetry Mirror Charts**: The flagship visualization contrasting feature impacts on failure versus growth is produced by `code/18_symmetry_mirror_chart.py`.

### Tables
- **Model Performance Metrics**: LaTeX (`.tex`) and CSV tables summarizing classification metrics (PR-AUC, ROC-AUC, Brier score, Precision, Recall) across all models on the test set are generated by `code/13_evaluation.py` and saved to `results/tables/`.
- **Linear Coefficients & Significance**: Tables reporting standardized coefficients and p-values for the statistical baselines are generated by `code/06_baselines_logit_lasso.py`, `code/07_mixed_effects_logit.py`, and `code/08_gee.py`.
- **Descriptive Statistics**: Attrition tables and feature summary statistics are produced during `code/01_data_exploration.Rmd` and `code/02_data_cleaning.Rmd`.

---

## 6. How to Run the Pipeline

### Execution Steps
1. **Prepare the Data**: Ensure all files listed in **3.1 Raw Data** are present inside the `data/raw_data/` directory.
2. **Open Terminal**: Navigate to the root directory of this repository (`capstone_project/`).
3. **Run the Master Script**:
   Execute the automated run script. This script automatically provisions dependencies (Python packages via `requirements.txt` and R packages via `install_deps.R`), creates necessary output directories, and runs stages 1 through 18 sequentially.
   ```bash
   zsh run_all.sh
   # OR
   bash run_all.sh
   ```
4. **Monitor Progress**: The console will display timestamped updates for each stage. Total runtime depends heavily on the hyperparameter tuning search space and your machine's hardware capabilities.
5. **Review Outputs**: Once complete, navigate to the `results/` directory to view all generated tables, evaluation metrics, and figures.

### Running Individual Stages
If you need to re-run a specific stage without executing the entire pipeline:
- **For an R Markdown stage:**
  ```bash
  Rscript -e "rmarkdown::render('code/04_feature_target_eda.Rmd')"
  ```
- **For a Python stage:**
  Ensure your `PYTHONPATH` includes the `code/` directory.
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/code
  python code/09_train_xgboost.py
  ```
