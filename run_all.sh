#!/zsh
set -e  # Exit on error

echo "============================================================"
echo "Starting 18-Stage Capstone Pipeline at $(date '+%H:%M:%S')"
echo "============================================================"

# 1. Environment Setup
export PYTHONPATH=$PYTHONPATH:$(pwd)/code
mkdir -p results/figures results/tables results/metrics results/logs data/processed_data data/splits models

# 1.1 System Prerequisites (Homebrew, Python3, R, Pandoc)
echo "Checking system prerequisites..."

if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing via Homebrew..."
    brew install python
fi
echo "Using Python: $(python3 --version)"

if ! command -v Rscript &> /dev/null; then
    echo "R not found. Installing via Homebrew..."
    brew install r
fi
echo "Using R: $(R --version | head -1)"

if ! command -v pandoc &> /dev/null; then
    echo "Pandoc not found. Installing via Homebrew..."
    brew install pandoc
fi
echo "Using Pandoc: $(pandoc --version | head -1)"

# 1.2 Package Dependencies
echo "Installing Python packages..."
python3 -m pip install -r requirements.txt --quiet
echo "Installing R packages..."
Rscript code/install_deps.R

# 2. R Markdown Stages (01–04)
STAGE=1
for f in code/01_data_exploration.Rmd code/02_data_cleaning.Rmd code/03_data_preparation.Rmd code/04_feature_target_eda.Rmd; do
    echo "[$(date '+%H:%M:%S')] [Stage $STAGE/18] Running: $f..."
    Rscript -e "rmarkdown::render('$f', output_format='html_document', quiet=TRUE)"
    STAGE=$((STAGE + 1))
done

# 3. Python Stages (05–18)
PYTHON_FILES=(
    code/05_splits.py
    code/06_baselines_logit_lasso.py
    code/07_mixed_effects_logit.py
    code/08_gee.py
    code/09_train_xgboost.py
    code/10_train_lightgbm.py
    code/11_train_rf.py
    code/12_calibration.py
    code/13_evaluation.py
    code/14_shap_analysis.py
    code/15_symmetry_test.py
    code/16_robustness.py
    code/17_global_interpretability.py
    code/18_symmetry_mirror_chart.py
)

for f in $PYTHON_FILES; do
    echo "[$(date '+%H:%M:%S')] [Stage $STAGE/18] Running: $f..."
    python3 "$f"
    STAGE=$((STAGE + 1))
done

echo "============================================================"
echo "Pipeline complete at $(date '+%H:%M:%S')!"
echo "Check 'results/' for figures and tables."
echo "============================================================"