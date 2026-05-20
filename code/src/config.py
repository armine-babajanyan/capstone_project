"""Project-wide constants. Single source of truth for paths, labels, split years."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = ROOT / "data" / "processed_data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

def get_clean_csv(label: str) -> Path:
    return DATA_PROCESSED / f"df_model_{label}.csv"

def get_splits_csv(label: str) -> Path:
    return DATA_PROCESSED / f"splits_{label}.csv"

LABELS = ["crisis", "growth"]
ID_COLS = ["id_rssd", "year"]

# Feature-year boundaries. Row at year t has a label describing year t+1.
# The one-year gaps (2014, 2018) prevent training labels from leaking into
# validation/test features.
TRAIN_YEARS = (1980, 2011)
VAL_YEARS = (2013, 2016)
TEST_YEARS = (2018, 2023)

# Columns used to *construct* the label. Must never appear as features.
LABEL_COMPONENT_COLS = [
    "notice_flag", "occ_old_flag", "occ_new_flag",
    "fail_flag", "critically_undercapitalized_flag"
    ]

RANDOM_SEED = 42
N_CV_FOLDS = 5
CV_GAP = 1

# Dollar-amount and count features with extreme right skew.
# These are log1p-transformed before standardization to compress
# the 25,000× max/median ratios that otherwise dominate linear models.
DOLLAR_COLS = [
    "assets", "equity", "ln_tot_gross", "deposits", "liab_tot",
    "cash", "securities", "llres", "fixed_ass", "oreo",
    "othbor_liab", "subdebt", "ytdnonint_inc", "ytdnonint_exp",
    "num_employees", "pop_total", "gdp",
    # Rolling standard deviations of dollar amounts inherit the same skew
    "equity_sd_3y", "assets_sd_3y", "loans_sd_3y", "deposits_sd_3y",
    "netinc_sd_3y",
    # Intra-year dynamics (quarterly dollar-amount volatilities)
    "assets_sd_intra", "loans_sd_intra", "deposits_sd_intra",
    "equity_sd_intra",
]

# Training samples from this year onward receive 2× weight.
# Helps the model focus on the modern banking regime (post-Basel II,
# post-Riegle-Neal) that is more representative of val/test conditions.
RECENCY_WEIGHT_YEAR = 2000

# Canonical pretty feature names for charts.
FEATURE_NAMES: dict[str, str] = {
    "assets": "Assets",
    "equity": "Equity",
    "ln_tot_gross": "Loans",
    "liab_tot": "Liabilities",
    "ytdnetinc": "Net Income",
    "cash": "Cash",
    "securities": "Securities",
    "llres": "Loan Loss Reserves",
    "fixed_ass": "Fixed Assets",
    "oreo": "Other Real Estate Owned",
    "othbor_liab": "Other Borrowed Liabilities",
    "subdebt": "Subordinated Debt",
    "ytdnonint_inc": "Non-interest Income",
    "ytdnonint_exp": "Non-interest Expense",
    "num_employees": "Number of Employees",
    "age": "Bank Age",
    "capital_ratio": "Capital Ratio",
    "roa": "ROA",
    "roe": "ROE",
    "nim": "NIM",
    "cost_of_funding": "Cost of Funding",
    "npl_ratio": "NPL Ratio",
    "asset_growth": "Asset Growth",
    "deposit_growth": "Deposit Growth",
    "loan_growth": "Loan Growth",
    "liab_growth": "Liability Growth",
    "equity_growth": "Equity Growth",
    "capital_ratio_chg": "Capital Ratio Change",
    "roa_chg": "ROA Change",
    "roe_chg": "ROE Change",
    "nim_chg": "NIM Change",
    "cost_of_funding_chg": "Cost of Funding Change",
    "npl_chg": "NPL Change",
    "equity_sd_3y": "Equity Volatility (3Y)",
    "assets_sd_3y": "Assets Volatility (3Y)",
    "loans_sd_3y": "Loans Volatility (3Y)",
    "deposits_sd_3y": "Deposits Volatility (3Y)",
    "netinc_sd_3y": "Net Income Volatility (3Y)",
    "roa_sd_3y": "ROA Volatility (3Y)",
    "prop_ln_re": "Prop. Real Estate Loans",
    "prop_ln_ci": "Prop. C&I Loans",
    "prop_ln_cons": "Prop. Consumer Loans",
    "prop_ln_agr": "Prop. Agricultural Loans",
    "prop_ln_cc": "Prop. Credit Cards",
    "prop_ln_fi": "Prop. Financial Institutions",
    "prop_demand_dep": "Prop. Demand Deposits",
    "assets_sd_intra": "Assets Volatility (Intra)",
    "loans_sd_intra": "Loans Volatility (Intra)",
    "deposits_sd_intra": "Deposits Volatility (Intra)",
    "equity_sd_intra": "Equity Volatility (Intra)",
    "assets_max_decline": "Assets Max Decline",
    "loans_max_decline": "Loans Max Decline",
    "deposits_max_decline": "Deposits Max Decline",
    "equity_max_decline": "Equity Max Decline",
    "gdp": "GDP",
    "inflation": "Inflation",
    "unemployment": "Unemployment Rate",
    "real_int_rate": "Real Interest Rate",
    "pop_total": "Population Total",
    "fed_funds_rate": "Fed Funds Rate",
}


def pretty(feat: str) -> str:
    """Return the pretty name for a feature, or title-case the raw name."""
    return FEATURE_NAMES.get(feat, feat.replace("_", " ").title())
