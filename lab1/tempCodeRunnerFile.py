import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.stats.api as sms

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import root_mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan, linear_harvey_collier
from scipy.stats import jarque_bera

warnings.filterwarnings("ignore")

df = pd.read_csv("ames_data.csv")

df.columns = [col.replace(".", "") for col in df.columns]
df = df.drop(["Order", "PID"], axis="columns")
df = df.loc[~df["Neighborhood"].isin(["GrnHill", "Landmrk"]), :]
df = df.loc[~(df["GrLivArea"] > 4000), :]
df["SalePrice"] = np.log1p(df["SalePrice"])

def replace_na(df: pd.DataFrame, col: str, value) -> None:
    df.loc[:, col] = df.loc[:, col].fillna(value)

    # Alley : data description says NA means "no alley access"
replace_na(df, "Alley", value="None")

# BedroomAbvGr : NA most likely means 0
replace_na(df, "BedroomAbvGr", value=0)

# BsmtQual etc : data description says NA for basement features is "no basement"
replace_na(df, "BsmtQual", value="No")
replace_na(df, "BsmtCond", value="No")
replace_na(df, "BsmtExposure", value="No")
replace_na(df, "BsmtFinType1", value="No")
replace_na(df, "BsmtFinType2", value="No")
replace_na(df, "BsmtFullBath", value=0)
replace_na(df, "BsmtHalfBath", value=0)
replace_na(df, "BsmtUnfSF", value=0)

# Condition : NA most likely means Normal
replace_na(df, "Condition1", value="Norm")
replace_na(df, "Condition2", value="Norm")

# External stuff : NA most likely means average
replace_na(df, "ExterCond", value="TA")
replace_na(df, "ExterQual", value="TA")

# Fence : data description says NA means "no fence"
replace_na(df, "Fence", value="No")

# Functional : data description says NA means typical
replace_na(df, "Functional", value="Typ")

# GarageType etc : data description says NA for garage features is "no garage"
replace_na(df, "GarageType", value="No")
replace_na(df, "GarageFinish", value="No")
replace_na(df, "GarageQual", value="No")
replace_na(df, "GarageCond", value="No")
replace_na(df, "GarageArea", value=0)
replace_na(df, "GarageCars", value=0)

# HalfBath : NA most likely means no half baths above grade
replace_na(df, "HalfBath", value=0)

# HeatingQC : NA most likely means typical
replace_na(df, "HeatingQC", value="Ta")

# KitchenAbvGr : NA most likely means 0
replace_na(df, "KitchenAbvGr", value=0)

# KitchenQual : NA most likely means typical
replace_na(df, "KitchenQual", value="TA")

# LotFrontage : NA most likely means no lot frontage
replace_na(df, "LotFrontage", value=0)

# LotShape : NA most likely means regular
replace_na(df, "LotShape", value="Reg")

# MasVnrType : NA most likely means no veneer
replace_na(df, "MasVnrType", value="None")
replace_na(df, "MasVnrArea", value=0)

# MiscFeature : data description says NA means "no misc feature"
replace_na(df, "MiscFeature", value="No")
replace_na(df, "MiscVal", value=0)

# OpenPorchSF : NA most likely means no open porch
replace_na(df, "OpenPorchSF", value=0)

# PavedDrive : NA most likely means not paved
replace_na(df, "PavedDrive", value="N")

# PoolQC : data description says NA means "no pool"
replace_na(df, "PoolQC", value="No")
replace_na(df, "PoolArea", value=0)

# SaleCondition : NA most likely means normal sale
replace_na(df, "SaleCondition", value="Normal")

# ScreenPorch : NA most likely means no screen porch
replace_na(df, "ScreenPorch", value=0)

# TotRmsAbvGrd : NA most likely means 0
replace_na(df, "TotRmsAbvGrd", value=0)

# Utilities : NA most likely means all public utilities
replace_na(df, "Utilities", value="AllPub")

# WoodDeckSF : NA most likely means no wood deck
replace_na(df, "WoodDeckSF", value=0)

# FireplaceQu : data description says NA means "no fireplace"
replace_na(df, "FireplaceQu", value="No")

# CentralAir : NA most likely means no central air
replace_na(df, "CentralAir", value="N")

# EnclosedPorch : NA most likely means no enclosed porch
replace_na(df, "EnclosedPorch", value=0)

# Fireplaces : NA most likely means no fireplace
replace_na(df, "Fireplaces", value=0)

# SaleCondition : NA most likely means normal sale
replace_na(df, "SaleCondition", value="Normal")

df = df.replace(
    {
        "MSSubClass": {
            20: "SC20",
            30: "SC30",
            40: "SC40",
            45: "SC45",
            50: "SC50",
            60: "SC60",
            70: "SC70",
            75: "SC75",
            80: "SC80",
            85: "SC85",
            90: "SC90",
            120: "SC120",
            150: "SC150",
            160: "SC160",
            180: "SC180",
            190: "SC190",
        },
        "MoSold": {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        },
    }
)

df = df.replace(
    {
        "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
        "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
        "BsmtFinType1": {
            "No": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6,
        },
        "BsmtFinType2": {
            "No": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6,
        },
        "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Functional": {
            "Sal": 1,
            "Sev": 2,
            "Maj2": 3,
            "Maj1": 4,
            "Mod": 5,
            "Min2": 6,
            "Min1": 7,
            "Typ": 8,
        },
        "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
        "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
        "PavedDrive": {"N": 0, "P": 1, "Y": 2},
        "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
        "Street": {"Grvl": 0, "Pave": 1},
        "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    }
)

df = df.astype(
    {
        "Alley": np.int64,
        "BsmtCond": np.int64,
        "BsmtExposure": np.int64,
        "BsmtFinType1": np.int64,
        "BsmtFinType2": np.int64,
        "BsmtQual": np.int64,
        "ExterCond": np.int64,
        "ExterQual": np.int64,
        "FireplaceQu": np.int64,
        "Functional": np.int64,
        "GarageCond": np.int64,
        "GarageQual": np.int64,
        "HeatingQC": np.int64,
        "KitchenQual": np.int64,
        "LandSlope": np.int64,
        "LotShape": np.int64,
        "PavedDrive": np.int64,
        "PoolQC": np.int64,
        "Street": np.int64,
        "Utilities": np.int64,
    }
)

y = df.pop("SalePrice")

categorical_features = df.select_dtypes(include="object").columns
numerical_features = df.select_dtypes(exclude="object").columns

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.3, random_state=0
)

RIDGE_ALPHA = 2.9
LASSO_ALPHA = 0.0003

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def condition_number(X):
    # współczynnik uwarunkowania macierzy projektowej (z wyrazem wolnym)
    return np.linalg.cond(X)

def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def diagnose_model(model, name, run_hc=False):
    """
    Trenuje pipeline [preprocessor + model], liczy metryki i testy diagnostyczne.
    HC (Harvey–Collier) jest zdefiniowany dla OLS — uruchamiamy go tylko jeśli run_hc=True.
    """
    print_header(f"Model: {name}")

    pipe = Pipeline(steps=[("prep", preprocessor), ("est", model)])
    pipe.fit(X_train, y_train)

    # Metryki
    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)
    print(f"R^2  (train): {pipe.score(X_train, y_train):.4f}")
    print(f"R^2  (test) : {pipe.score(X_test, y_test):.4f}")
    print(f"RMSE (train): {rmse(y_train, y_pred_tr):.4f}")
    print(f"RMSE (test) : {rmse(y_test, y_pred_te):.4f}")

    # Rezydua na zbiorze testowym
    resid_test = y_test - y_pred_te

    # Przetransformowana macierz projektowa (test) + stała do testów
    X_te_design = preprocessor.fit(X_train, y_train).transform(X_test)
    X_te_design = sm.add_constant(X_te_design, has_constant="add")

    # --- Testy normalności błędów: Jarque–Bera ---
    jb_stat, jb_p, skew, kurt = jarque_bera(resid_test)
    print("\n[Jarque–Bera] stat={:.4f}, p-value={:.4g} -> {}".format(
        jb_stat, jb_p, "NIE ODRZUCAMY H0 (normalność)" if jb_p >= 0.05 else "ODRZUCAMY H0 (nienormalność)")
    )

    # --- Test homoskedastyczności: Breusch–Pagan ---
    # H0: homoskedastyczność
    bp_stat, bp_p, f_stat, f_p = het_breuschpagan(resid_test, X_te_design)
    print("[Breusch–Pagan] stat={:.4f}, p-value={:.4g} -> {}".format(
        bp_stat, bp_p, "NIE ODRZUCAMY H0 (homoskedastyczność)" if bp_p >= 0.05 else "ODRZUCAMY H0 (heteroskedastyczność)")
    )

    # --- Test liniowości: Harvey–Collier (tylko OLS) ---
    if run_hc:
        # Fit OLS na train, ale diagnozę robimy zgodnie z definicją na modelu OLS
        X_tr_design = preprocessor.transform(X_train)
        X_tr_design = sm.add_constant(X_tr_design, has_constant="add")
        ols_res = sm.OLS(y_train, X_tr_design).fit()
        hc_stat, hc_p = linear_harvey_collier(ols_res)
        print("[Harvey–Collier] stat={:.4f}, p-value={:.4g} -> {}".format(
            hc_stat, hc_p, "NIE ODRZUCAMY H0 (liniowość funkcji)" if hc_p >= 0.05 else "ODRZUCAMY H0 (nieliniowość)")
        )
    else:
        print("[Harvey–Collier] pomijam (test zdefiniowany dla OLS).")

    # --- Współliniowość: współczynnik uwarunkowania macierzy X (train) ---
    cn = condition_number(X_tr_design if run_hc else sm.add_constant(preprocessor.transform(X_train), has_constant="add"))
    print(f"\nWspółczynnik uwarunkowania (cond(X)): {cn:.2f}")
    print("Heurystyka: >30 wskazuje na poważną współliniowość; 15–30 umiarkowaną.\n")

    return {
        "pipeline": pipe,
        "jb_p": jb_p,
        "bp_p": bp_p,
        "cond_number": cn,
    }

# === 1) OLS (bez regularyzacji) ===
ols_results = diagnose_model(LinearRegression(), "OLS (LinearRegression)", run_hc=True)

# === 2) Ridge (L2) ===
ridge_results = diagnose_model(Ridge(alpha=RIDGE_ALPHA, random_state=0), f"Ridge (alpha={RIDGE_ALPHA})", run_hc=False)

# === 3) Lasso (L1) ===
lasso_results = diagnose_model(Lasso(alpha=LASSO_ALPHA, max_iter=50_000, random_state=0), f"Lasso (alpha={LASSO_ALPHA})", run_hc=False)

# === Zbiorcza interpretacja przy α=0.05 ===
def verdict(p, name, h0_text, h1_text):
    return f"{name}: {'NIE ODRZUCAMY H0 ('+h0_text+')' if p >= 0.05 else 'ODRZUCAMY H0 ('+h1_text+')'} (p={p:.4g})"

print_header("Wnioski (α = 0.05)")
print("OLS:")
print(" - " + verdict(ols_results["jb_p"], "Normalność (JB)", "normalność", "nienormalność"))
print(" - " + verdict(ols_results["bp_p"], "Homoskedastyczność (BP)", "stała wariancja", "heteroskedastyczność"))
print(" - Harvey–Collier: patrz wyżej (drukowane przy OLS).")
print(f" - Współczynnik uwarunkowania: {ols_results['cond_number']:.2f}")

print("\nRidge:")
print(" - " + verdict(ridge_results["jb_p"], "Normalność (JB)", "normalność", "nienormalność"))
print(" - " + verdict(ridge_results["bp_p"], "Homoskedastyczność (BP)", "stała wariancja", "heteroskedastyczność"))
print(f" - Współczynnik uwarunkowania: {ridge_results['cond_number']:.2f}")

print("\nLasso:")
print(" - " + verdict(lasso_results["jb_p"], "Normalność (JB)", "normalność", "nienormalność"))
print(" - " + verdict(lasso_results["bp_p"], "Homoskedastyczność (BP)", "stała wariancja", "heteroskedastyczność"))
print(f" - Współczynnik uwarunkowania: {lasso_results['cond_number']:.2f}")

print("\nUwaga:")
print("- Test Harvey–Collier jest przeznaczony dla OLS; dla modeli z regularyzacją raportujemy JB i BP.")
print("- Wysoki współczynnik uwarunkowania sugeruje współliniowość; regularyzacja zwykle ją łagodzi, ale nie zmienia samej macierzy X.")
