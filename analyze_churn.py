from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
RANDOM_STATE = 42


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = X.select_dtypes(exclude="object").columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
    return preprocessor, numeric, categorical


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    blanks = int(pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum())
    df = load_and_clean(DATA_PATH)

    churn_rate = float(df["Churn"].mean())
    contract_rates = (
        df.groupby("Contract")["Churn"].mean().mul(100).sort_values(ascending=False)
    )
    tenure_summary = (
        df.assign(tenure_group=pd.cut(
            df["tenure"], bins=[-1, 12, 24, 48, 72],
            labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
        )).groupby("tenure_group", observed=False)["Churn"].mean().mul(100)
    )

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    results = {}
    fitted = {}
    for name, estimator in models.items():
        preprocessor, _, _ = build_preprocessor(X)
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision_churn": report["1"]["precision"],
            "recall_churn": report["1"]["recall"],
            "f1_churn": report["1"]["f1-score"],
            "roc_auc": roc_auc_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = pipe

    rf = fitted["Random Forest"]
    feature_names = rf.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(
        rf.named_steps["model"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    top_features = {k: float(v) for k, v in importances.head(15).items()}

    output = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "totalcharges_blanks_removed": blanks,
        "overall_churn_pct": churn_rate * 100,
        "contract_churn_pct": {k: float(v) for k, v in contract_rates.items()},
        "tenure_group_churn_pct": {str(k): float(v) for k, v in tenure_summary.items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": results,
        "random_forest_top_features": top_features,
    }
    Path("results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
