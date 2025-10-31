import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = Path(os.getenv("DATA_DIR", "src/data"))

def log(msg):
    print(f"[INFO] {msg}")

def load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        log(f"ERROR: {name} not found at {path.resolve()}.")
        sys.exit(1)
    log(f"Loading {name} from {path.resolve()}")
    return pd.read_csv(path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Engineering features...")
    out = df.copy()

    out["Title"] = out["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

    out["CabinFirstLetter"] = out["Cabin"].astype(str).str[0]
    out.loc[out["Cabin"].isna(), "CabinFirstLetter"] = np.nan

    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1

    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("count")

    keep = [
        "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "Title", "CabinFirstLetter", "FamilySize", "TicketGroupSize"
    ]
    cols = [c for c in keep if c in out.columns]
    out = out[cols]

    log(f"Feature columns now: {list(out.columns)}")
    return out

def main():
    log(f"Expecting Titanic CSVs in: {DATA_DIR.resolve()}")
    train = load_csv("train.csv")
    test  = load_csv("test.csv")

    log(f"Train shape: {train.shape}")
    log(f"Test  shape: {test.shape}")

    log("Preview missingness (top 10 columns):")
    print(train.isna().sum().sort_values(ascending=False).head(10))

    train_fe = engineer_features(train)
    test_fe  = engineer_features(test)

    if "Survived" not in train_fe.columns:
        log("ERROR: 'Survived' column not found in train.csv.")
        sys.exit(1)

    y = train_fe["Survived"].astype(int)
    X = train_fe.drop(columns=["Survived"])

    numeric = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "TicketGroupSize"]
    categorical = ["Pclass", "Sex", "Embarked", "Title", "CabinFirstLetter"]

    log(f"Numeric features: {numeric}")
    log(f"Categorical features: {categorical}")

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", categorical_tf, categorical)
        ]
    )

    clf = LogisticRegression(max_iter=200)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf)
    ])

    log("Fitting Logistic Regression on full training set...")
    pipe.fit(X, y)

    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    log(f"TRAIN accuracy (on full train.csv): {acc:.4f}")

    log("Generating predictions on test.csv...")
    test_preds = pipe.predict(test_fe)
    test_out = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_preds.astype(int)
    })
    out_path = Path("python_submission.csv")
    test_out.to_csv(out_path, index=False)

    log(f"Saved predictions to: {out_path.resolve()}")
    log("Note: test.csv has no 'Survived' column; skipping accuracy check as instructed.")

if __name__ == "__main__":
    main()
