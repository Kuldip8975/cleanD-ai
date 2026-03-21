import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


CLASSIFIERS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

REGRESSORS = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
}


def detect_task_type(y):
    """Auto-detect classification vs regression."""
    if y.dtype == 'object' or y.nunique() <= 15:
        return "classification"
    return "regression"


def prepare_data(df, target_col, test_size=0.2):
    """Prepare X, y with encoded categoricals."""
    df = df.copy().dropna()
    if target_col not in df.columns:
        return None, None, None, None, "Target column not found."

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode object columns in X
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode y if classification target is string
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))
        y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test, None


def train_and_evaluate(df, target_col, model_names, test_size=0.2):
    """Train multiple models and return results dict."""
    X_train, X_test, y_train, y_test, err = prepare_data(df, target_col, test_size)
    if err:
        return None, err

    task = detect_task_type(df[target_col])
    model_pool = CLASSIFIERS if task == "classification" else REGRESSORS
    results = {}

    for name in model_names:
        if name not in model_pool:
            continue
        model = model_pool[name]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "classification":
                metrics = {
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "F1 Score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                }
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)
                        if y_proba.shape[1] == 2:
                            metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
                        else:
                            metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_proba, multi_class='ovr'), 4)
                except Exception:
                    metrics["ROC-AUC"] = "N/A"
                metrics["Report"] = classification_report(y_test, y_pred, zero_division=0)
                metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred).tolist()
            else:
                metrics = {
                    "R² Score": round(r2_score(y_test, y_pred), 4),
                    "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                fi = pd.Series(model.feature_importances_,
                               index=X_train.columns).sort_values(ascending=False)
                metrics["Feature Importance"] = fi
            elif hasattr(model, 'coef_'):
                fi = pd.Series(np.abs(model.coef_).flatten()[:len(X_train.columns)],
                               index=X_train.columns).sort_values(ascending=False)
                metrics["Feature Importance"] = fi

            results[name] = {"metrics": metrics, "model": model, "task": task}
        except Exception as e:
            results[name] = {"error": str(e)}

    return results, None


def get_feature_importance_df(results):
    """Aggregate feature importances from all models."""
    all_fi = {}
    for name, res in results.items():
        if "error" in res:
            continue
        fi = res["metrics"].get("Feature Importance")
        if fi is not None:
            all_fi[name] = fi
    if not all_fi:
        return None
    df_fi = pd.DataFrame(all_fi).fillna(0)
    df_fi["Average"] = df_fi.mean(axis=1)
    return df_fi.sort_values("Average", ascending=False)