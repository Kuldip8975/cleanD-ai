import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def handle_missing_values(df, strategy, num_fill=None):
    df = df.copy()
    if strategy == "Drop Rows":
        df.dropna(inplace=True)
    elif strategy == "Fill Values":
        for col in df.columns:
            if df[col].dtype == 'object':
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean() if num_fill == "Mean" else df[col].median(), inplace=True)
    elif strategy == "KNN Impute":
        from sklearn.impute import KNNImputer
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])
        for col in df.select_dtypes(include='object').columns:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
    elif strategy == "Iterative Impute":
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            df[num_cols] = IterativeImputer(random_state=42, max_iter=10).fit_transform(df[num_cols])
        for col in df.select_dtypes(include='object').columns:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def handle_outliers(df, method):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return df
    if method == "IQR Method":
        Q1, Q3 = df[numeric_cols].quantile(0.25), df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[numeric_cols] < (Q1 - 1.5*IQR)) | (df[numeric_cols] > (Q3 + 1.5*IQR))).any(axis=1)]
    elif method == "Z-Score Method":
        std = df[numeric_cols].std().replace(0, np.nan)
        z = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / std)
        df = df[(z < 3).all(axis=1)]
    elif method == "Isolation Forest":
        preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(
            df[numeric_cols].fillna(df[numeric_cols].median()))
        df = df[preds == 1]
    elif method == "Winsorization":
        for col in numeric_cols:
            df[col] = df[col].clip(df[col].quantile(0.05), df[col].quantile(0.95))
    return df


def encode_data(df, encoding_type):
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return df
    if encoding_type == "Label Encoding":
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    elif encoding_type == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif encoding_type == "Ordinal Encoding":
        from sklearn.preprocessing import OrdinalEncoder
        df[cat_cols] = OrdinalEncoder().fit_transform(df[cat_cols].astype(str))
    return df


def normalize_data(df_subset, normalization):
    df_subset = df_subset.copy()
    if normalization == "Min-Max Scaler":
        df_subset[:] = MinMaxScaler().fit_transform(df_subset)
    elif normalization == "Standard Scaler":
        df_subset[:] = StandardScaler().fit_transform(df_subset)
    elif normalization == "Robust Scaler":
        df_subset[:] = RobustScaler().fit_transform(df_subset)
    elif normalization == "Z-Score (Custom)":
        df_subset = (df_subset - df_subset.mean()) / df_subset.std().replace(0, 1)
    elif normalization == "Log Transform":
        df_subset = np.log1p(df_subset - df_subset.min() + 1)
    return df_subset


def drop_low_correlation_columns(df, target_col, threshold):
    df = df.copy()
    if target_col not in df.columns:
        return df
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        try:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        except Exception:
            df_enc.drop(columns=[col], inplace=True)
    numeric_cols = df_enc.select_dtypes(include=np.number).columns.tolist()
    if target_col not in numeric_cols:
        return df
    corr = df_enc[numeric_cols].corr()[target_col].abs()
    cols_to_drop = [c for c in numeric_cols if c != target_col and corr.get(c, 1.0) < threshold and c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    return df, cols_to_drop


def preprocess_dataset(df, drop_cols, handle_missing, num_fill,
                       remove_duplicates, outlier_method,
                       encoding_type, normalization,
                       target_col=None, drop_low_corr=False, corr_threshold=0.05,
                       progress_callback=None):
    df_processed = df.copy()
    total_steps = 8
    step = 0
    log = []

    def tick(msg):
        nonlocal step
        log.append(msg)
        step += 1
        if progress_callback:
            progress_callback(step, total_steps)

    if drop_cols:
        existing = [c for c in drop_cols if c in df_processed.columns]
        df_processed.drop(columns=existing, inplace=True)
        tick(f"Dropped {len(existing)} selected column(s).")
    else:
        tick("No columns manually dropped.")

    if drop_low_corr and target_col:
        result = drop_low_correlation_columns(df_processed, target_col, corr_threshold)
        if isinstance(result, tuple):
            df_processed, dropped = result
            tick(f"Low-corr filter: dropped {len(dropped)} column(s).")
        else:
            df_processed = result
            tick("Low-corr filter applied.")
    else:
        tick("Low-correlation filter skipped.")

    before_na = df_processed.isna().sum().sum()
    df_processed = handle_missing_values(df_processed, handle_missing, num_fill)
    filled = before_na - df_processed.isna().sum().sum()
    tick(f"Missing values: {filled} handled via {handle_missing}.")

    if remove_duplicates:
        before = len(df_processed)
        df_processed.drop_duplicates(inplace=True)
        tick(f"{before - len(df_processed)} duplicate rows removed.")
    else:
        tick("Duplicate removal skipped.")

    if outlier_method and outlier_method != "None":
        before = len(df_processed)
        df_processed = handle_outliers(df_processed, outlier_method)
        tick(f"{before - len(df_processed)} outliers removed ({outlier_method}).")
    else:
        tick("Outlier removal skipped.")

    if normalization and normalization != "None":
        cols_to_norm = [c for c in df_processed.select_dtypes(include=['int64','float64']).columns if c != target_col]
        if cols_to_norm:
            df_processed[cols_to_norm] = normalize_data(df_processed[cols_to_norm], normalization)
        tick(f"Normalized {len(cols_to_norm)} column(s) with {normalization}.")
    else:
        tick("Normalization skipped.")

    if encoding_type and encoding_type != "None":
        before_cols = df_processed.shape[1]
        df_processed = encode_data(df_processed, encoding_type)
        tick(f"Encoding ({encoding_type}): {df_processed.shape[1] - before_cols} new col(s).")
    else:
        tick("Encoding skipped.")

    if progress_callback:
        progress_callback(total_steps, total_steps)

    return df_processed, log


def handle_imbalance(df, target_col, method="None"):
    if not target_col or target_col not in df.columns:
        return df, "No valid target column."
    if df[target_col].nunique() <= 1:
        return df, "Target needs at least 2 classes."
    X_numeric = df.drop(columns=[target_col]).select_dtypes(include=np.number)
    y = df[target_col]
    if X_numeric.shape[1] == 0:
        return df, "No numeric features for resampling."
    try:
        if method == "SMOTE Oversampling":
            X_res, y_res = SMOTE(random_state=42).fit_resample(X_numeric, y)
        elif method == "Random Undersampling":
            X_res, y_res = RandomUnderSampler(random_state=42).fit_resample(X_numeric, y)
        else:
            return df, "No method applied."
    except Exception as e:
        return df, str(e)
    df_resampled = pd.concat([pd.DataFrame(X_res, columns=X_numeric.columns),
                               pd.Series(y_res, name=target_col)], axis=1)
    return df_resampled, "Success"
