import pandas as pd
import numpy as np
from fpdf import FPDF
import io
import tempfile
import os


class ProfileXReport(FPDF):
    def __init__(self, dataset_name="Dataset"):
        super().__init__()
        self.dataset_name = dataset_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(15, 23, 42)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  ProfileX Report — {self.dataset_name}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 64, 175)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 9)
        self.cell(55, 6, str(key) + ":", ln=False)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, str(value), ln=True)

    def small_text(self, text):
        self.set_font("Helvetica", "", 8)
        self.multi_cell(0, 5, str(text))
        self.ln(1)


def generate_pdf_report(df_original, df_processed, log, ml_results=None, dataset_name="Dataset"):
    pdf = ProfileXReport(dataset_name=dataset_name)
    pdf.add_page()

    # ── Section 1: Overview ─────────────────────────────────
    pdf.section_title("1. Dataset Overview")
    pdf.key_value("Original Shape", f"{df_original.shape[0]} rows × {df_original.shape[1]} columns")
    pdf.key_value("Processed Shape", f"{df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
    pdf.key_value("Rows Removed", df_original.shape[0] - df_processed.shape[0])
    pdf.key_value("Columns Changed", df_processed.shape[1] - df_original.shape[1])
    pdf.key_value("Missing Values (original)", int(df_original.isna().sum().sum()))
    pdf.key_value("Missing Values (processed)", int(df_processed.isna().sum().sum()))
    mem_kb = round(df_processed.memory_usage(deep=True).sum() / 1024, 2)
    pdf.key_value("Memory (processed)", f"{mem_kb} KB")
    pdf.ln(3)

    # ── Section 2: Column Details ────────────────────────────
    pdf.section_title("2. Column Details (Processed)")
    pdf.set_font("Helvetica", "B", 8)
    col_widths = [55, 30, 30, 30, 35]
    headers = ["Column", "Non-Null", "Missing", "Missing %", "Dtype"]
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for col in df_processed.columns:
        non_null = df_processed[col].notnull().sum()
        missing = df_processed[col].isnull().sum()
        miss_pct = round(missing / len(df_processed) * 100, 1)
        dtype = str(df_processed[col].dtype)
        row = [col[:22], str(non_null), str(missing), f"{miss_pct}%", dtype]
        for val, w in zip(row, col_widths):
            pdf.cell(w, 5, val, border=1)
        pdf.ln()
    pdf.ln(4)

    # ── Section 3: Summary Statistics ───────────────────────
    pdf.section_title("3. Summary Statistics (Numeric, Processed)")
    num_cols = df_processed.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        stats = df_processed[num_cols].describe().T.round(3)
        stat_cols = ["mean", "std", "min", "50%", "max"]
        avail = [c for c in stat_cols if c in stats.columns]
        header_labels = ["Column"] + avail
        widths = [40] + [25] * len(avail)
        pdf.set_font("Helvetica", "B", 7)
        for h, w in zip(header_labels, widths):
            pdf.cell(w, 6, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for col in stats.index[:25]:
            row_vals = [col[:18]] + [str(stats.loc[col, c]) for c in avail]
            for val, w in zip(row_vals, widths):
                pdf.cell(w, 5, str(val)[:14], border=1)
            pdf.ln()
        if len(stats) > 25:
            pdf.small_text(f"... and {len(stats) - 25} more columns.")
    else:
        pdf.small_text("No numeric columns available.")
    pdf.ln(4)

    # ── Section 4: Preprocessing Log ────────────────────────
    pdf.section_title("4. Preprocessing Log")
    for entry in log:
        pdf.small_text(f"• {entry}")
    pdf.ln(3)

    # ── Section 5: ML Results ────────────────────────────────
    if ml_results:
        pdf.add_page()
        pdf.section_title("5. ML Model Results")
        for model_name, res in ml_results.items():
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, f"  Model: {model_name}", ln=True)
            pdf.set_font("Helvetica", "", 8)
            if "error" in res:
                pdf.small_text(f"  Error: {res['error']}")
            else:
                for k, v in res["metrics"].items():
                    if k in ("Report", "Confusion Matrix", "Feature Importance"):
                        continue
                    pdf.key_value(f"  {k}", v)
                report = res["metrics"].get("Report")
                if report:
                    pdf.set_font("Courier", "", 7)
                    pdf.multi_cell(0, 4, report)
            pdf.ln(2)

    # Return as bytes
    out = io.BytesIO()
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)
