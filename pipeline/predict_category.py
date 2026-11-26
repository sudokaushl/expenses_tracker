"""
Predict category / subcategory for every transaction from its statement's labelled table.

Features come from expenses_normalized (the same columns exist for new, unlabelled rows):
tokens, trans_mode, trans_type, amount and the normalized source. Low-confidence rows can
optionally be resolved by the local LLM, restricted to the classifier's top candidates.

Usage:
    python predict_category.py              # evaluate, train, write expenses_categorized
    python predict_category.py --llm        # also let the LLM pick among top candidates
    python predict_category.py --no-write   # evaluate and report only
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from . import config
from .utils import LoggerFactory, create_duckdb_connection, get_log_file_path, set_logger

def clean_sql(col: str) -> str:
    return f"regexp_replace(lower({col}), '[^a-z0-9]', '', 'g')"


# ---------- data ----------

def load_data(con, statement: str = "account") -> pd.DataFrame:
    """All normalized rows of a statement type, with manual labels attached where a match exists."""
    n = f"{config.SCHEMA_NAME}.{config.tables(statement).normalized}"
    m = f"{config.SCHEMA_NAME}.{config.labelled_table(statement)}"
    rows = con.query(f"""
        SELECT trans_id, trans_dt, particulars, trans_amt, trans_type, trans_mode,
               normalized_trans_src, CAST(tokenized AS VARCHAR) AS txt
        FROM {n}
    """).to_df()

    # Manual ids use another hash and its text has no spaces, so match on date + amount + cleaned text.
    labels = con.query(f"""
        SELECT n.trans_id, m.normalized_trans_src AS labelled_trans_src,
               m.category, m.subcategory
        FROM {m} m JOIN {n} n
          ON CAST(m.trans_dt AS DATE) = CAST(n.trans_dt AS DATE)
                 AND TRY_CAST(REPLACE(CAST(m.trans_amt AS VARCHAR), ',', '') AS DOUBLE)
                         = TRY_CAST(REPLACE(CAST(n.trans_amt AS VARCHAR), ',', '') AS DOUBLE)
         AND {clean_sql('m.particulars')} = {clean_sql('n.particulars')}
        WHERE m.category IS NOT NULL AND m.subcategory IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY m.trans_id ORDER BY n.trans_id) = 1
    """).to_df().drop_duplicates("trans_id")

    return rows.merge(labels, on="trans_id", how="left")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["txt"] = out["txt"].fillna("").str.lower().apply(lambda s: " ".join(re.findall(r"[a-z0-9]+", s)))
    out["src"] = (
        out["normalized_trans_src"].fillna("none")
        .str.replace(r"^\[AI\]\s*", "", regex=True)  # LLM-tagged sources count as the source itself
        .str.lower().str.replace(r"\W+", "_", regex=True)
    )
    return out


# ---------- model ----------

class Featurizer:
    def __init__(self):
        self.tv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.oh = OneHotEncoder(handle_unknown="ignore")

    @staticmethod
    def _cats(df):
        return df[["trans_mode", "trans_type", "src"]].fillna("na").astype(str)

    @staticmethod
    def _amt(df):
        return csr_matrix(np.log1p(df["trans_amt"].fillna(0).abs().values)[:, None] / 10)

    def fit_transform(self, df):
        return hstack([self.tv.fit_transform(df["txt"]), self.oh.fit_transform(self._cats(df)), self._amt(df)]).tocsr()

    def transform(self, df):
        return hstack([self.tv.transform(df["txt"]), self.oh.transform(self._cats(df)), self._amt(df)]).tocsr()


def fit_models(train: pd.DataFrame):
    feat = Featurizer()
    X = feat.fit_transform(train)
    cat = LogisticRegression(C=20, max_iter=2000).fit(X, train["category"])
    joint = LogisticRegression(C=20, max_iter=2000).fit(X, train["category"] + " | " + train["subcategory"])
    return feat, cat, joint


def predict(models, df: pd.DataFrame):
    """Category from its own model; subcategory is the best joint label inside that category."""
    feat, cat, joint = models
    X = feat.transform(df)

    pc = cat.predict_proba(X)
    cats = cat.classes_[pc.argmax(1)]

    pj = joint.predict_proba(X)
    joint_cat = np.array([c.split(" | ", 1)[0] for c in joint.classes_])
    joint_sub = np.array([c.split(" | ", 1)[1] for c in joint.classes_])
    masked = np.where(joint_cat[None, :] == cats[:, None], pj, 0.0)
    best = masked.argmax(1)
    sub_conf = masked[np.arange(len(df)), best] / np.maximum(masked.sum(1), 1e-9)

    pred = pd.DataFrame({
        "category": cats,
        "subcategory": joint_sub[best],
        "category_conf": pc.max(1),
        "subcategory_conf": sub_conf,
    })
    top5 = np.argsort(-pj, axis=1)[:, :5]
    return pred, top5, joint.classes_


# ---------- evaluation ----------

def evaluate(lab: pd.DataFrame, n_splits: int = 5):
    lab = lab.reset_index(drop=True)
    parts = []
    for tr_i, te_i in KFold(n_splits, shuffle=True, random_state=0).split(lab):
        pred, _, _ = predict(fit_models(lab.iloc[tr_i]), lab.iloc[te_i])
        pred.index = te_i
        parts.append(pred)
    p = pd.concat(parts).sort_index()

    ok_cat = p["category"].values == lab["category"].values
    ok_both = ok_cat & (p["subcategory"].values == lab["subcategory"].values)
    cat_precision, cat_recall, cat_f1, _ = precision_recall_fscore_support(
        lab["category"], p["category"], average="macro", zero_division=0
    )
    true_joint = lab["category"].astype(str) + " | " + lab["subcategory"].astype(str)
    predicted_joint = p["category"].astype(str) + " | " + p["subcategory"].astype(str)
    joint_precision, joint_recall, joint_f1, _ = precision_recall_fscore_support(
        true_joint, predicted_joint, average="macro", zero_division=0
    )
    calibration_brier = brier_score_loss(ok_cat.astype(int), p["category_conf"])
    print(f"\n{n_splits}-fold cross-validation on {len(lab)} labelled rows")
    print(f"  category accuracy:               {ok_cat.mean():.1%}")
    print(f"  category macro precision/recall/F1: {cat_precision:.1%} / {cat_recall:.1%} / {cat_f1:.1%}")
    print(f"  category + subcategory accuracy: {ok_both.mean():.1%}")
    print(f"  joint macro precision/recall/F1:    {joint_precision:.1%} / {joint_recall:.1%} / {joint_f1:.1%}")
    print(f"  category confidence Brier score:    {calibration_brier:.4f} (lower is better)")

    accepted = ((p["category_conf"] >= config.CATEGORY_CONF_THRESHOLD)
                & (p["subcategory_conf"] >= config.SUBCATEGORY_CONF_THRESHOLD)).values
    accepted_accuracy = f"{ok_both[accepted].mean():.1%}" if accepted.any() else "n/a (no accepted rows)"
    print(f"  auto-accepted (cat>={config.CATEGORY_CONF_THRESHOLD}, sub>={config.SUBCATEGORY_CONF_THRESHOLD}): "
          f"{accepted.mean():.1%} of rows, category+subcategory accuracy {accepted_accuracy}")

    per = pd.DataFrame({"category": lab["category"], "ok": ok_cat}).groupby("category")["ok"].agg(["mean", "size"])
    weak = per[per["size"] >= 15].sort_values("mean").head(5)
    print("  weakest categories (>=15 rows):", ", ".join(f"{i} {r['mean']:.0%} (n={int(r['size'])})" for i, r in weak.iterrows()))


# ---------- optional LLM step ----------

def llm_pick(df: pd.DataFrame, top5: np.ndarray, classes, batch_size: int = 8) -> dict:
    """Ask the LLM to choose among the classifier's top candidates; anything else is discarded."""
    from langchain_openai import ChatOpenAI
    from tqdm import tqdm

    llm = ChatOpenAI(
        base_url=config.LLM_CONFIG.LOCAL_LLM_BASE_URL,
        api_key=config.LLM_CONFIG.LOCAL_LLM_API_KEY,
        model=config.LLM_CONFIG.LOCAL_LLM_MODEL,
        temperature=config.LLM_CONFIG.TEMPERATURE,
        max_tokens=200,
        timeout=120,
        max_retries=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    answers = {}
    pbar = tqdm(range(0, len(df), batch_size), desc="LLM category match", unit="batch")
    for start in pbar:
        idx = range(start, min(start + batch_size, len(df)))
        prompt = (
            "For each numbered transaction choose the best 'category | subcategory' from its candidates. "
            "Output one line 'n|k' where k is the candidate number, or 'n|0' if none fits. "
            "Do not guess. No other text.\n\nTransactions:"
        )
        for n, i in enumerate(idx, start=1):
            r = df.iloc[i]
            cands = "; ".join(f"{k}) {classes[j]}" for k, j in enumerate(top5[i], start=1))
            prompt += f"\n{n}. {str(r['particulars'])[:90]} | {r['trans_type']} {abs(r['trans_amt']):.0f} | {cands}"
        try:
            text = llm.invoke(prompt).content or ""
            for line in text.splitlines():
                m = re.match(r"\s*(\d+)\s*\|\s*(\d+)", line)
                if m and 1 <= int(m[1]) <= len(idx) and 1 <= int(m[2]) <= top5.shape[1]:
                    i = idx[int(m[1]) - 1]
                    answers[i] = classes[top5[i][int(m[2]) - 1]].split(" | ", 1)
        except Exception as e:
            tqdm.write(f"  batch failed: {e}")
        pbar.set_postfix(answered=len(answers))
    return answers


# ---------- main ----------

def main(use_llm: bool, write: bool, skip_eval: bool, con=None, statement: str = "account"):
    con = con or create_duckdb_connection(config.DATABASE_PATH)
    df = prep(load_data(con, statement))
    lab = df[df["category"].notna()].reset_index(drop=True)
    label_table = config.labelled_table(statement)
    print(f"{len(df)} {statement} transactions; training on {len(lab)} manually labelled rows from {label_table}")
    if len(lab) < 10:
        raise SystemExit(f"Too few manually labelled rows matched; check {label_table} and {config.tables(statement).normalized}.")

    if not skip_eval:
        evaluate(lab)

    pred, top5, classes = predict(fit_models(lab), df)
    out = pd.concat([
        df[["trans_id", "trans_dt", "particulars", "trans_amt", "trans_type", "trans_mode", "normalized_trans_src"]],
        pred,
    ], axis=1)
    out["label_source"] = "model"

    manual = df["category"].notna().values
    out.loc[manual, "normalized_trans_src"] = df.loc[manual, "labelled_trans_src"].values
    out.loc[manual, ["category", "subcategory"]] = df.loc[manual, ["category", "subcategory"]].values
    out.loc[manual, ["category_conf", "subcategory_conf"]] = 1.0
    out.loc[manual, "label_source"] = "manual"

    low = (~manual) & ((out["category_conf"] < config.CATEGORY_CONF_THRESHOLD)
                       | (out["subcategory_conf"] < config.SUBCATEGORY_CONF_THRESHOLD)).values
    out.loc[low, "label_source"] = "low_confidence"

    if use_llm and low.any():
        pos = np.flatnonzero(low)
        for i, (cat, sub) in llm_pick(df.iloc[pos], top5[pos], classes).items():
            out.loc[pos[i], ["category", "subcategory", "label_source"]] = [f"[AI] {cat}", sub, "llm"]

    print("\nlabel_source counts:\n" + out["label_source"].value_counts().to_string())

    if write:
        con.register("out_df", out)
        table = f"{config.SCHEMA_NAME}.{config.tables(statement).categorized}"
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM out_df")
        print(f"💾 Saved {len(out)} rows to {table}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="let the local LLM resolve low-confidence rows")
    ap.add_argument("--no-write", action="store_true", help="evaluate and report without writing the table")
    ap.add_argument("--skip-eval", action="store_true", help="skip cross-validation")
    ap.add_argument("--statement", choices=config.STATEMENTS, default="account")
    args = ap.parse_args()

    script_name = os.path.basename(__file__).replace(".py", "")
    set_logger(LoggerFactory.get_logger(get_log_file_path(script_name), config.LOG_LEVEL))
    main(use_llm=args.llm, write=not args.no_write, skip_eval=args.skip_eval, statement=args.statement)
