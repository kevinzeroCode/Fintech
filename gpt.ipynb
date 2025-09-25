# -*- coding: utf-8 -*-
"""
T-Brain 2025 - E.SUN Alert Account Prediction
LightGBM 強化版（可直接輸出 submission_lightgbm.csv）

功能：
- 以 chunk 讀取 acct_transaction.csv，建立帳戶特徵
  * 出/入金額與筆數、平均/最大/標準差
  * 夜間/大額/自轉 比例
  * 通路(channel_type) 分布與比例
  * 每小時最大交易筆數（爆發度）
  * 對手多樣性與集中度（distinct、Top1、HHI）
  * 近因視窗 last7 / last30（出/入金筆數與金額；trend 7/30）
- 以帳戶最近交易日的中位數做時間切分（train/valid）
- 負樣本下採樣 + 多種子 bagging
- 依驗證集掃描機率閾值以最大化 F1
- 依 acct_predict.csv 順序輸出 submission_lightgbm.csv

需求：
pip install -U pandas numpy scikit-learn lightgbm
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import f1_score
import lightgbm as lgb

TXN_PATH = "acct_transaction.csv"
ALERT_PATH = "acct_alert.csv"
PRED_PATH = "acct_predict.csv"
SUBMISSION_OUT = "submission_lightgbm.csv"


# ============= Feature Builder (chunked) =============
def build_features(txn_path: str, chunksize: int = 1_000_000) -> pd.DataFrame:
    if not os.path.exists(txn_path):
        raise FileNotFoundError(f"Not found: {txn_path}")

    out_aggs, in_aggs = [], []
    out_pair_list, in_pair_list = [], []
    out_hour_peak_list, in_hour_peak_list = [], []
    out_day_list, in_day_list = [], []
    out_channel_list, in_channel_list = [], []

    dtypes = {
        "from_acct": "object",
        "from_acct_type": "object",
        "to_acct": "object",
        "to_acct_type": "object",
        "is_self_txn": "object",
        "txn_amt": "float64",
        "txn_date": "int64",
        "txn_time": "object",
        "currency_type": "object",
        "channel_type": "object",
    }

    for chunk in pd.read_csv(txn_path, dtype=dtypes, chunksize=chunksize):
        hr = pd.to_numeric(chunk["txn_time"].str.slice(0, 2), errors="coerce").fillna(-1).astype(int)
        is_night = ((hr >= 22) | (hr < 6)).astype(int)
        is_large = (chunk["txn_amt"] >= 100_000).astype(int)
        is_self  = (chunk["is_self_txn"].astype(str).str.upper() == "Y").astype(int)
        hour_bucket = (chunk["txn_date"].astype("int64") * 24 + hr).astype("int64")

        # 出金聚合
        g_out = pd.DataFrame({
            "from_acct": chunk["from_acct"],
            "txn_amt": chunk["txn_amt"],
            "txn_date": chunk["txn_date"],
            "is_night": is_night,
            "is_large": is_large,
            "is_self":  is_self
        }).groupby("from_acct").agg(
            out_cnt=("txn_amt", "size"),
            out_amt_sum=("txn_amt", "sum"),
            out_amt_sqsum=("txn_amt", lambda x: np.square(x).sum()),
            out_amt_max=("txn_amt", "max"),
            out_night_sum=("is_night", "sum"),
            out_large_sum=("is_large", "sum"),
            out_self_sum=("is_self", "sum"),
            out_min_date=("txn_date", "min"),
            out_max_date=("txn_date", "max"),
        )
        out_aggs.append(g_out)

        # 入金聚合
        g_in = pd.DataFrame({
            "to_acct": chunk["to_acct"],
            "txn_amt": chunk["txn_amt"],
            "txn_date": chunk["txn_date"],
            "is_night": is_night,
            "is_large": is_large,
        }).groupby("to_acct").agg(
            in_cnt=("txn_amt", "size"),
            in_amt_sum=("txn_amt", "sum"),
            in_amt_sqsum=("txn_amt", lambda x: np.square(x).sum()),
            in_amt_max=("txn_amt", "max"),
            in_night_sum=("is_night", "sum"),
            in_large_sum=("is_large", "sum"),
            in_min_date=("txn_date", "min"),
            in_max_date=("txn_date", "max"),
        )
        in_aggs.append(g_in)

        # 對手 pair（多樣性/HHI/top1）
        out_pair_list.append(chunk.groupby(["from_acct", "to_acct"]).size().rename("cnt").reset_index())
        in_pair_list.append(
            chunk.groupby(["to_acct", "from_acct"]).size().rename("cnt").reset_index()
                 .rename(columns={"to_acct": "to_acct_key", "from_acct": "from_peer"})
        )

        # 每小時爆發度
        out_hour_peak_list.append(
            pd.DataFrame({"from_acct": chunk["from_acct"], "hb": hour_bucket})
              .groupby(["from_acct", "hb"]).size().groupby(level=0).max()
              .rename("out_max_per_hour")
        )
        in_hour_peak_list.append(
            pd.DataFrame({"to_acct": chunk["to_acct"], "hb": hour_bucket})
              .groupby(["to_acct", "hb"]).size().groupby(level=0).max()
              .rename("in_max_per_hour")
        )

        # 日聚合（last7/30）
        out_day_list.append(
            pd.DataFrame({"from_acct": chunk["from_acct"], "txn_date": chunk["txn_date"], "txn_amt": chunk["txn_amt"]})
              .groupby(["from_acct", "txn_date"]).agg(out_day_cnt=("txn_amt", "size"),
                                                      out_day_amt=("txn_amt", "sum"))
        )
        in_day_list.append(
            pd.DataFrame({"to_acct": chunk["to_acct"], "txn_date": chunk["txn_date"], "txn_amt": chunk["txn_amt"]})
              .groupby(["to_acct", "txn_date"]).agg(in_day_cnt=("txn_amt", "size"),
                                                    in_day_amt=("txn_amt", "sum"))
        )

        # 通路分布
        out_ch = chunk.groupby(["from_acct", "channel_type"]).size().unstack(fill_value=0)
        out_ch.columns = [f"out_ch_{c}" for c in out_ch.columns]
        out_channel_list.append(out_ch)

        in_ch = chunk.groupby(["to_acct", "channel_type"]).size().unstack(fill_value=0)
        in_ch.columns = [f"in_ch_{c}" for c in in_ch.columns]
        in_channel_list.append(in_ch)

    # 合併基礎聚合
    out_df = pd.concat(out_aggs).groupby(level=0).agg({
        "out_cnt": "sum", "out_amt_sum": "sum", "out_amt_sqsum": "sum",
        "out_amt_max": "max", "out_night_sum": "sum", "out_large_sum": "sum",
        "out_self_sum": "sum", "out_min_date": "min", "out_max_date": "max",
    }).rename_axis("acct")
    in_df = pd.concat(in_aggs).groupby(level=0).agg({
        "in_cnt": "sum", "in_amt_sum": "sum", "in_amt_sqsum": "sum",
        "in_amt_max": "max", "in_night_sum": "sum", "in_large_sum": "sum",
        "in_min_date": "min", "in_max_date": "max",
    }).rename_axis("acct")

    feat = out_df.join(in_df, how="outer").fillna(0)

    # 每小時爆發度
    if out_hour_peak_list:
        feat = feat.join(pd.concat(out_hour_peak_list).groupby(level=0).max(), how="left")
    if in_hour_peak_list:
        feat = feat.join(pd.concat(in_hour_peak_list).groupby(level=0).max(), how="left")
    feat[["out_max_per_hour", "in_max_per_hour"]] = feat[["out_max_per_hour", "in_max_per_hour"]].fillna(0)

    # 通路分布（把所有 chunk 對齊後加總）
    if out_channel_list:
        feat = feat.join(pd.concat(out_channel_list).fillna(0).groupby(level=0).sum(), how="left")
    if in_channel_list:
        feat = feat.join(pd.concat(in_channel_list).fillna(0).groupby(level=0).sum(), how="left")
    feat = feat.fillna(0)

    # 對手多樣性 / HHI / top1
    if out_pair_list:
        op = pd.concat(out_pair_list).groupby(["from_acct", "to_acct"])["cnt"].sum().reset_index()
        out_distinct = op.groupby("from_acct")["to_acct"].nunique().rename("out_distinct_to")
        out_tot = op.groupby("from_acct")["cnt"].sum().rename("out_pair_tot")
        out_top1 = op.groupby("from_acct")["cnt"].max().rename("out_top1_cnt")
        op = op.merge(out_tot, on="from_acct")
        out_hhi = (op.assign(p=op["cnt"] / op["out_pair_tot"])
                     .assign(p2=lambda d: d["p"] ** 2)
                     .groupby("from_acct")["p2"].sum().rename("out_hhi"))
        feat = feat.join(out_distinct, how="left").join(out_top1, how="left").join(out_tot, how="left").join(out_hhi, how="left")
        feat["out_top1_ratio"] = feat["out_top1_cnt"] / feat["out_pair_tot"].replace(0, np.nan)

    if in_pair_list:
        ip = pd.concat(in_pair_list).rename(columns={"to_acct_key": "to_acct"})
        ip = ip.groupby(["to_acct", "from_peer"])["cnt"].sum().reset_index()
        in_distinct = ip.groupby("to_acct")["from_peer"].nunique().rename("in_distinct_from")
        in_tot = ip.groupby("to_acct")["cnt"].sum().rename("in_pair_tot")
        in_top1 = ip.groupby("to_acct")["cnt"].max().rename("in_top1_cnt")
        ip = ip.merge(in_tot, on="to_acct")
        in_hhi = (ip.assign(p=ip["cnt"] / ip["in_pair_tot"])
                    .assign(p2=lambda d: d["p"] ** 2)
                    .groupby("to_acct")["p2"].sum().rename("in_hhi"))
        feat = feat.join(in_distinct, how="left").join(in_top1, how="left").join(in_tot, how="left").join(in_hhi, how="left")
        feat["in_top1_ratio"] = feat["in_top1_cnt"] / feat["in_pair_tot"].replace(0, np.nan)

    # last7 / last30
    if out_day_list:
        od = pd.concat(out_day_list).groupby(level=[0, 1]).sum().reset_index().rename(columns={"from_acct": "acct"})
    else:
        od = pd.DataFrame(columns=["acct", "txn_date", "out_day_cnt", "out_day_amt"])
    if in_day_list:
        idf = pd.concat(in_day_list).groupby(level=[0, 1]).sum().reset_index().rename(columns={"to_acct": "acct"})
    else:
        idf = pd.DataFrame(columns=["acct", "txn_date", "in_day_cnt", "in_day_amt"])

    last_date = feat[["out_max_date", "in_max_date"]].replace(0, np.nan).max(axis=1).rename("last_date").reset_index()
    last_date = last_date.rename(columns={"index": "acct"})

    def win_sum(df: pd.DataFrame, cols: List[str], win: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        z = df.merge(last_date, on="acct", how="left")
        z["diff"] = z["last_date"] - z["txn_date"]
        z = z[(z["diff"] >= 0) & (z["diff"] <= win)]
        g = z.groupby("acct")[cols].sum()
        g.columns = [f"{c}_last{win}" for c in g.columns]
        return g

    for g in [win_sum(od, ["out_day_cnt", "out_day_amt"], 7),
              win_sum(od, ["out_day_cnt", "out_day_amt"], 30),
              win_sum(idf, ["in_day_cnt", "in_day_amt"], 7),
              win_sum(idf, ["in_day_cnt", "in_day_amt"], 30)]:
        if not g.empty:
            feat = feat.join(g, how="left")

    feat = feat.fillna(0)

    # 比例/平均/標準差/趨勢
    feat["out_amt_mean"] = feat["out_amt_sum"] / feat["out_cnt"].replace(0, np.nan)
    feat["in_amt_mean"] = feat["in_amt_sum"] / feat["in_cnt"].replace(0, np.nan)

    def safe_std(sum_sq, sum_, n):
        var = (sum_sq / n.replace(0, np.nan)) - (sum_ / n.replace(0, np.nan)) ** 2
        return np.sqrt(np.clip(var, 0, None))

    feat["out_amt_std"] = safe_std(feat["out_amt_sqsum"], feat["out_amt_sum"], feat["out_cnt"]).fillna(0)
    feat["in_amt_std"] = safe_std(feat["in_amt_sqsum"], feat["in_amt_sum"], feat["in_cnt"]).fillna(0)

    feat["out_night_ratio"] = feat["out_night_sum"] / feat["out_cnt"].replace(0, np.nan)
    feat["in_night_ratio"] = feat["in_night_sum"] / feat["in_cnt"].replace(0, np.nan)
    feat["out_large_ratio"] = feat["out_large_sum"] / feat["out_cnt"].replace(0, np.nan)
    feat["in_large_ratio"] = feat["in_large_sum"] / feat["in_cnt"].replace(0, np.nan)
    feat["self_ratio"] = feat["out_self_sum"] / feat["out_cnt"].replace(0, np.nan)

    feat["out_hour_peak_ratio"] = feat["out_max_per_hour"] / feat["out_cnt"].replace(0, np.nan)
    feat["in_hour_peak_ratio"] = feat["in_max_per_hour"] / feat["in_cnt"].replace(0, np.nan)

    out_cnt = feat["out_cnt"].replace(0, np.nan)
    in_cnt = feat["in_cnt"].replace(0, np.nan)
    for c in [c for c in feat.columns if c.startswith("out_ch_")]:
        feat[c + "_ratio"] = feat[c] / out_cnt
    for c in [c for c in feat.columns if c.startswith("in_ch_")]:
        feat[c + "_ratio"] = feat[c] / in_cnt

    feat["tot_cnt"] = feat["out_cnt"] + feat["in_cnt"]
    feat["tot_amt_sum"] = feat["out_amt_sum"] + feat["in_amt_sum"]
    feat["tot_amt_max"] = feat[["out_amt_max", "in_amt_max"]].max(axis=1)

    min_dates = feat[["out_min_date", "in_min_date"]].replace(0, np.nan).min(axis=1)
    max_dates = feat[["out_max_date", "in_max_date"]].replace(0, np.nan).max(axis=1)
    feat["active_span"] = (max_dates - min_dates).fillna(0)

    # 清理中繼欄位
    drop_cols = [c for c in feat.columns if c.endswith("_sqsum")] + ["out_min_date", "in_min_date"]
    feat = feat.drop(columns=[c for c in drop_cols if c in feat.columns]).fillna(0)

    feat = feat.reset_index().rename(columns={"index": "acct"})
    return feat


# ============= Labels / Split / Threshold =============
def load_labels(alert_path: str) -> pd.DataFrame:
    df = pd.read_csv(alert_path)
    pos = df[["acct"]].drop_duplicates().copy()
    pos["label"] = 1
    return pos

def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recency = df[["acct", "out_max_date", "in_max_date"]].copy()
    recency["max_d"] = recency[["out_max_date", "in_max_date"]].max(axis=1)
    m = recency["max_d"].median()
    valid_ids = set(recency[recency["max_d"] > m]["acct"])
    tr = df[~df["acct"].isin(valid_ids)]
    va = df[df["acct"].isin(valid_ids)]
    return tr, va

def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


# ============= Train + Downsample + Seed Bagging =============
def train_and_predict(feat: pd.DataFrame,
                      labels: pd.DataFrame,
                      pred_accts: pd.DataFrame,
                      neg_pos_ratio: int = 20,
                      seeds: List[int] = [42, 2025, 7, 99, 1234]) -> Tuple[np.ndarray, float]:
    df = feat.merge(labels, on="acct", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    tr, va = time_split(df)
    X_tr_full = tr.drop(columns=["acct", "label"])
    y_tr_full = tr["label"].values
    X_va = va.drop(columns=["acct", "label"])
    y_va = va["label"].values

    # 下採樣：保留全體正樣本 + 隨機抽取負樣本至某倍率
    pos_idx = np.where(y_tr_full == 1)[0]
    neg_idx = np.where(y_tr_full == 0)[0]
    n_pos = len(pos_idx)
    n_neg_keep = min(len(neg_idx), n_pos * neg_pos_ratio if n_pos > 0 else len(neg_idx))
    rng = np.random.default_rng(123)
    keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False) if len(neg_idx) > n_neg_keep else neg_idx
    keep_idx = np.concatenate([pos_idx, keep_neg])

    X_tr = X_tr_full.iloc[keep_idx]
    y_tr = y_tr_full[keep_idx]

    # 測試特徵（依 acct_predict 順序）
    pred_feat = pred_accts.merge(feat, on="acct", how="left").fillna(0)
    X_test = pred_feat.drop(columns=["acct"])

    # 多種子 bagging
    va_prob_sum = np.zeros(len(X_va), dtype=float)
    te_prob_sum = np.zeros(len(X_test), dtype=float)

    for seed in seeds:
        pos = y_tr.sum(); neg = len(y_tr) - pos
        spw = (neg / pos) if pos > 0 else 1.0

        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=127,
            min_data_in_leaf=60,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            reg_alpha=0.5,
            reg_lambda=1.0,
            metric="binary_logloss",
            scale_pos_weight=spw,
            seed=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            verbose=-1,
        )
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        callbacks = [lgb.early_stopping(150), lgb.log_evaluation(200)]
        model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid], callbacks=callbacks)

        va_prob_sum += model.predict(X_va, num_iteration=model.best_iteration)
        te_prob_sum += model.predict(X_test, num_iteration=model.best_iteration)

    va_prob = va_prob_sum / len(seeds)
    te_prob = te_prob_sum / len(seeds)

    t, _ = best_threshold(y_va, va_prob)
    te_label = (te_prob >= t).astype(int)
    return te_label, t


# ============= Main =============
def main():
    print("Building features (this may take a while)...")
    feat = build_features(TXN_PATH)
    print(f"Accounts with features: {len(feat):,}")

    labels = load_labels(ALERT_PATH)
    pred_accts = pd.read_csv(PRED_PATH)[["acct"]]

    print("Training & predicting ...")
    te_label, threshold = train_and_predict(
        feat, labels, pred_accts,
        neg_pos_ratio=20,                           # 調整負樣本倍率（10~50之間試驗）
        seeds=[42, 2025, 7, 99, 1234]               # 種子數可多可少；多會更穩但更慢
    )

    submission = pd.DataFrame({"acct": pred_accts["acct"], "label": te_label})
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"Saved -> {SUBMISSION_OUT}")
    print(f"Chosen threshold = {threshold:.3f}")

if __name__ == "__main__":
    main()
