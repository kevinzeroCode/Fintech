# -*- coding: utf-8 -*-
"""
E.SUN Alert – GPU-Boosted High-Score Ensemble (Max Search, CUDA)
輸出：
- submission_ensemble_threshold.csv
- submission_ensemble_topk.csv

重點：GPU=True、多視角時間切分、下採樣倍率網格、LGB+XGB(+CatBoost) 多種子 bagging、
      logit 加權搜尋 + 閾值最佳化 + Top-K 校準，並對多組結果以 OOF F1 加權堆疊。
需求：pandas numpy scikit-learn lightgbm xgboost catboost pyarrow
"""

import os, time, math
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb

# ========= 路徑 =========
BASE = "."
TXN_PATH       = os.path.join(BASE, "acct_transaction.csv")
ALERT_PATH     = os.path.join(BASE, "acct_alert.csv")
PRED_PATH      = os.path.join(BASE, "acct_predict.csv")
FEAT_CACHE     = os.path.join(BASE, "features_speed.parquet")
SUB_THR        = os.path.join(BASE, "submission_ensemble_threshold.csv")
SUB_TOPK       = os.path.join(BASE, "submission_ensemble_topk.csv")

# ========= 控制旋鈕（已開最大檔）=========
USE_GPU          = True    # <<< 已開啟 GPU
GPU_DEV          = "0"     # 多卡時改成 "0,1" 等
CHUNKSIZE        = 2_000_000   # 視記憶體調整；小機器可降回 1_000_000
NEG_POS_CAND     = [20, 25, 30, 35, 40, 45, 50]   # 下採樣倍率搜尋
QS_CAND          = [0.45, 0.50, 0.55, 0.60, 0.65] # 時間切分量化點
SEEDS_LGB        = [42, 7, 2025, 99]              # 多種子 bagging
SEEDS_XGB        = [2025, 7, 42, 99]
SEEDS_CAT        = [7, 42, 2025, 99]
LGB_ROUNDS       = 6000
XGB_ROUNDS       = 6000
EARLY_STOP       = 300
USE_BIDIR        = True
USE_DECAY        = True

# <<< 新增：綁定 CUDA 裝置（XGB/CatBoost 會讀這個），並讓容器/驅動對齊 >>>
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"]   = GPU_DEV.split(",")[0]
    os.environ["NVIDIA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_DEVICE_ORDER"]      = "PCI_BUS_ID"
    print(f"[env] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# ========= 小工具 =========
def tic(): return time.perf_counter()
def toc(t0, msg): print(f"[time] {msg}: {time.perf_counter()-t0:.1f}s")

def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def best_threshold(y_true, y_prob):
    grid = np.concatenate([np.linspace(0.02,0.5,97), np.linspace(0.5,0.92,43)])
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_t = float(f1), float(t)
    # 局部微調
    local = np.linspace(max(0.02,best_t-0.04), min(0.92,best_t+0.04), 81)
    for t in local:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_t = float(f1), float(t)
    return best_t, best_f1

def entropy_from_cols(df, prefix, name):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols: return pd.Series(0, index=df.index, name=name)
    m = df[cols].astype(float)
    s = m.sum(axis=1).replace(0, np.nan)
    p = m.div(s, axis=0)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    return ent.fillna(0).rename(name)

# ========= 標籤 / 切分 / 下採樣 =========
def load_labels(alert_path):
    df = pd.read_csv(alert_path)
    pos = df[["acct"]].drop_duplicates().copy()
    pos["label"] = 1
    return pos

def time_split(df: pd.DataFrame, q: float):
    recency = df[["acct","out_max_date","in_max_date"]].copy()
    recency["max_d"] = recency[["out_max_date","in_max_date"]].max(axis=1)
    thr = recency["max_d"].quantile(q)
    valid_ids = set(recency[recency["max_d"] > thr]["acct"])
    tr = df[~df["acct"].isin(valid_ids)]
    va = df[df["acct"].isin(valid_ids)]
    return tr, va

def downsample(X, y, ratio, seed=123):
    pos_idx = np.where(y==1)[0]; neg_idx = np.where(y==0)[0]
    n_pos = len(pos_idx)
    n_neg_keep = min(len(neg_idx), n_pos*ratio if n_pos>0 else len(neg_idx))
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False) if len(neg_idx)>n_neg_keep else neg_idx
    keep_idx = np.concatenate([pos_idx, keep_neg])
    return X.iloc[keep_idx], y[keep_idx]

# ========= 建特徵（速度友善 + 強化） =========
def build_features(txn_path, chunksize=1_000_000, enable_bidir=True, enable_decay=True):
    dtypes = {
        "from_acct":"object","from_acct_type":"object","to_acct":"object","to_acct_type":"object",
        "is_self_txn":"object","txn_amt":"float64","txn_date":"int64","txn_time":"object",
        "currency_type":"object","channel_type":"object",
    }
    out_aggs, in_aggs = [], []
    out_pair_list, in_pair_list = [], []
    out_hour_peak_list, in_hour_peak_list = [], []
    out_day_list, in_day_list = [], []
    out_channel_list, in_channel_list = [], []
    pair_dir_flags = []

    t0 = tic()
    for i,chunk in enumerate(pd.read_csv(txn_path, dtype=dtypes, chunksize=chunksize)):
        if i%5==0 and i>0: print(f"[build] rows ~{i*chunksize:,} in {time.perf_counter()-t0:.1f}s")

        hr       = pd.to_numeric(chunk["txn_time"].str.slice(0,2), errors="coerce").fillna(-1).astype(int)
        is_night = ((hr>=22)|(hr<6)).astype(int)
        is_large = (chunk["txn_amt"]>=100_000).astype(int)
        is_self  = (chunk["is_self_txn"].astype(str).str.upper()=="Y").astype(int)
        t_morn   = ((hr>=6)&(hr<12)).astype(int)
        t_noon   = ((hr>=12)&(hr<18)).astype(int)
        t_even   = ((hr>=18)&(hr<22)).astype(int)
        hour_bucket = (chunk["txn_date"].astype("int64")*24+hr).astype("int64")

        g_out = pd.DataFrame({
            "from_acct":chunk["from_acct"],"txn_amt":chunk["txn_amt"],"txn_date":chunk["txn_date"],
            "is_night":is_night,"is_large":is_large,"is_self":is_self,"t_m":t_morn,"t_n":t_noon,"t_e":t_even
        }).groupby("from_acct").agg(
            out_cnt=("txn_amt","size"), out_amt_sum=("txn_amt","sum"),
            out_amt_sqsum=("txn_amt", lambda x: np.square(x).sum()), out_amt_max=("txn_amt","max"),
            out_night_sum=("is_night","sum"), out_large_sum=("is_large","sum"), out_self_sum=("is_self","sum"),
            out_morn_sum=("t_m","sum"), out_noon_sum=("t_n","sum"), out_even_sum=("t_e","sum"),
            out_min_date=("txn_date","min"), out_max_date=("txn_date","max"),
        ); out_aggs.append(g_out)

        g_in = pd.DataFrame({
            "to_acct":chunk["to_acct"],"txn_amt":chunk["txn_amt"],"txn_date":chunk["txn_date"],
            "is_night":is_night,"is_large":is_large,"t_m":t_morn,"t_n":t_noon,"t_e":t_even
        }).groupby("to_acct").agg(
            in_cnt=("txn_amt","size"), in_amt_sum=("txn_amt","sum"),
            in_amt_sqsum=("txn_amt", lambda x: np.square(x).sum()), in_amt_max=("txn_amt","max"),
            in_night_sum=("is_night","sum"), in_large_sum=("is_large","sum"),
            in_morn_sum=("t_m","sum"), in_noon_sum=("t_n","sum"), in_even_sum=("t_e","sum"),
            in_min_date=("txn_date","min"), in_max_date=("txn_date","max"),
        ); in_aggs.append(g_in)

        out_pair_list.append(chunk.groupby(["from_acct","to_acct"]).size().rename("cnt").reset_index())
        in_pair_list.append(
            chunk.groupby(["to_acct","from_acct"]).size().rename("cnt").reset_index()
                 .rename(columns={"to_acct":"to_acct_key","from_acct":"from_peer"})
        )

        if enable_bidir:
            a = chunk["from_acct"].astype(str).values; b = chunk["to_acct"].astype(str).values
            lo = np.where(a<=b,a,b); hi = np.where(a<=b,b,a); dir_flag = (a<=b).astype(int)
            flags = pd.DataFrame({"lo":lo,"hi":hi,"dir":dir_flag}).groupby(["lo","hi"])["dir"]\
                     .agg(dir_min="min", dir_max="max").reset_index()
            pair_dir_flags.append(flags)

        out_hour_peak_list.append(
            pd.DataFrame({"from_acct":chunk["from_acct"],"hb":hour_bucket}).groupby(["from_acct","hb"]).size()
              .groupby(level=0).max().rename("out_max_per_hour"))
        in_hour_peak_list.append(
            pd.DataFrame({"to_acct":chunk["to_acct"],"hb":hour_bucket}).groupby(["to_acct","hb"]).size()
              .groupby(level=0).max().rename("in_max_per_hour"))

        out_day_list.append(
            pd.DataFrame({"from_acct":chunk["from_acct"],"txn_date":chunk["txn_date"],"txn_amt":chunk["txn_amt"]})
              .groupby(["from_acct","txn_date"]).agg(out_day_cnt=("txn_amt","size"), out_day_amt=("txn_amt","sum"))
        ); in_day_list.append(
            pd.DataFrame({"to_acct":chunk["to_acct"],"txn_date":chunk["txn_date"],"txn_amt":chunk["txn_amt"]})
              .groupby(["to_acct","txn_date"]).agg(in_day_cnt=("txn_amt","size"), in_day_amt=("txn_amt","sum"))
        )

        out_ch = chunk.groupby(["from_acct","channel_type"]).size().unstack(fill_value=0)
        out_ch.columns = [f"out_ch_{c}" for c in out_ch.columns]; out_channel_list.append(out_ch)
        in_ch  = chunk.groupby(["to_acct","channel_type"]).size().unstack(fill_value=0)
        in_ch.columns  = [f"in_ch_{c}" for c in in_ch.columns];  in_channel_list.append(in_ch)

    out_df = pd.concat(out_aggs).groupby(level=0).agg({
        "out_cnt":"sum","out_amt_sum":"sum","out_amt_sqsum":"sum","out_amt_max":"max",
        "out_night_sum":"sum","out_large_sum":"sum","out_self_sum":"sum",
        "out_morn_sum":"sum","out_noon_sum":"sum","out_even_sum":"sum",
        "out_min_date":"min","out_max_date":"max",
    }).rename_axis("acct")
    in_df = pd.concat(in_aggs).groupby(level=0).agg({
        "in_cnt":"sum","in_amt_sum":"sum","in_amt_sqsum":"sum","in_amt_max":"max",
        "in_night_sum":"sum","in_large_sum":"sum",
        "in_morn_sum":"sum","in_noon_sum":"sum","in_even_sum":"sum",
        "in_min_date":"min","in_max_date":"max",
    }).rename_axis("acct")
    feat = out_df.join(in_df, how="outer").fillna(0)

    if out_hour_peak_list: feat = feat.join(pd.concat(out_hour_peak_list).groupby(level=0).max(), how="left")
    if in_hour_peak_list:  feat = feat.join(pd.concat(in_hour_peak_list).groupby(level=0).max(),  how="left")
    feat[["out_max_per_hour","in_max_per_hour"]] = feat[["out_max_per_hour","in_max_per_hour"]].fillna(0)

    if out_channel_list: feat = feat.join(pd.concat(out_channel_list).fillna(0).groupby(level=0).sum(), how="left")
    if in_channel_list:  feat = feat.join(pd.concat(in_channel_list).fillna(0).groupby(level=0).sum(),  how="left")

    if out_pair_list:
        op = pd.concat(out_pair_list).groupby(["from_acct","to_acct"])["cnt"].sum().reset_index()
        out_distinct = op.groupby("from_acct")["to_acct"].nunique().rename("out_distinct_to")
        out_tot      = op.groupby("from_acct")["cnt"].sum().rename("out_pair_tot")
        out_top1     = op.groupby("from_acct")["cnt"].max().rename("out_top1_cnt")
        out_top3     = op.groupby("from_acct")["cnt"].apply(lambda s: s.nlargest(3).sum()).rename("out_top3_cnt")
        op = op.merge(out_tot, on="from_acct")
        out_hhi = (op.assign(p=op["cnt"]/op["out_pair_tot"]).assign(p2=lambda d:d["p"]**2)
                     .groupby("from_acct")["p2"].sum().rename("out_hhi"))
        feat = feat.join([out_distinct,out_top1,out_top3,out_tot,out_hhi], how="left")
        feat["out_top1_ratio"] = feat["out_top1_cnt"]/feat["out_pair_tot"].replace(0,np.nan)
        feat["out_top3_ratio"] = feat["out_top3_cnt"]/feat["out_pair_tot"].replace(0,np.nan)

    if in_pair_list:
        ip = pd.concat(in_pair_list).rename(columns={"to_acct_key":"to_acct"})\
             .groupby(["to_acct","from_peer"])["cnt"].sum().reset_index()
        in_distinct = ip.groupby("to_acct")["from_peer"].nunique().rename("in_distinct_from")
        in_tot      = ip.groupby("to_acct")["cnt"].sum().rename("in_pair_tot")
        in_top1     = ip.groupby("to_acct")["cnt"].max().rename("in_top1_cnt")
        in_top3     = ip.groupby("to_acct")["cnt"].apply(lambda s: s.nlargest(3).sum()).rename("in_top3_cnt")
        ip = ip.merge(in_tot, on="to_acct")
        in_hhi = (ip.assign(p=ip["cnt"]/ip["in_pair_tot"]).assign(p2=lambda d:d["p"]**2)
                    .groupby("to_acct")["p2"].sum().rename("in_hhi"))
        feat = feat.join([in_distinct,in_top1,in_top3,in_tot,in_hhi], how="left")
        feat["in_top1_ratio"] = feat["in_top1_cnt"]/feat["in_pair_tot"].replace(0,np.nan)
        feat["in_top3_ratio"] = feat["in_top3_cnt"]/feat["in_pair_tot"].replace(0,np.nan)

    if enable_bidir and pair_dir_flags:
        flags = pd.concat(pair_dir_flags).groupby(["lo","hi"]).agg(dir_min=("dir_min","min"),
                                                                   dir_max=("dir_max","max")).reset_index()
        flags["bidir"] = (flags["dir_min"]==0)&(flags["dir_max"]==1)
        b = flags[flags["bidir"]]
        lo_cnt = b.groupby("lo").size().rename("bidir_partner_cnt")
        hi_cnt = b.groupby("hi").size().rename("bidir_partner_cnt")
        bidir_cnt = lo_cnt.add(hi_cnt, fill_value=0)
        feat = feat.join(bidir_cnt, how="left").fillna({"bidir_partner_cnt":0})
        if "out_distinct_to" in feat.columns:
            feat["bidir_out_ratio"] = feat["bidir_partner_cnt"]/feat["out_distinct_to"].replace(0,np.nan)
        if "in_distinct_from" in feat.columns:
            feat["bidir_in_ratio"] = feat["bidir_partner_cnt"]/feat["in_distinct_from"].replace(0,np.nan)

    # 日聚合 + 視窗
    if out_day_list:
        od = pd.concat(out_day_list).groupby(level=[0,1]).sum().reset_index().rename(columns={"from_acct":"acct"})
    else:
        od = pd.DataFrame(columns=["acct","txn_date","out_day_cnt","out_day_amt"])
    if in_day_list:
        idf = pd.concat(in_day_list).groupby(level=[0,1]).sum().reset_index().rename(columns={"to_acct":"acct"})
    else:
        idf = pd.DataFrame(columns=["acct","txn_date","in_day_cnt","in_day_amt"])

    last_date = feat[["out_max_date","in_max_date"]].replace(0,np.nan).max(axis=1).rename("last_date").reset_index()
    last_date = last_date.rename(columns={"index":"acct"})

    def win_sum(df, cols, win):
        if df.empty: return pd.DataFrame()
        z = df.merge(last_date, on="acct", how="left")
        z["diff"] = z["last_date"] - z["txn_date"]
        z = z[(z["diff"]>=0) & (z["diff"]<=win)]
        g = z.groupby("acct")[cols].sum()
        g.columns = [f"{c}_last{win}" for c in cols]
        return g

    for g in [win_sum(od, ["out_day_cnt","out_day_amt"], 7),
              win_sum(od, ["out_day_cnt","out_day_amt"], 30),
              win_sum(idf, ["in_day_cnt","in_day_amt"], 7),
              win_sum(idf, ["in_day_cnt","in_day_amt"], 30)]:
        if not g.empty: feat = feat.join(g, how="left")

    # 指數衰減（半衰期 7/30）
    def decay_agg(df, cnt_col, amt_col, name_prefix, hl):
        if df.empty: return pd.DataFrame()
        lam = math.log(2.0)/hl
        z = df.merge(last_date, on="acct", how="left")
        d = (z["last_date"] - z["txn_date"]).clip(lower=0)
        w = np.exp(-lam*d)
        res = z.groupby("acct").apply(lambda t: pd.Series({
            f"{name_prefix}_cnt_decay_hl{int(hl)}": float((t[cnt_col]*w.loc[t.index]).sum()),
            f"{name_prefix}_amt_decay_hl{int(hl)}": float((t[amt_col]*w.loc[t.index]).sum()),
        }))
        return res

    if enable_decay:
        for hl in [7,30]:
            g1 = decay_agg(od,  "out_day_cnt","out_day_amt","out", hl)
            g2 = decay_agg(idf, "in_day_cnt","in_day_amt","in",  hl)
            if g1 is not None and not g1.empty: feat = feat.join(g1, how="left")
            if g2 is not None and not g2.empty: feat = feat.join(g2,  how="left")

    # 衍生（均值/方差/比例/熵/方向性）
    def safe_std(sum_sq, sum_, n):
        var = (sum_sq/n.replace(0,np.nan)) - (sum_/n.replace(0,np.nan))**2
        return np.sqrt(np.clip(var,0,None))
    feat["out_amt_mean"] = feat["out_amt_sum"]/feat["out_cnt"].replace(0,np.nan)
    feat["in_amt_mean"]  = feat["in_amt_sum"]/feat["in_cnt"].replace(0,np.nan)
    feat["out_amt_std"]  = safe_std(feat["out_amt_sqsum"], feat["out_amt_sum"], feat["out_cnt"]).fillna(0)
    feat["in_amt_std"]   = safe_std(feat["in_amt_sqsum"],  feat["in_amt_sum"],  feat["in_cnt"]).fillna(0)

    feat["out_night_ratio"] = feat["out_night_sum"]/feat["out_cnt"].replace(0,np.nan)
    feat["in_night_ratio"]  = feat["in_night_sum"]/feat["in_cnt"].replace(0,np.nan)
    feat["out_large_ratio"] = feat["out_large_sum"]/feat["out_cnt"].replace(0,np.nan)
    feat["in_large_ratio"]  = feat["in_large_sum"]/feat["in_cnt"].replace(0,np.nan)
    feat["self_ratio"]      = feat["out_self_sum"]/feat["out_cnt"].replace(0,np.nan)

    out_cnt = feat["out_cnt"].replace(0,np.nan); in_cnt = feat["in_cnt"].replace(0,np.nan)
    for c in [c for c in feat.columns if c.startswith("out_ch_")]: feat[c+"_ratio"] = feat[c]/out_cnt
    for c in [c for c in feat.columns if c.startswith("in_ch_")]:  feat[c+"_ratio"]  = feat[c]/in_cnt
    feat["out_ch_entropy"] = entropy_from_cols(feat,"out_ch_","out_ch_entropy")
    feat["in_ch_entropy"]  = entropy_from_cols(feat,"in_ch_", "in_ch_entropy")

    feat["tot_cnt"] = feat["out_cnt"]+feat["in_cnt"]; feat["tot_amt_sum"]=feat["out_amt_sum"]+feat["in_amt_sum"]
    feat["net_cnt"] = feat["in_cnt"]-feat["out_cnt"]; feat["net_amt_sum"]=feat["in_amt_sum"]-feat["out_amt_sum"]
    feat["out_share_cnt"] = feat["out_cnt"]/feat["tot_cnt"].replace(0,np.nan)
    feat["in_share_cnt"]  = feat["in_cnt"]/feat["tot_cnt"].replace(0,np.nan)

    # 清理
    drop_cols = [c for c in feat.columns if c.endswith("_sqsum")] + ["out_min_date","in_min_date"]
    feat = feat.drop(columns=[c for c in drop_cols if c in feat.columns]).fillna(0).reset_index().rename(columns={"index":"acct"})
    return feat

# ========= 三模型（GPU 版，含自動回退）=========
def predict_lgb(X_tr,y_tr,X_va,y_va,X_test,seed=42):
    pos, neg = y_tr.sum(), len(y_tr)-y_tr.sum(); spw = (neg/max(pos,1.0))
    params = dict(objective="binary", learning_rate=0.03, num_leaves=223, min_data_in_leaf=80,
                  feature_fraction=0.90, bagging_fraction=0.90, bagging_freq=5,
                  reg_alpha=0.6, reg_lambda=1.3, metric="binary_logloss",
                  scale_pos_weight=spw, seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
                  verbose=-1, num_threads=os.cpu_count() or 4)
    if USE_GPU:
        params.update(device_type="gpu", gpu_use_dp=False, max_bin=255)
    dtr=lgb.Dataset(X_tr,label=y_tr); dva=lgb.Dataset(X_va,label=y_va,reference=dtr)
    try:
        model=lgb.train(params,dtr,num_boost_round=LGB_ROUNDS,valid_sets=[dva],
                        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(300)])
    except Exception as e:
        print("[warn] LightGBM GPU 啟動失敗，改用 CPU；原因：", e)
        params.pop("device_type", None)
        params.pop("gpu_use_dp", None)
        model=lgb.train(params,dtr,num_boost_round=LGB_ROUNDS,valid_sets=[dva],
                        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(300)])
    return model.predict(X_va, num_iteration=model.best_iteration), \
           model.predict(X_test,num_iteration=model.best_iteration)

def predict_xgb(X_tr,y_tr,X_va,y_va,X_test,seed=2025):
    pos, neg = y_tr.sum(), len(y_tr)-y_tr.sum(); spw = (neg/max(pos,1.0))
    dtr=xgb.DMatrix(X_tr,label=y_tr); dva=xgb.DMatrix(X_va,label=y_va); dte=xgb.DMatrix(X_test)

    params=dict(objective="binary:logistic", eval_metric="logloss",
                eta=0.03, max_depth=8, min_child_weight=2,
                subsample=0.90, colsample_bytree=0.90,
                reg_lambda=1.3, alpha=0.5,
                nthread=os.cpu_count() or 4, scale_pos_weight=spw, seed=seed)

    # XGBoost 1.x / 2.x 相容處理 + GPU 優先
    try:
        from packaging.version import Version
        ver = Version(xgb.__version__)
        if USE_GPU:
            if ver >= Version("2.0.0"):
                params.update(tree_method="hist", device="cuda")
            else:
                params.update(tree_method="gpu_hist", predictor="gpu_predictor")
        else:
            params.update(tree_method="hist")
        model = xgb.train(params,dtr,num_boost_round=XGB_ROUNDS,evals=[(dtr,'train'),(dva,'valid')],
                          early_stopping_rounds=EARLY_STOP,verbose_eval=300)
    except Exception as e:
        print("[warn] XGBoost GPU 失敗，改用 CPU；原因：", e)
        # 回退 CPU
        params.pop("device", None); params.pop("predictor", None)
        params.update(tree_method="hist")
        model = xgb.train(params,dtr,num_boost_round=XGB_ROUNDS,evals=[(dtr,'train'),(dva,'valid')],
                          early_stopping_rounds=EARLY_STOP,verbose_eval=300)

    return model.predict(dva, iteration_range=(0, model.best_iteration+1)), \
           model.predict(dte, iteration_range=(0, model.best_iteration+1))

def predict_cat(X_tr,y_tr,X_va,y_va,X_test,seed=7):
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        print("[warn] catboost 不可用，略過；原因：", e)
        return np.zeros(len(X_va)), np.zeros(len(X_test))
    pos, neg = y_tr.sum(), len(y_tr)-y_tr.sum()
    kwargs=dict(iterations=XGB_ROUNDS, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                random_seed=seed, eval_metric="Logloss", loss_function="Logloss",
                class_weights=[1.0, float(neg/max(pos,1.0))], subsample=0.90, rsm=0.90, verbose=300)
    if USE_GPU: kwargs.update(task_type="GPU", devices=os.environ.get("CUDA_VISIBLE_DEVICES","0"))
    model=CatBoostClassifier(**kwargs)
    try:
        model.fit(X_tr,y_tr,eval_set=(X_va,y_va),use_best_model=True,early_stopping_rounds=EARLY_STOP)
    except Exception as e:
        print("[warn] CatBoost GPU 失敗，改用 CPU；原因：", e)
        kwargs.pop("task_type", None); kwargs.pop("devices", None)
        model=CatBoostClassifier(**kwargs)
        model.fit(X_tr,y_tr,eval_set=(X_va,y_va),use_best_model=True,early_stopping_rounds=EARLY_STOP)
    return model.predict_proba(X_va)[:,1], model.predict_proba(X_test)[:,1]

# ========= 單組 (切分 q, 下採樣 r) → 多種子訓練/融合 =========
def run_one(feat, labels, pred_accts, q, ratio):
    df = feat.merge(labels, on="acct", how="left").fillna({"label":0})
    df["label"]=df["label"].astype(int)
    tr, va = time_split(df, q)
    X_tr_full = tr.drop(columns=["acct","label"]); y_tr_full = tr["label"].values
    X_va      = va.drop(columns=["acct","label"]); y_va      = va["label"].values
    X_tr, y_tr = downsample(X_tr_full, y_tr_full, ratio=ratio, seed=123)

    test_feat = pred_accts.merge(feat, on="acct", how="left").fillna(0)
    X_test = test_feat.drop(columns=["acct"])

    # 多種子 bagging
    lgb_va_list, lgb_te_list = [], []
    for sd in SEEDS_LGB:
        v,t = predict_lgb(X_tr,y_tr,X_va,y_va,X_test,seed=sd)
        lgb_va_list.append(v); lgb_te_list.append(t)
    xgb_va_list, xgb_te_list = [], []
    for sd in SEEDS_XGB:
        v,t = predict_xgb(X_tr,y_tr,X_va,y_va,X_test,seed=sd)
        xgb_va_list.append(v); xgb_te_list.append(t)
    cat_va_list, cat_te_list = [], []
    try:
        from catboost import CatBoostClassifier  # 檢查是否可用
        CAT_ON = True
        for sd in SEEDS_CAT:
            v,t = predict_cat(X_tr,y_tr,X_va,y_va,X_test,seed=sd)
            cat_va_list.append(v); cat_te_list.append(t)
    except Exception:
        CAT_ON = False

    lgb_va = np.mean(lgb_va_list, axis=0); lgb_te = np.mean(lgb_te_list, axis=0)
    xgb_va = np.mean(xgb_va_list, axis=0); xgb_te = np.mean(xgb_te_list, axis=0)
    if CAT_ON:
        cat_va = np.mean(cat_va_list, axis=0); cat_te = np.mean(cat_te_list, axis=0)
    else:
        cat_va = np.zeros_like(lgb_va); cat_te = np.zeros_like(lgb_te)

    # logit 加權搜尋（粗→細）
    def search_weights(L, X, C=None):
        best = (-1.0, None, 0.5, None)  # f1, weights, thr, va_mix
        if C is None:
            # 2 模型
            for w1 in np.linspace(0.1,0.9,17):
                va_mix = sigmoid(w1*L + (1.0-w1)*X)
                t,f1 = best_threshold(y_va, va_mix)
                if f1 > best[0]: best = (f1, (w1,1.0-w1), t, va_mix)
            # 細調
            w1_star = best[1][0]
            for w1 in np.linspace(max(0.05,w1_star-0.15), min(0.95,w1_star+0.15), 25):
                va_mix = sigmoid(w1*L + (1.0-w1)*X)
                t,f1 = best_threshold(y_va, va_mix)
                if f1 > best[0]: best = (f1, (w1,1.0-w1), t, va_mix)
        else:
            # 3 模型
            for w1 in np.linspace(0.15,0.75,13):
                for w2 in np.linspace(0.15,0.75,13):
                    if w1+w2>=0.98: continue
                    w3 = 1.0-w1-w2
                    va_mix = sigmoid(w1*L + w2*X + w3*C)
                    t,f1 = best_threshold(y_va, va_mix)
                    if f1 > best[0]: best = (f1, (w1,w2,w3), t, va_mix)
            # 細調
            w1,w2,w3 = best[1]
            for W1 in np.linspace(max(0.05,w1-0.15), min(0.9,w1+0.15), 19):
                for W2 in np.linspace(max(0.05,w2-0.15), min(0.9,w2+0.15), 19):
                    if W1+W2>=0.98: continue
                    W3 = 1.0-W1-W2
                    va_mix = sigmoid(W1*L + W2*X + W3*C)
                    t,f1 = best_threshold(y_va, va_mix)
                    if f1 > best[0]: best = (f1, (W1,W2,W3), t, va_mix)
        return best

    L, X = logit(lgb_va), logit(xgb_va)
    if CAT_ON:
        C = logit(cat_va)
        best_f1,(w1,w2,w3),best_t,va_mix = search_weights(L,X,C)
        te_mix = sigmoid(w1*logit(lgb_te)+w2*logit(xgb_te)+w3*logit(cat_te))
        print(f"[q={q:.2f}, r={ratio}] 3-Model weights=({w1:.3f},{w2:.3f},{w3:.3f})  OOF F1={best_f1:.4f} @ thr={best_t:.3f}")
    else:
        best_f1,(w1,w2),best_t,va_mix = search_weights(L,X,None)
        te_mix = sigmoid(w1*logit(lgb_te)+w2*logit(xgb_te))
        print(f"[q={q:.2f}, r={ratio}] 2-Model weights=({w1:.3f},{w2:.3f})      OOF F1={best_f1:.4f} @ thr={best_t:.3f}")

    pos_rate = (va_mix >= best_t).mean()

    return dict(
        oof_f1=float(best_f1), thr=float(best_t), pos_rate=float(pos_rate),
        te_mix=te_mix, n_test=len(te_mix)
    )

# ========= 主流程 =========
def main():
    # 讀/建特徵（快取）
    if os.path.exists(FEAT_CACHE):
        feat = pd.read_parquet(FEAT_CACHE)
        print(f"[cache] loaded {FEAT_CACHE}, shape={feat.shape}")
    else:
        print("Building features (GPU-boosted, enhanced)...")
        t0 = tic()
        feat = build_features(TXN_PATH, chunksize=CHUNKSIZE, enable_bidir=USE_BIDIR, enable_decay=USE_DECAY)
        toc(t0, "feature building")
        try:
            feat.to_parquet(FEAT_CACHE, index=False)
            print(f"[cache] saved -> {FEAT_CACHE}")
        except Exception as e:
            print("[cache] save failed:", e)

    labels = load_labels(ALERT_PATH)
    pred_accts = pd.read_csv(PRED_PATH)[["acct"]]

    # 掃描多組 (q, ratio)
    results = []
    for q in QS_CAND:
        for r in NEG_POS_CAND:
            results.append(run_one(feat, labels, pred_accts, q, r))

    # 以 OOF F1 當權重做堆疊
    weights = np.array([max(1e-6, r["oof_f1"]) for r in results], dtype=float)
    weights = weights/weights.sum()
    te_stack = np.zeros(results[0]["n_test"], dtype=float)
    for w,r in zip(weights, results):
        te_stack += w * r["te_mix"]

    thr_avg  = float(np.sum([w*r["thr"] for w,r in zip(weights, results)]))
    rate_avg = float(np.sum([w*r["pos_rate"] for w,r in zip(weights, results)]))

    # 兩種提交：threshold 與 top-k
    te_label_thr = (te_stack >= thr_avg).astype(int)
    k = int(round(rate_avg * len(te_stack)))
    order = np.argsort(-te_stack)
    te_label_topk = np.zeros_like(te_stack, dtype=int)
    te_label_topk[order[:k]] = 1

    sub_thr  = pd.DataFrame({"acct": pred_accts["acct"], "label": te_label_thr})
    sub_topk = pd.DataFrame({"acct": pred_accts["acct"], "label": te_label_topk})
    sub_thr.to_csv(SUB_THR, index=False)
    sub_topk.to_csv(SUB_TOPK, index=False)
    print(f"Saved -> {SUB_THR}")
    print(f"Saved -> {SUB_TOPK}")
    print(f"[ensemble] weighted thr={thr_avg:.3f}, pos_rate={rate_avg:.5f}")
    print("submission (thr)  positive rate =", sub_thr["label"].mean())
    print("submission (topk) positive rate =", sub_topk["label"].mean())

if __name__ == "__main__":
    main()

