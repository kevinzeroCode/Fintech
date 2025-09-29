# -*- coding: utf-8 -*-
"""
E.SUN – Post-run Booster (No retraining)
讀取 ckpt/meta_results.csv 與 te_mix_*.npy，做：
1) prob-mix（oof_f1 → logit 加權）
2) rank-mix（oof_f1 → 排名加權）
3) 估計最合理陽性數 K_base（權重平均各模型的 thr→在 test 上的陽性數）
4) 依 K_base 在附近輸出多份 top-K 提交檔 + 一份 threshold 檔

會輸出：
- submission_probmix_threshold.csv
- submission_probmix_topk_auto.csv
- submission_rankmix_topk_auto.csv
- submission_rankmix_topk_K{K}.csv（多個 K）
"""

import os
import numpy as np
import pandas as pd

BASE = "."
CKPT_DIR = os.path.join(BASE, "ckpt")
META_CSV = os.path.join(CKPT_DIR, "meta_results.csv")
PRED_PATH = os.path.join(BASE, "acct_predict.csv")

# ---------- 小工具 ----------
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p/(1-p))

def rank01(x):
    # 轉成 [0,1] 的排名分數（越大代表越前面）
    order = np.argsort(x)               # 小→大
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks / max(len(x)-1, 1)

def z01(x):
    x = np.asarray(x, float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi <= lo: return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def save_submission(path, accts, labels):
    pd.DataFrame({"acct": accts, "label": labels.astype(int)}).to_csv(path, index=False)
    print(f"Saved -> {path} | positives={int(labels.sum())}/{len(labels)}")

# ---------- 讀檔 ----------
assert os.path.exists(META_CSV), f"找不到 {META_CSV}"
meta = pd.read_csv(META_CSV)  # 必須包含 cols: q,ratio,oof_f1,thr,mix_path,n_test...
needed = {"oof_f1","thr","mix_path"}
missing = needed - set(meta.columns)
if missing:
    raise RuntimeError(f"meta_results.csv 缺少欄位：{missing}")

pred_accts = pd.read_csv(PRED_PATH)["acct"].values
n_test = len(pred_accts)

# 只保留真的存在且長度正確的 te 檔
rows = []
for _, r in meta.iterrows():
    p = os.path.normpath(os.path.join(CKPT_DIR, os.path.basename(str(r["mix_path"]))))
    if not os.path.exists(p): 
        print(f"[skip] not found: {p}")
        continue
    arr = np.load(p)
    if arr.shape[0] != n_test:
        print(f"[skip] shape mismatch: {p}, got {arr.shape[0]} != {n_test}")
        continue
    rows.append(dict(path=p, oof_f1=float(r["oof_f1"]), thr=float(r["thr"]), prob=arr))

assert rows, "沒有可用的 te_mix_*.npy，請確認 ckpt/ 內檔案"

# 權重（用 OOF F1，避免全 0）
w = np.array([max(1e-6, r["oof_f1"]) for r in rows], dtype=float)
w = w / w.sum()

print(f"Loaded {len(rows)} test-prob files. n_test={n_test}")
for i, r in enumerate(rows[:4]):  # 顯示前幾個
    a = r["prob"]
    print(f"  [{i}] {os.path.basename(r['path'])}  min/max={a.min():.6f}/{a.max():.6f}  oof_w={w[i]:.3f}")

# ---------- 兩種融合 ----------
# (A) prob-mix：logit 加權再 sigmoid
L = np.zeros(n_test, dtype=float)
for wi, r in zip(w, rows):
    L += wi * logit(r["prob"])
probmix = sigmoid(L)

# (B) rank-mix：各模型先做 rank01，再加權平均
R = np.zeros(n_test, dtype=float)
for wi, r in zip(w, rows):
    R += wi * rank01(r["prob"])
rankmix = z01(R)  # 轉回 [0,1] 便於直覺

print(f"[probmix] min/max={probmix.min():.6f}/{probmix.max():.6f}")
print(f"[rankmix] min/max={rankmix.min():.6f}/{rankmix.max():.6f}")

# ---------- 估計最合理 K（權重平均） ----------
# 各模型：用它自己的 thr 去切 test，得到陽性數 → 再做權重平均得到 K_base
K_each = np.array([(r["prob"] >= r["thr"]).sum() for r in rows], dtype=float)
K_base = int(round(np.dot(w, K_each)))
K_base = max(1, min(K_base, n_test))
print(f"Estimated K_base (weighted by oof_f1) = {K_base}  | per-model Ks = {K_each.astype(int).tolist()}")

# 也給一個「平均 threshold」產生的 threshold 版
thr_star = float(np.dot(w, np.array([r["thr"] for r in rows], dtype=float)))
labels_thr = (probmix >= thr_star).astype(int)
save_submission(os.path.join(BASE, "submission_probmix_threshold.csv"), pred_accts, labels_thr)

# ---------- 產生多個鄰近 K 的 top-K 檔 ----------
# 以 K_base 為中心，放大縮小一些倍率；再加幾個常見 K
alphas = [0.70, 0.85, 0.95, 1.00, 1.15, 1.30, 1.60]
extraK = [5, 10, 15, 20, 25, 30, 40]
K_grid = sorted({max(1, min(n_test, int(round(K_base*a)))) for a in alphas} | set(extraK))

# 用兩種分數都輸出：probmix 與 rankmix（auto 主推 rankmix）
def topk_labels(score, K):
    idx = np.argsort(-score)[:K]  # 取前 K 名
    lab = np.zeros(n_test, dtype=int); lab[idx] = 1
    return lab

# Auto（採用 rankmix 與 K_base）
lab_rank_auto = topk_labels(rankmix, K_base)
save_submission(os.path.join(BASE, "submission_rankmix_topk_auto.csv"), pred_accts, lab_rank_auto)

# 另給一份 probmix 的 auto，便於 A/B
lab_prob_auto = topk_labels(probmix, K_base)
save_submission(os.path.join(BASE, "submission_probmix_topk_auto.csv"), pred_accts, lab_prob_auto)

# K-grid（rankmix 版本）
for K in K_grid:
    lab = topk_labels(rankmix, K)
    out = os.path.join(BASE, f"submission_rankmix_topk_K{K}.csv")
    save_submission(out, pred_accts, lab)

# ---------- 診斷資訊 ----------
def head_ids(score, K=10):
    idx = np.argsort(-score)[:K]
    return pred_accts[idx].tolist()

print("\n[Diagnostics]")
print(f"Top-10 by rankmix: {head_ids(rankmix, 10)}")
print(f"Top-10 by probmix: {head_ids(probmix, 10)}")
print(f"K_grid around {K_base}: {K_grid}")
print("建議先投：submission_rankmix_topk_auto.csv 與 submission_probmix_threshold.csv；\n"
      "再視榜分決定是否改用 K_grid 中其他 K 的檔案。")
