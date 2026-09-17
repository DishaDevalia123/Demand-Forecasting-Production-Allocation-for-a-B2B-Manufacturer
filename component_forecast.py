# %%
import pandas as pd, numpy as np, sqlite3
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

SEASONS = ["Summer 2021","Diwali 2021","Christmas 2021","Pongal 2022","Summer 2022",
           "Diwali 2022","Christmas 2022","Pongal 2023","Summer 2023","Diwali 2023"]
SIDX  = {s: i for i, s in enumerate(SEASONS)}
QS    = [0.1, 0.5, 0.9]
FOLDS = [(list(range(0, k)), k) for k in range(5, 10)]
FOLD_NAMES = [SEASONS[i] for _, i in FOLDS]

# %%
con = sqlite3.connect("garment.db")
df = pd.read_sql("""
    SELECT a.season, a.design_code, a.demand, a.n_samples_carried,
           d.main_group AS brand, d.item_group, d.pattern, d.sleeve,
           d.colour, d.rate, d.season_introduced
    FROM analysis_demand a
    JOIN dim_design d ON a.design_code = d.design_code
""", con)
con.close()

df["si"]       = df["season"].map(SIDX)
df["festival"] = df["season"].str.split().str[0]
df["is_new"]   = df["season"] == df["season_introduced"]
print(df.shape)

# %%
def pinball(y_true, y_pred, q):
    """Quantile loss. Under-prediction costs q per unit, over-prediction costs (1-q)."""
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))


def group_quantiles(train, keys, qs=QS, min_n=40):
    """Empirical quantiles per group; back off to the first key when a group is sparse."""
    fine   = train.groupby(keys, observed=True)["demand"].quantile(list(qs)).unstack()
    counts = train.groupby(keys, observed=True)["demand"].size()
    coarse = train.groupby(keys[0], observed=True)["demand"].quantile(list(qs)).unstack()
    for idx in counts[counts < min_n].index:
        fine.loc[idx] = coarse.loc[idx[0]].values
    return fine


def add_encodings(data):
    """As-of encodings: each row uses ONLY seasons strictly before its own season."""
    out = []
    for s in sorted(data.si.unique()):
        cur, hist = data[data.si == s].copy(), data[data.si < s]

        if len(hist) == 0:
            for c in ["enc_bp_mean", "enc_bp_zero", "enc_brand_lastyear"]:
                cur[c] = np.nan
            out.append(cur)
            continue

        bp = hist.groupby(["brand", "pattern"])["demand"].agg(
            enc_bp_mean="mean", enc_bp_zero=lambda x: (x == 0).mean()).reset_index()
        cur = cur.merge(bp, on=["brand", "pattern"], how="left")

        h = hist.copy()
        h["year"] = h["season"].str.split().str[1].astype(int)
        fy = h.groupby(["brand", "festival", "year"])["demand"].mean().reset_index(
            name="enc_brand_lastyear")
        fy["year"] += 1
        cur["year"] = cur["season"].str.split().str[1].astype(int)
        cur = cur.merge(fy, on=["brand", "festival", "year"], how="left")

        out.append(cur)
    return pd.concat(out, ignore_index=True)

# %%
enc = add_encodings(df)
print(enc.groupby("si")[["enc_bp_mean", "enc_bp_zero", "enc_brand_lastyear"]]
        .apply(lambda g: g.isna().sum()))

FEATS = ["brand", "item_group", "pattern", "sleeve", "colour", "rate", "festival",
         "enc_bp_mean", "enc_bp_zero", "enc_brand_lastyear"]
CATS  = ["brand", "item_group", "pattern", "sleeve", "colour", "festival"]

Xall = pd.get_dummies(enc[FEATS], columns=CATS).fillna(-999)

# %%
def fit_fold(tr_idx, te_idx):
    """Fit all quantiles on one fold. Returns the test frame with model + baseline columns."""
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    Xtr, ytr = Xall[m_tr], enc.loc[m_tr, "demand"].to_numpy(float)
    te = enc.loc[m_te].reset_index(drop=True)

    bq = group_quantiles(enc[m_tr], ["brand", "pattern"])
    bq.columns = [f"b{q}" for q in QS]
    te = te.merge(bq.reset_index(), on=["brand", "pattern"], how="left")

    for q in QS:
        mdl = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=400,
                                        learning_rate=0.05, max_depth=4,
                                        min_samples_leaf=20, random_state=0)
        mdl.fit(Xtr, ytr)
        te[f"m{q}"] = np.maximum(mdl.predict(Xall[m_te]), 0)
    return te

preds = {SEASONS[te]: fit_fold(tr, te) for tr, te in FOLDS}
print("folds fitted:", list(preds))

# %%
rows = []
for fold, te in preds.items():
    for q in QS:
        rows.append({"fold": fold, "q": q,
                     "model": pinball(te["demand"].values, te[f"m{q}"].values, q),
                     "base":  pinball(te["demand"].values, te[f"b{q}"].values, q)})
res = pd.DataFrame(rows)

model = res.pivot(index="fold", columns="q", values="model").reindex(FOLD_NAMES).round(1)
base  = res.pivot(index="fold", columns="q", values="base").reindex(FOLD_NAMES).round(1)

print("MODEL:");      print(model)
print("\nBASELINE:"); print(base)
print("\nIMPROVEMENT % (positive = model better):")
print(((base - model) / base * 100).round(1))

# %%
rows = []
for fold, te in preds.items():
    for b, g in te.groupby("brand", observed=True):
        for q in QS:
            rows.append({"brand": b, "q": q,
                         "model": pinball(g["demand"].values, g[f"m{q}"].values, q),
                         "base":  pinball(g["demand"].values, g[f"b{q}"].values, q)})

pb = pd.DataFrame(rows).groupby(["brand", "q"])[["model", "base"]].mean()
pb["improv_%"] = ((pb["base"] - pb["model"]) / pb["base"] * 100).round(1)
print(pb.round(1))
# %%
def fit_fold_perbrand(tr_idx, te_idx, min_train=150):
    """Fit a separate model per brand. Falls back to the pooled model when a brand is thin."""
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    te = enc.loc[m_te].reset_index(drop=True)

    bq = group_quantiles(enc[m_tr], ["brand", "pattern"])
    bq.columns = [f"b{q}" for q in QS]
    te = te.merge(bq.reset_index(), on=["brand", "pattern"], how="left")

    # pooled fallback, fitted once
    pooled = {}
    for q in QS:
        mdl = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=400,
                                        learning_rate=0.05, max_depth=4,
                                        min_samples_leaf=20, random_state=0)
        mdl.fit(Xall[m_tr], enc.loc[m_tr, "demand"].to_numpy(float))
        pooled[q] = mdl

    for q in QS:
        te[f"m{q}"] = np.nan

    for b in enc["brand"].unique():
        tr_b = m_tr & (enc["brand"] == b).values
        te_b = (te["brand"] == b).values
        if te_b.sum() == 0:
            continue

        if tr_b.sum() < min_train:                      # too thin -> pooled model
            for q in QS:
                te.loc[te_b, f"m{q}"] = np.maximum(pooled[q].predict(Xall[m_te][te_b]), 0)
            continue

        for q in QS:
            mdl = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=300,
                                            learning_rate=0.05, max_depth=3,
                                            min_samples_leaf=15, random_state=0)
            mdl.fit(Xall[tr_b], enc.loc[tr_b, "demand"].to_numpy(float))
            te.loc[te_b, f"m{q}"] = np.maximum(mdl.predict(Xall[m_te][te_b]), 0)

    return te

preds_pb = {SEASONS[te]: fit_fold_perbrand(tr, te) for tr, te in FOLDS}
print("done")
# %%
rows = []
for fold in preds:
    for b in preds[fold]["brand"].unique():
        g1 = preds[fold][preds[fold]["brand"] == b]
        g2 = preds_pb[fold][preds_pb[fold]["brand"] == b]
        for q in QS:
            rows.append({"brand": b, "q": q,
                         "base":     pinball(g1["demand"].values, g1[f"b{q}"].values, q),
                         "pooled":   pinball(g1["demand"].values, g1[f"m{q}"].values, q),
                         "perbrand": pinball(g2["demand"].values, g2[f"m{q}"].values, q)})

cmp = pd.DataFrame(rows).groupby(["brand", "q"])[["base", "pooled", "perbrand"]].mean()
cmp["pooled_%"]   = ((cmp["base"] - cmp["pooled"])   / cmp["base"] * 100).round(1)
cmp["perbrand_%"] = ((cmp["base"] - cmp["perbrand"]) / cmp["base"] * 100).round(1)
print(cmp.round(1))
# %%
from sklearn.linear_model import QuantileRegressor
from itertools import product

GRID = list(product(
    [200, 400, 800],      # n_estimators
    [0.03, 0.05, 0.1],    # learning_rate
    [2, 3, 4],            # max_depth
))
print(f"{len(GRID)} configs x {len(QS)} quantiles x {len(FOLDS)} folds")


def tune_and_fit(tr_idx, te_idx, q):
    """Inner loop picks the config on the last training season; outer refits and predicts."""
    inner_tr = tr_idx[:-1]
    inner_va = tr_idx[-1]

    m_itr = enc.si.isin(inner_tr).values
    m_iva = (enc.si == inner_va).values
    y_itr = enc.loc[m_itr, "demand"].to_numpy(float)
    y_iva = enc.loc[m_iva, "demand"].to_numpy(float)

    best, best_loss = None, np.inf
    for n, lr, dep in GRID:
        mdl = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=n,
                                        learning_rate=lr, max_depth=dep,
                                        min_samples_leaf=20, random_state=0)
        mdl.fit(Xall[m_itr], y_itr)
        loss = pinball(y_iva, np.maximum(mdl.predict(Xall[m_iva]), 0), q)
        if loss < best_loss:
            best, best_loss = (n, lr, dep), loss

    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    n, lr, dep = best
    final = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=n,
                                      learning_rate=lr, max_depth=dep,
                                      min_samples_leaf=20, random_state=0)
    final.fit(Xall[m_tr], enc.loc[m_tr, "demand"].to_numpy(float))
    return np.maximum(final.predict(Xall[m_te]), 0), best
# %%
tuned, chosen = {}, []
for tr_idx, te_idx in FOLDS:
    te = preds[SEASONS[te_idx]].copy()
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    ytr = enc.loc[m_tr, "demand"].to_numpy(float)

    for q in QS:
        p, cfg = tune_and_fit(tr_idx, te_idx, q)
        te[f"t{q}"] = p
        chosen.append({"fold": SEASONS[te_idx], "q": q, "config": cfg})

        lin = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
        lin.fit(Xall[m_tr].astype(float), ytr)
        te[f"l{q}"] = np.maximum(lin.predict(Xall[m_te].astype(float)), 0)

    tuned[SEASONS[te_idx]] = te

print(pd.DataFrame(chosen).to_string(index=False))
# %%
rows = []
for fold, te in tuned.items():
    for q in QS:
        rows.append({"fold": fold, "q": q,
                     "base":   pinball(te["demand"].values, te[f"b{q}"].values, q),
                     "linear": pinball(te["demand"].values, te[f"l{q}"].values, q),
                     "gbm":    pinball(te["demand"].values, te[f"m{q}"].values, q),
                     "gbm_tuned": pinball(te["demand"].values, te[f"t{q}"].values, q)})

r = pd.DataFrame(rows).groupby("q")[["base","linear","gbm","gbm_tuned"]].mean().round(1)
for c in ["linear","gbm","gbm_tuned"]:
    r[f"{c}_%"] = ((r["base"] - r[c]) / r["base"] * 100).round(1)
print(r)

# %%
import time

t0 = time.time()
tuned, chosen = {}, []
for tr_idx, te_idx in FOLDS:
    te = preds[SEASONS[te_idx]].copy()
    for q in QS:
        p, cfg = tune_and_fit(tr_idx, te_idx, q)
        te[f"t{q}"] = p
        chosen.append({"fold": SEASONS[te_idx], "q": q, "config": cfg})
    tuned[SEASONS[te_idx]] = te
    print(f"  {SEASONS[te_idx]} done  ({time.time()-t0:.0f}s)")

print(pd.DataFrame(chosen).to_string(index=False))
# %%
rows = []
for fold, te in tuned.items():
    for q in QS:
        rows.append({"fold": fold, "q": q,
                     "base":      pinball(te["demand"].values, te[f"b{q}"].values, q),
                     "gbm":       pinball(te["demand"].values, te[f"m{q}"].values, q),
                     "gbm_tuned": pinball(te["demand"].values, te[f"t{q}"].values, q)})

r = pd.DataFrame(rows).groupby("q")[["base","gbm","gbm_tuned"]].mean().round(1)
r["gbm_%"]   = ((r["base"] - r["gbm"])       / r["base"] * 100).round(1)
r["tuned_%"] = ((r["base"] - r["gbm_tuned"]) / r["base"] * 100).round(1)
print(r)
# %%
rows = []
for fold, te in tuned.items():
    for q in QS:
        rows.append({"q": q,
                     "tuned_cov": (te["demand"].values <= te[f"t{q}"].values).mean(),
                     "gbm_cov":   (te["demand"].values <= te[f"m{q}"].values).mean(),
                     "base_cov":  (te["demand"].values <= te[f"b{q}"].values).mean()})
cov = pd.DataFrame(rows).groupby("q")[["tuned_cov","gbm_cov","base_cov"]].mean().round(3)
print("\nCOVERAGE (should match the quantile):")
print(cov)

print("\nQUANTILE CROSSING:")
for fold, te in tuned.items():
    print(f"  {fold}: {((te['t0.1'] > te['t0.5']) | (te['t0.5'] > te['t0.9'])).sum()} of {len(te)}")
# %%
for fold, te in tuned.items():
    cols = [f"t{q}" for q in QS]
    te[cols] = np.sort(te[cols].values, axis=1)

rows = []
for fold, te in tuned.items():
    for q in QS:
        rows.append({"q": q,
                     "base":  pinball(te["demand"].values, te[f"b{q}"].values, q),
                     "tuned": pinball(te["demand"].values, te[f"t{q}"].values, q)})
r2 = pd.DataFrame(rows).groupby("q")[["base","tuned"]].mean().round(1)
r2["improv_%"] = ((r2["base"] - r2["tuned"]) / r2["base"] * 100).round(1)
print(r2)

print("\ncrossings after rearrangement:")
for fold, te in tuned.items():
    print(f"  {fold}: {((te['t0.1'] > te['t0.5']) | (te['t0.5'] > te['t0.9'])).sum()}")
# %%
import time
from sklearn.linear_model import QuantileRegressor

t0 = time.time()
for tr_idx, te_idx in FOLDS:
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    ytr = enc.loc[m_tr, "demand"].to_numpy(float)
    for q in QS:
        lin = QuantileRegressor(quantile=q, alpha=1.0, solver="highs-ds")
        lin.fit(Xall[m_tr].astype(float).values, ytr)
        tuned[SEASONS[te_idx]][f"l{q}"] = np.maximum(
            lin.predict(Xall[m_te].astype(float).values), 0)
    print(f"  {SEASONS[te_idx]} ({time.time()-t0:.0f}s)")

rows = []
for fold, te in tuned.items():
    for q in QS:
        rows.append({"q": q,
                     "base":   pinball(te["demand"].values, te[f"b{q}"].values, q),
                     "linear": pinball(te["demand"].values, te[f"l{q}"].values, q),
                     "gbm":    pinball(te["demand"].values, te[f"t{q}"].values, q)})
r3 = pd.DataFrame(rows).groupby("q")[["base","linear","gbm"]].mean().round(1)
for c in ["linear","gbm"]:
    r3[f"{c}_%"] = ((r3["base"] - r3[c]) / r3["base"] * 100).round(1)
print(r3)
# %%
allp = pd.concat([te.assign(fold=f) for f, te in tuned.items()], ignore_index=True)
keep = ["fold","season","design_code","brand","item_group","pattern","sleeve",
        "colour","rate","demand","n_samples_carried"] + \
       [f"t{q}" for q in QS] + [f"b{q}" for q in QS]
allp[keep].to_csv("data/clean/component1_predictions.csv", index=False)

con = sqlite3.connect("garment.db")
allp[keep].to_sql("component1_predictions", con, if_exists="replace", index=False)
con.close()
print(allp[keep].shape)
# %%
QS = [0.1, 0.5, 0.75, 0.9, 0.95]

def fit_fold(tr_idx, te_idx):
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    Xtr, ytr = Xall[m_tr], enc.loc[m_tr, "demand"].to_numpy(float)
    te = enc.loc[m_te].reset_index(drop=True)

    bq = group_quantiles(enc[m_tr], ["brand", "pattern"], qs=QS)   # <- explicit
    bq.columns = [f"b{q}" for q in QS]
    te = te.merge(bq.reset_index(), on=["brand", "pattern"], how="left")

    for q in QS:
        mdl = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=400,
                                        learning_rate=0.05, max_depth=4,
                                        min_samples_leaf=20, random_state=0)
        mdl.fit(Xtr, ytr)
        te[f"m{q}"] = np.maximum(mdl.predict(Xall[m_te]), 0)
    return te

preds = {SEASONS[te]: fit_fold(tr, te) for tr, te in FOLDS}

for fold, te in preds.items():
    cols = [f"m{q}" for q in QS]
    te[cols] = np.sort(te[cols].values, axis=1)

rows = []
for fold, te in preds.items():
    for q in QS:
        rows.append({"q": q,
                     "base":  pinball(te["demand"].values, te[f"b{q}"].values, q),
                     "model": pinball(te["demand"].values, te[f"m{q}"].values, q),
                     "cov":   (te["demand"].values <= te[f"m{q}"].values).mean()})
r = pd.DataFrame(rows).groupby("q")[["base","model","cov"]].mean().round(3)
r["improv_%"] = ((r["base"] - r["model"]) / r["base"] * 100).round(1)
print(r)
# %%
allp = pd.concat([te.assign(fold=f) for f, te in preds.items()], ignore_index=True)
keep = ["fold","season","design_code","brand","item_group","pattern","sleeve",
        "colour","rate","demand"] + [f"m{q}" for q in QS] + [f"b{q}" for q in QS]
con = sqlite3.connect("garment.db")
allp[keep].to_sql("component1_predictions", con, if_exists="replace", index=False)
con.close()
print(allp[keep].shape)
# %%
con = sqlite3.connect("garment.db")
chk = pd.read_sql("SELECT * FROM component1_predictions LIMIT 3", con)
n = pd.read_sql("SELECT COUNT(*) AS n FROM component1_predictions", con).iloc[0,0]
con.close()

print("rows:", n)
print("columns:", list(chk.columns))

# %%
# 5. is the +18% broad or concentrated?
def row_pinball(y, p, q):
    e = y - p
    return np.maximum(q * e, (q - 1) * e)

for q in [0.75, 0.9]:
    d = row_pinball(P["demand"].values, P[f"b{q}"].values, q) - \
        row_pinball(P["demand"].values, P[f"m{q}"].values, q)
    print(f"q={q}: mean gain/row {d.mean():.1f}   median {np.median(d):.1f}   "
          f"share of rows improved {(d > 0).mean():.1%}")
    top = np.sort(d)[::-1]
    print(f"   top 1% of rows contribute {top[:int(.01*len(d))].sum()/d.sum():.1%} of total gain")
    print(f"   top 10% contribute {top[:int(.10*len(d))].sum()/d.sum():.1%}")
# %%
# 3. does any brand x pattern group have zero rate under 10%?
con = sqlite3.connect("garment.db")
ad = pd.read_sql("""SELECT a.season, a.demand, d.main_group AS brand, d.pattern
                    FROM analysis_demand a JOIN dim_design d ON a.design_code=d.design_code""", con)
con.close()
z = ad.groupby(["brand","pattern"])["demand"].apply(lambda s: (s==0).mean()).sort_values()
print(z.head(8).round(3))
print("\ngroups below 10% zero rate:", (z < 0.10).sum())
# %%
# 1. coverage by brand - is the under-coverage uniform or concentrated?
for q in [0.75, 0.9, 0.95]:
    cov = P.groupby("brand").apply(lambda g: (g["demand"] <= g[f"m{q}"]).mean())
    print(f"\nq={q} (target {q}):")
    print(cov.round(3))
# %%
# scale artifact check: is the gain proportional or just bigger denominators?
for q in [0.75, 0.9]:
    b = row_pinball(P["demand"].values, P[f"b{q}"].values, q)
    m = row_pinball(P["demand"].values, P[f"m{q}"].values, q)
    d = pd.DataFrame({"demand": P["demand"].values, "base": b, "model": m})
    d["gain"] = d["base"] - d["model"]
    d["gain_frac"] = np.where(d["base"] > 0, d["gain"] / d["base"], 0)
    d["dband"] = pd.qcut(d["demand"].rank(method="first"), 5,
                         labels=["v.low","low","mid","high","v.high"])

    print(f"\nq={q}")
    print(d.groupby("dband", observed=True).agg(
        n=("gain","size"),
        median_demand=("demand","median"),
        mean_abs_gain=("gain","mean"),
        mean_prop_gain=("gain_frac","mean"),
        share_improved=("gain", lambda s: (s > 0).mean())).round(3))
# %%
# median-row penalty: constant bias or noise?
for q in [0.75, 0.9]:
    b = row_pinball(P["demand"].values, P[f"b{q}"].values, q)
    m = row_pinball(P["demand"].values, P[f"m{q}"].values, q)
    worse = (b - m) < 0
    res_m = P["demand"].values - P[f"m{q}"].values
    res_b = P["demand"].values - P[f"b{q}"].values
    print(f"\nq={q}, rows where model is worse: {worse.sum()}")
    print(f"  model under-predicts (actual > pred) on {(res_m[worse] > 0).mean():.1%} of them")
    print(f"  base  under-predicts on {(res_b[worse] > 0).mean():.1%}")
    print(f"  median model pred {np.median(P[f'm{q}'].values[worse]):.0f} "
          f"vs base {np.median(P[f'b{q}'].values[worse]):.0f}")
# %%
def fit_fold_log(tr_idx, te_idx):
    """Same pipeline, but fit on log1p(demand) and back-transform."""
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    Xtr = Xall[m_tr]
    ytr = np.log1p(enc.loc[m_tr, "demand"].to_numpy(float))
    te = enc.loc[m_te].reset_index(drop=True)

    for q in QS:
        mdl = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=400,
                                        learning_rate=0.05, max_depth=4,
                                        min_samples_leaf=20, random_state=0)
        mdl.fit(Xtr, ytr)
        te[f"g{q}"] = np.maximum(np.expm1(mdl.predict(Xall[m_te])), 0)
    return te[["season","design_code"] + [f"g{q}" for q in QS]]

logp = pd.concat([fit_fold_log(tr, te) for tr, te in FOLDS], ignore_index=True)
P2 = P.merge(logp, on=["season","design_code"], how="left")
assert P2[[f"g{q}" for q in QS]].notna().all().all()
print(P2.shape)
# %%
rows = []
for q in QS:
    for col, name in [("b","base"), ("m","pieces"), ("g","log")]:
        rows.append({"q": q, "model": name,
                     "pinball": pinball(P2["demand"].values, P2[f"{col}{q}"].values, q),
                     "coverage": (P2["demand"].values <= P2[f"{col}{q}"].values).mean()})
print(pd.DataFrame(rows).pivot(index="q", columns="model",
                               values=["pinball","coverage"]).round(3))
# %%
# does the log model spread its gains more evenly?
for q in [0.75, 0.9]:
    b = row_pinball(P2["demand"].values, P2[f"b{q}"].values, q)
    g = row_pinball(P2["demand"].values, P2[f"g{q}"].values, q)
    d = pd.DataFrame({"demand": P2["demand"].values, "gain": b - g})
    d["dband"] = pd.qcut(d["demand"].rank(method="first"), 5,
                         labels=["v.low","low","mid","high","v.high"])
    print(f"\nq={q}")
    print(d.groupby("dband", observed=True)["gain"].agg(
        ["size","mean", lambda s: (s>0).mean()]).round(2))
# %%
keep = ["fold","season","design_code","brand","item_group","pattern","sleeve",
        "colour","rate","demand"] + [f"m{q}" for q in QS] + \
       [f"b{q}" for q in QS] + [f"g{q}" for q in QS]

con = sqlite3.connect("garment.db")
P2[keep].to_sql("component1_predictions", con, if_exists="replace", index=False)
con.close()
P2[keep].to_csv("data/clean/component1_predictions.csv", index=False)
print(P2[keep].shape, list(P2[keep].columns))
# %%
con = sqlite3.connect("garment.db")
chk = pd.read_sql("SELECT * FROM component1_predictions LIMIT 2", con)
n = pd.read_sql("SELECT COUNT(*) n FROM component1_predictions", con).iloc[0,0]
con.close()
print("rows:", n)
print("cols:", list(chk.columns))
print("\nhas g-columns:", all(f"g{q}" in chk.columns for q in QS))
# %%
con = sqlite3.connect("garment.db")
V = pd.read_sql("SELECT * FROM component1_predictions", con)
con.close()

print("rows:", len(V), "| expected 2375")
print("folds:", V["fold"].value_counts().sort_index().to_dict())
print("nulls:", V.isna().sum().sum())
print("\nseason == fold for every row:", (V["season"] == V["fold"]).all())
print("duplicate (season, design):", V.duplicated(["season","design_code"]).sum())
# %%
# monotonic quantiles within each policy?
for pre in ["b","m","g"]:
    arr = V[[f"{pre}{q}" for q in QS]].values
    bad = (np.diff(arr, axis=1) < -1e-9).any(axis=1).sum()
    print(f"{pre}: non-monotonic rows {bad}   negatives {(arr < 0).sum()}")
# %%
# do the numbers reproduce the results we reported?
def pb(y, p, q):
    e = y - p
    return np.mean(np.maximum(q*e, (q-1)*e))

for q in QS:
    print(f"q={q}:  base {pb(V['demand'], V[f'b{q}'], q):>8.1f}   "
          f"pieces {pb(V['demand'], V[f'm{q}'], q):>8.1f}   "
          f"log {pb(V['demand'], V[f'g{q}'], q):>8.1f}   |  "
          f"cov b {(V['demand']<=V[f'b{q}']).mean():.3f} "
          f"m {(V['demand']<=V[f'm{q}']).mean():.3f} "
          f"g {(V['demand']<=V[f'g{q}']).mean():.3f}")
# %%
# sanity: demand matches the source table
con = sqlite3.connect("garment.db")
src = pd.read_sql("SELECT season, design_code, demand FROM analysis_demand", con)
con.close()
chk = V[["season","design_code","demand"]].merge(src, on=["season","design_code"],
                                                  suffixes=("_pred","_src"))
print("rows matched:", len(chk), "of", len(V))
print("demand mismatches:", (chk["demand_pred"] != chk["demand_src"]).sum())
# %%
con = sqlite3.connect("garment.db")
V = pd.read_sql("SELECT * FROM component1_predictions", con)
con.close()

gcols = [f"g{q}" for q in QS]
V[gcols] = np.sort(V[gcols].values, axis=1)

arr = V[gcols].values
print("non-monotonic after sort:", (np.diff(arr, axis=1) < -1e-9).any(axis=1).sum())

for q in QS:
    print(f"q={q}: log pinball {pb(V['demand'], V[f'g{q}'], q):>8.1f}   "
          f"cov {(V['demand']<=V[f'g{q}']).mean():.3f}")

con = sqlite3.connect("garment.db")
V.to_sql("component1_predictions", con, if_exists="replace", index=False)
con.close()
V.to_csv("data/clean/component1_predictions.csv", index=False)
print("saved")
# %%
con = sqlite3.connect("garment.db")
V = pd.read_sql("SELECT * FROM component1_predictions", con)
si = pd.read_sql("SELECT design_code, season_introduced FROM dim_design", con)
con.close()

V = V.merge(si, on="design_code", how="left")
V["is_new"] = V["season"] == V["season_introduced"]
print("test rows:", len(V), " new:", V["is_new"].sum(), f"({V['is_new'].mean():.1%})")
print("carried forward:", (~V["is_new"]).sum())

for label, sub in [("ALL", V), ("NEW ONLY", V[V["is_new"]]), ("CARRIED", V[~V["is_new"]])]:
    print(f"\n--- {label}  (n={len(sub)}) ---")
    for q in [0.75, 0.9, 0.95]:
        b = pb(sub["demand"], sub[f"b{q}"], q)
        m = pb(sub["demand"], sub[f"m{q}"], q)
        g = pb(sub["demand"], sub[f"g{q}"], q)
        print(f"  q={q}: base {b:>8.1f}  pieces {m:>8.1f} ({100*(b-m)/b:+.1f}%)  "
              f"log {g:>8.1f} ({100*(b-g)/b:+.1f}%)")
# %%
from sklearn.ensemble import GradientBoostingRegressor
res = []
for seed in [0, 1, 2]:
    tot = 0
    for tr_idx, te_idx in FOLDS:
        m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
        mdl = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=400,
                                        learning_rate=0.05, max_depth=4,
                                        min_samples_leaf=20, random_state=seed)
        mdl.fit(Xall[m_tr], enc.loc[m_tr, "demand"].to_numpy(float))
        p = np.maximum(mdl.predict(Xall[m_te]), 0)
        tot += pinball(enc.loc[m_te, "demand"].to_numpy(float), p, 0.9) * m_te.sum()
    res.append(tot / len(enc[enc.si >= 5]))
print("q90 pinball across seeds:", [round(x,1) for x in res])

# %%
con = sqlite3.connect("garment.db")
V = pd.read_sql("SELECT * FROM component1_predictions", con)
con.close()

V = V[[c for c in V.columns if not c.startswith("hyb")]]

con = sqlite3.connect("garment.db")
V.to_sql("component1_predictions", con, if_exists="replace", index=False)
con.close()
V.to_csv("data/clean/component1_predictions.csv", index=False)

print(V.shape)
print(list(V.columns))
# %%
# %%
import pandas as pd, numpy as np, sqlite3

QS = [0.1, 0.5, 0.75, 0.9, 0.95]

con = sqlite3.connect("garment.db")
F   = pd.read_sql("SELECT * FROM component1_predictions", con) 
src = pd.read_sql("SELECT season, design_code, demand, n_samples_carried FROM analysis_demand", con)
dd  = pd.read_sql("SELECT design_code, season_introduced, main_group AS brand FROM dim_design", con)
con.close()

print("rows:", len(F), "| expect 2375")
print("nulls:", F.isna().sum().sum(), "| expect 0")
print("dupes (season,design):", F.duplicated(["season","design_code"]).sum(), "| expect 0")
print("season == fold:", (F["season"] == F["fold"]).all(), "| expect True")
print("hyb columns present:", any(c.startswith("hyb") for c in F.columns), "| expect False")
print("\nfolds:", F["fold"].value_counts().to_dict())
print("\ncolumns:", list(F.columns))
# %%
import os, sqlite3, pandas as pd
print("cwd:", os.getcwd())
print("garment.db here?", os.path.exists("garment.db"),
      os.path.getsize("garment.db") if os.path.exists("garment.db") else "")

con = sqlite3.connect("garment.db")
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con))
con.close()
# %%
# find every garment.db on the machine under your project root
for root, dirs, files in os.walk(r"C:\Users\USER\Downloads\Savla - Sero AI"):
    for f in files:
        if f.endswith(".db"):
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p):>12,}  {p}")
# %%
import os
os.remove(r"c:\Users\USER\AppData\Local\Programs\Microsoft VS Code\garment.db")
os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")
print("cwd:", os.getcwd())
print("db size:", os.path.getsize("garment.db"))
# %%
import os, gc, sqlite3

for o in gc.get_objects():
    if isinstance(o, sqlite3.Connection):
        try: o.close()
        except: pass

os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")
print("cwd:", os.getcwd())
print("db size:", os.path.getsize("garment.db"))

try:
    os.remove(r"c:\Users\USER\AppData\Local\Programs\Microsoft VS Code\garment.db")
    print("decoy deleted")
except PermissionError:
    print("decoy still locked - restart kernel and delete it manually in Explorer")
# %%
import pandas as pd
con = sqlite3.connect("garment.db")
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con))
con.close()
# %%
import os
os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")
# %%
# %%
import os, sqlite3
import pandas as pd, numpy as np
os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")

QS = [0.1, 0.5, 0.75, 0.9, 0.95]

con = sqlite3.connect("garment.db")
F   = pd.read_sql("SELECT * FROM component1_predictions", con)
src = pd.read_sql("SELECT season, design_code, demand FROM analysis_demand", con)
dd  = pd.read_sql("SELECT design_code, season_introduced FROM dim_design", con)
con.close()

print("rows:", len(F), "| expect 2375")
print("nulls:", F.isna().sum().sum(), "| expect 0")
print("dupes:", F.duplicated(["season","design_code"]).sum(), "| expect 0")
print("season==fold:", (F["season"] == F["fold"]).all(), "| expect True")
print("hyb present:", any(c.startswith("hyb") for c in F.columns), "| expect False")
print("\ncolumns:", list(F.columns))
# %%
# %%
chk = F[["season","design_code","demand"]].merge(src, on=["season","design_code"])
print("matched:", len(chk), "| expect 2375")
print("demand mismatches:", (chk["demand_x"] != chk["demand_y"]).sum(), "| expect 0")

for pre in ["b","m","g"]:
    a = F[[f"{pre}{q}" for q in QS]].values
    print(f"{pre}: crossings {(np.diff(a,axis=1) < -1e-9).any(axis=1).sum()}  "
          f"negatives {(a<0).sum()}   | expect 0, 0")
# %%
# %%
def pb(y, p, q):
    e = np.asarray(y, float) - np.asarray(p, float)
    return np.mean(np.maximum(q*e, (q-1)*e))

EXPECT = {0.1:(552.1,552.1,552.1), 0.5:(2719.3,2754.7,2704.4),
          0.75:(3602.1,3382.7,3263.0), 0.9:(2787.2,2288.5,2330.8),
          0.95:(1806.7,1495.9,1525.0)}

for q in QS:
    got = tuple(round(pb(F["demand"], F[f"{p}{q}"], q),1) for p in ["b","m","g"])
    e   = EXPECT[q]
    ok  = all(abs(g-x) < 0.6 for g,x in zip(got,e))
    print(f"q={q}: got {got}  expect {e}   {'OK' if ok else 'MISMATCH'}")

print("\ncoverage q0.9:", *[round((F['demand'] <= F[f'{p}0.9']).mean(),3) for p in "bmg"],
      "| expect 0.853 0.872 0.845")
# %%
# %%
if "is_new" not in F.columns:
    F = F.merge(dd, on="design_code", how="left")
    F["is_new"] = F["season"] == F["season_introduced"]
    print("(is_new rebuilt)")

print("new:", int(F["is_new"].sum()), "| expect 2226")
print("carried:", int((~F["is_new"]).sum()), "| expect 149")

EXP = {"NEW": {0.75:-13.9, 0.9:-5.7, 0.95:-6.5},
       "CARRIED": {0.75:28.2, 0.9:51.4, 0.95:53.1}}
for lab, sub in [("NEW", F[F.is_new]), ("CARRIED", F[~F.is_new])]:
    print(f"\n{lab} (n={len(sub)})")
    for q in [0.75, 0.9, 0.95]:
        b, m = pb(sub["demand"], sub[f"b{q}"], q), pb(sub["demand"], sub[f"m{q}"], q)
        got = 100*(b-m)/b
        print(f"  q={q}: {got:+6.1f}%   expect {EXP[lab][q]:+.1f}%   "
              f"{'OK' if abs(got-EXP[lab][q]) < 0.5 else 'MISMATCH'}")
# %%
F["is_new"] = F["is_new"].astype(bool)

print("new:", int(F["is_new"].sum()), "| expect 2226")
print("carried:", int((~F["is_new"]).sum()), "| expect 149")

EXP = {"NEW": {0.75:-13.9, 0.9:-5.7, 0.95:-6.5},
       "CARRIED": {0.75:28.2, 0.9:51.4, 0.95:53.1}}
for lab, sub in [("NEW", F[F["is_new"]]), ("CARRIED", F[~F["is_new"]])]:
    print(f"\n{lab} (n={len(sub)})")
    for q in [0.75, 0.9, 0.95]:
        b, m = pb(sub["demand"], sub[f"b{q}"], q), pb(sub["demand"], sub[f"m{q}"], q)
        got = 100*(b-m)/b
        print(f"  q={q}: {got:+6.1f}%   expect {EXP[lab][q]:+.1f}%   "
              f"{'OK' if abs(got-EXP[lab][q]) < 0.5 else 'MISMATCH'}")
# %%
EXPB = {"Fairdeal":24.9, "Formals":16.6, "Freezone Casuals/Aerix":9.3,
        "Kids":-0.2, "Marry Me":-6.6, "HV":-34.7}
for br, g in F.groupby("brand"):
    b, m = pb(g["demand"], g["b0.9"], 0.9), pb(g["demand"], g["m0.9"], 0.9)
    got = 100*(b-m)/b
    print(f"  {br:<24}{got:+7.1f}%   expect {EXPB[br]:+6.1f}%   "
          f"{'OK' if abs(got-EXPB[br]) < 0.5 else 'MISMATCH'}")
# %%
rows = []
for fold, g in F.groupby("fold"):
    for br, gb in g.groupby("brand"):
        rows.append({"brand": br,
                     "base":  pb(gb["demand"], gb["b0.9"], 0.9),
                     "model": pb(gb["demand"], gb["m0.9"], 0.9)})
pf = pd.DataFrame(rows).groupby("brand")[["base","model"]].mean()
pf["improv"] = (100*(pf["base"]-pf["model"])/pf["base"]).round(1)
print("mean-of-folds (matches the original numbers):")
print(pf["improv"])
# %%
con = sqlite3.connect("garment.db")
F.to_sql("component1_predictions", con, if_exists="replace", index=False)
con.close()
F.to_csv("data/clean/component1_predictions.csv", index=False)
print("saved", F.shape)
# %%
