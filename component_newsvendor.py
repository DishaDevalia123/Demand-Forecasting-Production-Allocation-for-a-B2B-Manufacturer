# %%
import pandas as pd, numpy as np, sqlite3

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

QS = [0.1, 0.5, 0.75, 0.9, 0.95]
SALVAGE = 0.68

con = sqlite3.connect("garment.db")
P2 = pd.read_sql("SELECT * FROM component1_predictions", con)
mk = pd.read_sql("""
    SELECT m.season, m.design_code, m.qty_unsold, m.markdown_pct, m.realised_value,
           d.rate, d.main_group AS brand
    FROM fact_inventory_markdown m JOIN dim_design d ON m.design_code = d.design_code
""", con)
dem = pd.read_sql("SELECT season, design_code, demand FROM analysis_demand", con)
con.close()

print("predictions:", P2.shape, " markdown:", mk.shape)
# %%
# %%
def critical_ratio(m, salvage=SALVAGE):
    """q* = Cu/(Cu+Co), Cu = m*rate, Co = (1-m)*rate - salvage*rate."""
    Co = (1 - m) - salvage
    return np.nan if Co <= 0 else m / (m + Co)

for m in np.arange(0.15, 0.36, 0.025):
    Co, q = (1 - m) - SALVAGE, critical_ratio(m)
    print(f"margin {m:.1%}   Cu={m:.3f}p   Co={Co:+.3f}p   "
          f"q* = {'produce to capacity' if np.isnan(q) else f'{q:.3f}'}")

# %%
print("total demand:", f"{P2['demand'].sum():,.0f}")
for q in QS:
    print(f"  q={q}:  base {P2[f'b{q}'].sum():>14,.0f}   "
          f"pieces {P2[f'm{q}'].sum():>14,.0f}   log {P2[f'g{q}'].sum():>14,.0f}")
# %%
POLICIES = {"base": "b", "pieces": "m", "log": "g"}

def quantity_at(P, q_target, prefix):
    """Linear interpolation between adjacent fitted quantiles."""
    xs = np.array(QS)
    Y = P[[f"{prefix}{q}" for q in QS]].values
    return np.array([np.interp(q_target, xs, row) for row in Y])


def expected_cost(P, q_target, prefix, m):
    """Realised newsvendor cost in rupees: Cu*shortfall + Co*leftover, per design."""
    Q = quantity_at(P, q_target, prefix)
    D = P["demand"].values
    rate = P["rate"].values

    Cu = m * rate
    Co = (0.32 - m) * rate

    shortfall = np.maximum(D - Q, 0)
    leftover  = np.maximum(Q - D, 0)
    return (Cu * shortfall + Co * leftover).sum()

# %%
# %%
def calibrated_q(P, q_target, prefix):
    """
    The model's stated quantiles are miscalibrated (its q90 covers ~87%).
    This inverts the empirical coverage curve: given a target coverage q_target,
    return the fitted quantile you must ASK for to actually achieve it.
    """
    cov = [(P["demand"].values <= P[f"{prefix}{q}"].values).mean() for q in QS]
    return float(np.interp(q_target, cov, QS))

# %%
rows = []
for m in np.arange(0.15, 0.30, 0.01):
    q = m / 0.32
    r = {"margin": round(m, 2), "q*": round(q, 3)}
    for name, pre in POLICIES.items():
        qc = calibrated_q(P2, q, pre)
        r[name] = expected_cost(P2, qc, pre, m) / 1e6
        r[f"{name}_q"] = round(qc, 3)
    r["best"] = min(POLICIES, key=lambda k: r[k])
    rows.append(r)

sweep_cal = pd.DataFrame(rows)
print(sweep_cal.round(2).to_string(index=False))
# %%
best = sweep_cal[(sweep_cal.margin >= 0.16) & (sweep_cal.margin <= 0.29)]
print("log vs base saving across the plausible range:")
print(((best["base"] - best["log"]) / best["base"] * 100).describe().round(2))
print("\nat 25% margin: base", round(best[best.margin==0.25]["base"].iloc[0],1),
      "-> log", round(best[best.margin==0.25]["log"].iloc[0],1))

# %%
# where does the saving come from - shortfall or leftover?
m, q = 0.25, 0.25/0.32
for name, pre in POLICIES.items():
    Q = quantity_at(P2, calibrated_q(P2, q, pre), pre)
    D, rate = P2["demand"].values, P2["rate"].values
    sf, lo = np.maximum(D-Q, 0), np.maximum(Q-D, 0)
    print(f"{name:<7} shortfall {sf.sum():>12,.0f} pcs (₹{(m*rate*sf).sum()/1e6:>7.1f}M)   "
          f"leftover {lo.sum():>12,.0f} pcs (₹{((0.32-m)*rate*lo).sum()/1e6:>7.1f}M)")
# %%
import os
os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")
# %%
P2["is_new"] = P2["is_new"].astype(bool)
# %%
mk["loss_pct_of_rate"] = (mk["qty_unsold"]*mk["rate"] - mk["realised_value"]) / \
                         (mk["qty_unsold"] * mk["rate"])
print("mean salvage fraction:", round(1 - mk["loss_pct_of_rate"].mean(), 3))
print((1 - mk.groupby("brand")["loss_pct_of_rate"].mean()).round(3))
# %%
m2 = mk.merge(dem, on=["season","design_code"], how="left")
m2["unsold_share"] = m2["qty_unsold"] / (m2["qty_unsold"] + m2["demand"])
print(m2[["qty_unsold","demand","unsold_share","markdown_pct"]].corr().round(3))
m2["sband"] = pd.qcut(m2["unsold_share"], 5, labels=["v.low","low","mid","high","v.high"])
print(m2.groupby("sband", observed=True)["markdown_pct"].agg(["size","mean"]).round(2))
# %%
# cost = (1-m)*rate, salvage = 0.68*rate
# Cu = rate - cost    = m*rate
# Co = cost - salvage = (0.32 - m)*rate
# q* = Cu/(Cu+Co) = m/0.32
# %%
NEW, CAR = P2[P2["is_new"]], P2[~P2["is_new"]]
print(f"new {len(NEW)}   carried {len(CAR)}")

for label, sub in [("ALL", P2), ("NEW", NEW), ("CARRIED", CAR)]:
    print(f"\n--- {label} (n={len(sub)}) ---")
    for m in [0.18, 0.22, 0.25, 0.28]:
        q = m / 0.32
        c = {}
        for name, pre in POLICIES.items():
            qc = calibrated_q(P2, q, pre)          # calibrate on the FULL set
            c[name] = expected_cost(sub, qc, pre, m) / 1e6
        b = c["base"]
        print(f"  m={m:.0%} q*={q:.2f}:  base {b:>7.1f}  "
              f"pieces {c['pieces']:>7.1f} ({100*(b-c['pieces'])/b:+5.1f}%)  "
              f"log {c['log']:>7.1f} ({100*(b-c['log'])/b:+5.1f}%)")
# %%
q = 0.25/0.32
for name, pre in POLICIES.items():
    Q = quantity_at(P2, calibrated_q(P2, q, pre), pre)
    print(f"{name:<7} median Q {np.median(Q):>8.0f}   "
          f"share under 12 pcs {(Q<12).mean():>6.1%}   "
          f"their share of total Q {Q[Q<12].sum()/Q.sum():.3%}")
# %%
rows = []
for m in np.arange(0.15, 0.30, 0.01):
    q = m / 0.32
    r = {"margin": round(m,2), "q*": round(q,3)}
    for name, pre in POLICIES.items():
        r[name] = expected_cost(P2, q, pre, m) / 1e6      # NO calibration
    r["best"] = min(POLICIES, key=lambda k: r[k])
    rows.append(r)
raw = pd.DataFrame(rows)
print(raw.round(2).to_string(index=False))
print("\nlog vs base saving %:")
print(((raw["base"] - raw["log"]) / raw["base"] * 100).describe().round(2))
# %%
m, q = 0.25, 0.25/0.32
Q = quantity_at(P2, calibrated_q(P2, q, "g"), "g")
d = pd.DataFrame({"demand": P2["demand"].values, "Q": Q, "rate": P2["rate"].values})
d["shortfall"] = np.maximum(d["demand"] - d["Q"], 0)
d["leftover"]  = np.maximum(d["Q"] - d["demand"], 0)
d["band"] = pd.qcut(d["demand"].rank(method="first"), 5,
                    labels=["v.low","low","mid","high","v.high"])

print(d.groupby("band", observed=True).agg(
    n=("demand","size"), demand=("demand","sum"),
    produced=("Q","sum"), shortfall=("shortfall","sum"),
    leftover=("leftover","sum")).astype(int))

g = d.groupby("band", observed=True)
print("\nshare of total shortfall:", (g["shortfall"].sum()/d["shortfall"].sum()).round(3).to_dict())
print("share of total leftover: ", (g["leftover"].sum()/d["leftover"].sum()).round(3).to_dict())
print("fill rate by band:", ((g["demand"].sum()-g["shortfall"].sum())/g["demand"].sum()).round(3).to_dict())
# %%
con = sqlite3.connect("garment.db")
cap = pd.read_sql("""
    SELECT SUBSTR(lot_id, 5, 4) AS code, SUM(qty_in) AS capacity
    FROM fact_production WHERE stage = 'Cutting'
    GROUP BY SUBSTR(lot_id, 5, 4)
""", con)
con.close()

SEASON_CODE = {"SU21":"Summer 2021","DI21":"Diwali 2021","CH21":"Christmas 2021",
               "PO22":"Pongal 2022","SU22":"Summer 2022","DI22":"Diwali 2022",
               "CH22":"Christmas 2022","PO23":"Pongal 2023","SU23":"Summer 2023",
               "DI23":"Diwali 2023"}
cap["season"] = cap["code"].map(SEASON_CODE)
cap = cap.dropna(subset=["season"]).set_index("season")["capacity"]

ctx = pd.DataFrame({"capacity": cap})
ctx["demand"] = P2.groupby("season")["demand"].sum()
ctx = ctx.dropna()
ctx["cap/demand"] = (ctx["capacity"] / ctx["demand"]).round(2)

# what the unconstrained newsvendor would want, at m=0.25
q = 0.25/0.32
P2["_Q"] = quantity_at(P2, q, "g")
ctx["wanted"] = P2.groupby("season")["_Q"].sum()
ctx["wanted/cap"] = (ctx["wanted"] / ctx["capacity"]).round(2)
print(ctx.astype(int, errors="ignore"))
# %%
print(ctx.round(2))
print("\ntotals:  capacity", f"{ctx['capacity'].sum():,.0f}",
      " demand", f"{ctx['demand'].sum():,.0f}",
      " wanted", f"{ctx['wanted'].sum():,.0f}")
# %%
def simulate_capped(df, q_target, prefix, m, cap_series, scale=True):
    """
    Per-season capacity ceiling. Newsvendor quantities scaled proportionally
    when they exceed capacity; left alone when they fit.
    """
    out = []
    for season, g in df.groupby("season"):
        Q = quantity_at(g, q_target, prefix)
        C = cap_series[season]
        factor = min(1.0, C / Q.sum()) if (scale and Q.sum() > 0) else 1.0
        Q = Q * factor

        D, rate = g["demand"].values, g["rate"].values
        sf, lo = np.maximum(D - Q, 0), np.maximum(Q - D, 0)
        out.append({
            "season": season, "scale_factor": factor,
            "produced": Q.sum(), "demand": D.sum(),
            "shortfall": sf.sum(), "leftover": lo.sum(),
            "fill_rate": 1 - sf.sum()/D.sum(),
            "deadstock_pcs": lo.sum(),
            "deadstock_value": (lo * rate * (1 - 0.684)).sum(),
            "cost": ((m*rate)*sf + ((0.316-m)*rate)*lo).sum(),
        })
    return pd.DataFrame(out)

m, q = 0.25, 0.25/0.316
for name, pre in POLICIES.items():
    r = simulate_capped(P2, q, pre, m, cap)
    print(f"\n=== {name} (q*={q:.3f}) ===")
    print(r.round(3).to_string(index=False))
    print(f"  TOTAL  fill {1 - r['shortfall'].sum()/r['demand'].sum():.3f}   "
          f"deadstock {r['deadstock_pcs'].sum():,.0f} pcs "
          f"(₹{r['deadstock_value'].sum()/1e6:.1f}M)   "
          f"cost ₹{r['cost'].sum()/1e6:.1f}M")
# %%
def simulate_oracle_alloc(df, m, cap_series):
    """Allocate the same capacity in proportion to each design's PREDICTED median
       (log model), not its q* quantity. Same total pieces, different split."""
    out = []
    for season, g in df.groupby("season"):
        w = quantity_at(g, 0.5, "g")                 # proportional to predicted median
        C = cap_series[season]
        Q = w / w.sum() * C if w.sum() > 0 else np.zeros(len(g))
        D, rate = g["demand"].values, g["rate"].values
        sf, lo = np.maximum(D-Q, 0), np.maximum(Q-D, 0)
        out.append({"season": season, "fill_rate": 1 - sf.sum()/D.sum(),
                    "deadstock_pcs": lo.sum(),
                    "cost": ((m*rate)*sf + ((0.316-m)*rate)*lo).sum()})
    return pd.DataFrame(out)

o = simulate_oracle_alloc(P2, 0.25, cap)
print(o.round(3).to_string(index=False))
print(f"TOTAL fill {1 - (P2['demand'].sum() - (o['fill_rate']*0).sum()):.3f}" if False else "")
print(f"  deadstock {o['deadstock_pcs'].sum():,.0f}   cost ₹{o['cost'].sum()/1e6:.1f}M")
# %%
def simulate_greedy(df, m, cap_series, prefix="g"):
    """
    Capacity allocated by marginal expected value.
    Each design's demand distribution is approximated by its fitted quantiles;
    a design's nth piece is worth Cu*P(D>=n) - Co*P(D<n).
    Equivalent to: allocate to the highest quantile level that still pays,
    then step down until capacity is exhausted.
    """
    Cu_f, Co_f = m, 0.316 - m
    out = []
    for season, g in df.groupby("season"):
        rate = g["rate"].values
        D = g["demand"].values
        C = cap_series[season]

        # binary search the critical ratio that exactly spends capacity
        lo_q, hi_q = 0.0, 0.999
        for _ in range(40):
            mid = (lo_q + hi_q) / 2
            Q = quantity_at(g, mid, prefix)
            if Q.sum() > C: hi_q = mid
            else:           lo_q = mid
        Q = quantity_at(g, lo_q, prefix)

        sf, le = np.maximum(D-Q, 0), np.maximum(Q-D, 0)
        out.append({"season": season, "q_used": round(lo_q,3),
                    "produced": Q.sum(), "fill_rate": 1 - sf.sum()/D.sum(),
                    "deadstock_pcs": le.sum(),
                    "cost": ((Cu_f*rate)*sf + (Co_f*rate)*le).sum()})
    return pd.DataFrame(out)

gr = simulate_greedy(P2, 0.25, cap, "g")
print(gr.round(3).to_string(index=False))
print(f"\nTOTAL fill {1 - (gr['fill_rate']*0).sum():.0f}" if False else "")
tot_sf = P2["demand"].sum() - sum(
    (gr.loc[gr.season==s,'fill_rate'].iloc[0] * P2[P2.season==s]['demand'].sum())
    for s in gr.season)
print(f"  fill {1 - tot_sf/P2['demand'].sum():.3f}   "
      f"deadstock {gr['deadstock_pcs'].sum():,.0f}   cost ₹{gr['cost'].sum()/1e6:.1f}M")
# %%
import os, sqlite3
import pandas as pd, numpy as np
os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

SEASONS = ["Summer 2021","Diwali 2021","Christmas 2021","Pongal 2022","Summer 2022",
           "Diwali 2022","Christmas 2022","Pongal 2023","Summer 2023","Diwali 2023"]
SIDX  = {s:i for i,s in enumerate(SEASONS)}
FOLDS = [(list(range(0,k)), k) for k in range(5,10)]

con = sqlite3.connect("garment.db")
df = pd.read_sql("""
    SELECT a.season, a.design_code, a.demand,
           d.main_group AS brand, d.item_group, d.pattern, d.sleeve,
           d.colour, d.rate, d.season_introduced
    FROM analysis_demand a JOIN dim_design d ON a.design_code = d.design_code
""", con)
con.close()

df["si"] = df["season"].map(SIDX)
df["festival"] = df["season"].str.split().str[0]
df["zero"] = (df["demand"] == 0).astype(int)
print("rows:", len(df), " zero rate:", round(df["zero"].mean(), 3))
# %%
def add_encodings(data):
    out = []
    for s in sorted(data.si.unique()):
        cur, hist = data[data.si == s].copy(), data[data.si < s]
        if len(hist) == 0:
            for c in ["enc_bp_mean","enc_bp_zero","enc_brand_lastyear"]:
                cur[c] = np.nan
            out.append(cur); continue
        bp = hist.groupby(["brand","pattern"])["demand"].agg(
            enc_bp_mean="mean", enc_bp_zero=lambda x:(x==0).mean()).reset_index()
        cur = cur.merge(bp, on=["brand","pattern"], how="left")
        h = hist.copy(); h["year"] = h["season"].str.split().str[1].astype(int)
        fy = h.groupby(["brand","festival","year"])["demand"].mean().reset_index(
            name="enc_brand_lastyear"); fy["year"] += 1
        cur["year"] = cur["season"].str.split().str[1].astype(int)
        cur = cur.merge(fy, on=["brand","festival","year"], how="left")
        out.append(cur)
    return pd.concat(out, ignore_index=True)

enc = add_encodings(df)
FEATS = ["brand","item_group","pattern","sleeve","colour","rate","festival",
         "enc_bp_mean","enc_bp_zero","enc_brand_lastyear"]
CATS  = ["brand","item_group","pattern","sleeve","colour","festival"]
X = pd.get_dummies(enc[FEATS], columns=CATS).fillna(-999)
print(X.shape)
# %%
rows = []
for tr_idx, te_idx in FOLDS:
    m_tr, m_te = enc.si.isin(tr_idx).values, (enc.si == te_idx).values
    ytr, yte = enc.loc[m_tr,"zero"].values, enc.loc[m_te,"zero"].values

    for name, mdl in [
        ("logistic", LogisticRegression(max_iter=2000, C=1.0)),
        ("gbm", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=3, random_state=0)),
    ]:
        mdl.fit(X[m_tr], ytr)
        p = mdl.predict_proba(X[m_te])[:,1]
        k = max(1, int(0.10*len(p)))
        top = np.argsort(p)[::-1][:k]
        rows.append({"fold": SEASONS[te_idx], "model": name,
                     "base_rate": yte.mean(),
                     "auc": roc_auc_score(yte, p),
                     "pr_auc": average_precision_score(yte, p),
                     "prec@10%": yte[top].mean()})

res = pd.DataFrame(rows)
print(res.groupby("model")[["base_rate","auc","pr_auc","prec@10%"]].mean().round(3))
print()
print(res.round(3).to_string(index=False))
# %%
con = sqlite3.connect("garment.db")
capfull = pd.read_sql("""
    SELECT SUBSTR(lot_id, 5, 4) AS code, SUM(qty_in) AS capacity
    FROM fact_production WHERE stage = 'Cutting'
    GROUP BY SUBSTR(lot_id, 5, 4)
""", con)
con.close()

SEASON_CODE = {"SU21":"Summer 2021","DI21":"Diwali 2021","CH21":"Christmas 2021",
               "PO22":"Pongal 2022","SU22":"Summer 2022","DI22":"Diwali 2022",
               "CH22":"Christmas 2022","PO23":"Pongal 2023","SU23":"Summer 2023",
               "DI23":"Diwali 2023"}
capfull["season"] = capfull["code"].map(SEASON_CODE)
capfull = capfull.dropna(subset=["season"]).set_index("season")["capacity"]
capfull = capfull.reindex(SEASONS)
print(capfull.astype(int))
# %%
# A: same festival last year, scaled by observed YoY growth in the two prior seasons
# B: prior season's capacity, scaled by the same growth rate
rows = []
for i, s in enumerate(SEASONS):
    if i < 4: 
        continue
    fest, yr = s.split()[0], int(s.split()[1])
    prev_same = f"{fest} {yr-1}"

    # growth observable at planning time: last two completed seasons
    g = capfull.iloc[i-1] / capfull.iloc[i-2]

    a = capfull.get(prev_same, np.nan) * g if prev_same in capfull.index else np.nan
    b = capfull.iloc[i-1] * g
    rows.append({"season": s, "actual": capfull.iloc[i],
                 "A_samefest_grown": a, "B_prevseason_grown": b,
                 "A_err%": 100*(a-capfull.iloc[i])/capfull.iloc[i],
                 "B_err%": 100*(b-capfull.iloc[i])/capfull.iloc[i]})
print(pd.DataFrame(rows).round(1).to_string(index=False))
# %%
rows = []
for mult in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    capx = capfull * mult
    for name, pre in POLICIES.items():
        r = simulate_capped(P2, 0.25/0.316, pre, 0.25, capx)
        rows.append({"cap_mult": mult, "policy": name,
                     "fill": 1 - r["shortfall"].sum()/r["demand"].sum(),
                     "deadstock_pcs": r["deadstock_pcs"].sum(),
                     "cost_M": r["cost"].sum()/1e6})
cs = pd.DataFrame(rows)
print(cs.pivot(index="cap_mult", columns="policy", values="fill").round(3))
print()
print(cs.pivot(index="cap_mult", columns="policy", values="cost_M").round(1))
# %%
r = simulate_capped(P2, 0.25/0.316, "g", 0.25, capfull)
lo_pcs = r["deadstock_pcs"].sum()
print(f"dead stock: {lo_pcs:,.0f} pcs")
print(f"  at full-rate value (rate - salvage, 0.316p): ₹{r['deadstock_value'].sum()/1e6:.1f}M")
# newsvendor overage cost only
m = 0.25
Q = np.concatenate([quantity_at(g, 0.25/0.316, "g")[
        :len(g)] * min(1.0, capfull[s]/quantity_at(g, 0.25/0.316, "g").sum())
    for s, g in P2.groupby("season")])
D = np.concatenate([g["demand"].values for _, g in P2.groupby("season")])
R = np.concatenate([g["rate"].values for _, g in P2.groupby("season")])
print(f"  at newsvendor overage cost ((0.316-m)p): ₹{(((0.316-m)*R)*np.maximum(Q-D,0)).sum()/1e6:.1f}M")
# %%
capB = {}
for i, s in enumerate(SEASONS):
    if i < 2:
        continue
    g = capfull.iloc[i-1] / capfull.iloc[i-2]
    capB[s] = capfull.iloc[i-1] * g
capB = pd.Series(capB)

print(pd.DataFrame({"actual": capfull, "forecast": capB,
                    "err%": (100*(capB-capfull)/capfull)}).round(1).dropna())

rows = []
for name, pre in POLICIES.items():
    r = simulate_capped(P2, 0.25/0.316, pre, 0.25, capB)
    rows.append({"policy": name,
                 "fill": 1 - r["shortfall"].sum()/r["demand"].sum(),
                 "deadstock_pcs": r["deadstock_pcs"].sum(),
                 "deadstock_fullrate_M": r["deadstock_value"].sum()/1e6,
                 "cost_M": r["cost"].sum()/1e6})
print()
print(pd.DataFrame(rows).round(3).to_string(index=False))
# %%
r = simulate_capped(P2, 0.25/0.316, "g", 0.25, capB)
rb = simulate_capped(P2, 0.25/0.316, "b", 0.25, capB)

for lab, rr in [("base", rb), ("log", r)]:
    ov = (0.316 - 0.25) / 0.316 * rr["deadstock_value"].sum()
    sh = rr["cost"].sum() - ov
    print(f"{lab:<6} overage ₹{ov/1e6:>7.1f}M + shortfall ₹{sh/1e6:>7.1f}M "
          f"= ₹{rr['cost'].sum()/1e6:>7.1f}M   (full-rate dead stock ₹{rr['deadstock_value'].sum()/1e6:.1f}M)")
# %%
import os
os.makedirs("docs", exist_ok=True)
# %%
# the capacity-multiplier sweep, for the robustness section
cs.to_csv("docs/component2_capacity_sweep.csv", index=False)

# the margin sweep, uncalibrated
raw.to_csv("docs/component2_margin_sweep.csv", index=False)
# %%
def expected_cost316(df, q_target, prefix, m):
    Q, D, rate = quantity_at(df, q_target, prefix), df["demand"].values, df["rate"].values
    return ((m*rate)*np.maximum(D-Q,0) + ((0.316-m)*rate)*np.maximum(Q-D,0)).sum()

rows = []
for m in np.arange(0.15, 0.30, 0.01):
    q = m / 0.316
    r = {"margin": round(m,2), "q*": round(q,3)}
    for name, pre in POLICIES.items():
        r[name] = expected_cost316(P2, q, pre, m) / 1e6
    r["best"] = min(POLICIES, key=lambda k: r[k])
    rows.append(r)
rawA = pd.DataFrame(rows)
print(rawA.round(2).to_string(index=False))
print("\nlog vs base saving %:")
print(((rawA["base"]-rawA["log"])/rawA["base"]*100).describe().round(2))

# A2 mechanism + A3 segments, same run, m=0.25
m, q = 0.25, 0.25/0.316
print(f"\n--- mechanism at m=0.25, q*={q:.3f} ---")
for name, pre in POLICIES.items():
    Q = quantity_at(P2, q, pre); D, R = P2["demand"].values, P2["rate"].values
    sf, lo = np.maximum(D-Q,0), np.maximum(Q-D,0)
    print(f"{name:<7} shortfall {sf.sum():>12,.0f} (₹{(m*R*sf).sum()/1e6:>6.1f}M)   "
          f"leftover {lo.sum():>12,.0f} (₹{((0.316-m)*R*lo).sum()/1e6:>6.1f}M)   "
          f"total ₹{expected_cost316(P2,q,pre,m)/1e6:.1f}M")

print("\n--- segments ---")
NEW, CAR = P2[P2["is_new"]], P2[~P2["is_new"]]
for lab, sub in [("ALL",P2),("NEW",NEW),("CARRIED",CAR)]:
    print(f"\n{lab} (n={len(sub)})")
    for mm in [0.18,0.22,0.25,0.28]:
        qq = mm/0.316
        b = expected_cost316(sub,qq,"b",mm)/1e6
        g = expected_cost316(sub,qq,"g",mm)/1e6
        p = expected_cost316(sub,qq,"m",mm)/1e6
        print(f"  m={mm:.0%} q*={qq:.2f}: base {b:>7.1f}  pieces {p:>7.1f} ({100*(b-p)/b:+5.1f}%)"
              f"  log {g:>7.1f} ({100*(b-g)/b:+5.1f}%)")
print(f"\ncarried share of ALL cost at m=0.25: "
      f"{expected_cost316(CAR,0.25/0.316,'b',0.25)/expected_cost316(P2,0.25/0.316,'b',0.25):.3f}")
# %%
for name, pre in POLICIES.items():
    rr = simulate_capped(P2, 0.25/0.316, pre, 0.25, capB)
    ov = (0.316-0.25)/0.316 * rr["deadstock_value"].sum()
    print(f"{name:<7} overage ₹{ov/1e6:>6.1f}M + shortfall ₹{(rr['cost'].sum()-ov)/1e6:>6.1f}M "
          f"= ₹{rr['cost'].sum()/1e6:>6.1f}M")

print("\n--- B4 allocation rules, forecastable ceiling ---")
prop = simulate_capped(P2, 0.25/0.316, "g", 0.25, capB)
print(f"proportional        fill {1-prop['shortfall'].sum()/prop['demand'].sum():.3f}  "
      f"dead {prop['deadstock_pcs'].sum():,.0f}  cost ₹{prop['cost'].sum()/1e6:.1f}M")

gr2 = simulate_greedy(P2, 0.25, capB, "g")
sf2 = sum((1-gr2.loc[gr2.season==s,'fill_rate'].iloc[0]) * P2[P2.season==s]['demand'].sum()
          for s in gr2.season)
print(f"Lagrangian (uniform) fill {1-sf2/P2['demand'].sum():.3f}  "
      f"dead {gr2['deadstock_pcs'].sum():,.0f}  cost ₹{gr2['cost'].sum()/1e6:.1f}M")

med = simulate_oracle_alloc(P2, 0.25, capB)
sfm = sum((1-med.loc[med.season==s,'fill_rate'].iloc[0]) * P2[P2.season==s]['demand'].sum()
          for s in med.season)
print(f"proportional-to-median fill {1-sfm/P2['demand'].sum():.3f}  "
      f"dead {med['deadstock_pcs'].sum():,.0f}  cost ₹{med['cost'].sum()/1e6:.1f}M")
# %%
m, q = 0.25, 0.25/0.316
Q = quantity_at(P2, q, "g")
d = pd.DataFrame({"demand": P2["demand"].values, "Q": Q, "rate": P2["rate"].values})
d["shortfall"] = np.maximum(d["demand"] - d["Q"], 0)
d["leftover"]  = np.maximum(d["Q"] - d["demand"], 0)
d["band"] = pd.qcut(d["demand"].rank(method="first"), 5,
                    labels=["v.low","low","mid","high","v.high"])

g = d.groupby("band", observed=True)
print(g.agg(n=("demand","size"), demand=("demand","sum"),
            produced=("Q","sum"), shortfall=("shortfall","sum"),
            leftover=("leftover","sum")).astype(int))
print("\nshare of shortfall:", (g["shortfall"].sum()/d["shortfall"].sum()).round(3).to_dict())
print("share of leftover: ", (g["leftover"].sum()/d["leftover"].sum()).round(3).to_dict())
print("fill by band:", ((g["demand"].sum()-g["shortfall"].sum())/g["demand"].sum()).round(3).to_dict())
# %%
rawA.to_csv("docs/component2_margin_sweep.csv", index=False)
cs.to_csv("docs/component2_capacity_sweep.csv", index=False)
pd.DataFrame({"actual": capfull, "forecast": capB}).to_csv("docs/component2_capacity.csv")
print("saved")
# %%
import os
os.makedirs("docs", exist_ok=True)
rawA.to_csv("docs/component2_margin_sweep.csv", index=False)
cs.to_csv("docs/component2_capacity_sweep.csv", index=False)
pd.DataFrame({"actual": capfull, "forecast": capB}).to_csv("docs/component2_capacity.csv")
print("saved")
# %%
