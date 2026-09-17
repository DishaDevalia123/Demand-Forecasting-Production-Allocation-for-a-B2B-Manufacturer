# %%
import os, sqlite3
import pandas as pd, numpy as np
os.chdir(r"C:\Users\USER\Downloads\Savla - Sero AI\garment-project")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

SEASONS = ["Summer 2021","Diwali 2021","Christmas 2021","Pongal 2022","Summer 2022",
           "Diwali 2022","Christmas 2022","Pongal 2023","Summer 2023","Diwali 2023"]
SIDX = {s:i for i,s in enumerate(SEASONS)}
SEASON_CODE = {"SU21":"Summer 2021","DI21":"Diwali 2021","CH21":"Christmas 2021",
               "PO22":"Pongal 2022","SU22":"Summer 2022","DI22":"Diwali 2022",
               "CH22":"Christmas 2022","PO23":"Pongal 2023","SU23":"Summer 2023",
               "DI23":"Diwali 2023"}
# %%
con = sqlite3.connect("garment.db")
fr = pd.read_sql("""
    WITH bk AS (SELECT order_no, design_code, party_code, season, SUM(qty) booked
                FROM fact_bookings GROUP BY 1,2,3,4),
         dp AS (SELECT order_no, design_code, SUM(qty_dispatched) shipped
                FROM fact_dispatch WHERE order_no != 'GENERAL' GROUP BY 1,2)
    SELECT bk.season, bk.party_code, bk.booked, COALESCE(dp.shipped,0) shipped
    FROM bk LEFT JOIN dp ON bk.order_no=dp.order_no AND bk.design_code=dp.design_code
""", con)
prod = pd.read_sql("""
    SELECT SUBSTR(lot_id,5,4) code, SUM(qty_in) produced
    FROM fact_production WHERE stage='Cutting' GROUP BY 1
""", con)
con.close()

prod["season"] = prod["code"].map(SEASON_CODE)
prod = prod.dropna(subset=["season"]).set_index("season")["produced"]

s = fr.groupby("season").agg(booked=("booked","sum"), shipped=("shipped","sum"))
s["fill_rate"]  = s["shipped"]/s["booked"]
s["zero_lines"] = fr.groupby("season").apply(lambda g: (g["shipped"]==0).mean())
s["produced"]   = prod
s["prod_ratio"] = s["produced"]/s["booked"]
s["festival"]   = [i.split()[0] for i in s.index]
print(s.reindex(SEASONS).round(3))
# %%
print("\nfill rate by festival:")
print(s.groupby("festival")["fill_rate"].agg(["size","mean","min","max"]).round(3))
# %%
# what differs in the 4 constrained seasons? start with WHO got starved
pf = fr.copy()
con = sqlite3.connect("garment.db")
pty = pd.read_sql("SELECT party_code, priority, region, first_season FROM dim_party", con)
con.close()

pf = pf.merge(pty, on="party_code", how="left")
ps = pf.groupby(["season","priority"]).agg(
        booked=("booked","sum"), shipped=("shipped","sum")).reset_index()
ps["fill"] = ps["shipped"]/ps["booked"]

tab = ps.pivot(index="season", columns="priority", values="fill").reindex(SEASONS).round(3)
tab["overall"] = s["fill_rate"].round(3)
tab["constrained"] = tab["overall"] < 0.80
print(tab)
# %%
# in constrained seasons, is the SHORTFALL concentrated on low priority?
ps["short"] = ps["booked"] - ps["shipped"]
ps = ps.merge(s[["fill_rate"]], left_on="season", right_index=True)
ps["constrained"] = ps["fill_rate"] < 0.80

print("\nshare of total shortfall borne by each tier:")
for c, g in ps.groupby("constrained"):
    sh = g.groupby("priority")["short"].sum() / g["short"].sum()
    bk = g.groupby("priority")["booked"].sum() / g["booked"].sum()
    print(f"\nconstrained={c}")
    print(pd.DataFrame({"share_of_shortfall": sh.round(3),
                        "share_of_bookings": bk.round(3),
                        "ratio": (sh/bk).round(2)}))
# %%
con = sqlite3.connect("garment.db")
gen = pd.read_sql("""
    SELECT SUBSTR(lot_id,5,4) code,
           SUM(CASE WHEN order_ref='GENERAL' THEN qty_in ELSE 0 END) AS general,
           SUM(qty_in) AS total
    FROM fact_production WHERE stage='Cutting' GROUP BY 1
""", con)
con.close()
gen["season"] = gen["code"].map(SEASON_CODE)
gen = gen.dropna(subset=["season"]).set_index("season")
gen["general_share"] = (gen["general"]/gen["total"]).round(3)
gen["fill_rate"] = s["fill_rate"].round(3)
gen["constrained"] = gen["fill_rate"] < 0.80
print(gen.reindex(SEASONS)[["general_share","fill_rate","constrained"]])
# %%
con = sqlite3.connect("garment.db")
bk = pd.read_sql("""SELECT party_code, season, SUM(total_amt) value, SUM(qty) units
                    FROM fact_bookings GROUP BY 1,2""", con)
p = pd.read_sql("SELECT * FROM dim_party", con)
con.close()

# (2) does priority ever change? -- settles classic DiD
print("parties with >1 priority value:", p.groupby("party_code")["priority"].nunique().gt(1).sum())

# (1) is there a running variable? -- settles RD
first = bk.merge(p[["party_code","first_season"]], on="party_code")
first = first[first["season"] == first["first_season"]]
rv = p.merge(first[["party_code","value","units"]], on="party_code", how="left")
print("\nfirst-season booking value by priority:")
print(rv.groupby("priority")["value"].describe()[["count","min","25%","50%","75%","max"]].round(0))
# %%
con = sqlite3.connect("garment.db")
bk = pd.read_sql("""SELECT party_code, season, SUM(total_amt) AS value, SUM(qty) AS units
                    FROM fact_bookings GROUP BY 1,2""", con)
pty = pd.read_sql("SELECT party_code, priority, region, first_season FROM dim_party", con)
con.close()

pan = bk.merge(pty, on="party_code", how="left")
pan["si"]       = pan["season"].map(SIDX)
pan["festival"] = pan["season"].str.split().str[0]
pan["year"]     = pan["season"].str.split().str[1].astype(int)

# season-level treatment, computed from fill rates in THAT season
cons = (s["fill_rate"] < 0.80)
pan["constrained"] = pan["season"].map(cons)

# same-festival YoY outcome: this season vs the same festival one year earlier
prev = pan.assign(year=pan["year"]+1)[["party_code","festival","year","value"]] \
          .rename(columns={"value":"prev_value"})
pan = pan.merge(prev, on=["party_code","festival","year"], how="inner")
pan["yoy"] = pan["value"]/pan["prev_value"] - 1

# TREATMENT LAG: the constraint the party experienced was in the PRIOR season,
# i.e. the season immediately before this booking
pan["prior_season"] = pan["si"].map(lambda i: SEASONS[i-1] if i > 0 else None)
pan["treated"] = pan["prior_season"].map(cons)

pan = pan.dropna(subset=["yoy","treated"])
pan = pan[pan["prior_season"] != "Diwali 2021"]          # exclude the ambiguous season

print("observations:", len(pan), " parties:", pan["party_code"].nunique())
print("\nby treatment x priority:")
print(pan.groupby(["treated","priority"]).agg(
    n=("yoy","size"), mean_yoy=("yoy","mean"), median_yoy=("yoy","median")).round(3))
# %%
print(pan.groupby(["prior_season","season"]).agg(
        n=("yoy","size"), treated=("treated","first"),
        mean_yoy=("yoy","mean")).round(3).sort_values("prior_season"))

print("\nby treated, per-season spread:")
print(pan.groupby("treated")["yoy"].describe().round(3))
# %%
print(pd.crosstab(pan["festival"], pan["treated"]))
print("\nseason pairs available:")
print(pan.groupby(["prior_season","season","treated"]).size())
# %%
pan2 = bk.merge(pty, on="party_code", how="left")
pan2["si"]       = pan2["season"].map(SIDX)
pan2["festival"] = pan2["season"].str.split().str[0]
pan2 = pan2.sort_values(["party_code","si"])

pan2["prev_value"] = pan2.groupby("party_code")["value"].shift()
pan2["prev_si"]    = pan2.groupby("party_code")["si"].shift()
pan2 = pan2[pan2["si"] - pan2["prev_si"] == 1]          # strictly consecutive only

pan2["yoy"] = pan2["value"]/pan2["prev_value"] - 1
pan2["prior_season"] = pan2["prev_si"].map(lambda i: SEASONS[int(i)])
pan2["treated"] = pan2["prior_season"].map(cons)
pan2 = pan2[pan2["prior_season"] != "Diwali 2021"].dropna(subset=["yoy","treated"])

print("observations:", len(pan2), " parties:", pan2["party_code"].nunique())
print("\nCOLLINEARITY CHECK — need both treated and untreated within festivals:")
print(pd.crosstab(pan2["festival"], pan2["treated"]))
# %%
# effective sample after festival FEs absorb the one-sided festivals
eff = pan2[pan2["festival"].isin(["Diwali","Pongal"])]
print("effective n:", len(eff), " parties:", eff["party_code"].nunique())
print(pd.crosstab([eff["festival"], eff["priority"]], eff["treated"]))
# %%
# outcome distribution - does it need a transform?
print(pan2.groupby("festival")["yoy"].describe().round(2))
print("\nlog1p version:")
pan2["log_yoy"] = np.log1p(pan2["yoy"])
print(pan2.groupby("festival")["log_yoy"].describe().round(2))
# %%
import statsmodels.formula.api as smf

d = pan2.copy()
d["log_yoy"] = np.log1p(d["yoy"])
d["treated"] = d["treated"].astype(int)
d["tier"]    = d["priority"].astype(int)
d["low"]     = (d["tier"] == 3).astype(int)

m1 = smf.ols("log_yoy ~ low*treated + C(festival) + C(party_code)", data=d) \
        .fit(cov_type="cluster", cov_kwds={"groups": d["party_code"]})
print(m1.summary().tables[1].as_text()[:1200])
print("\nlow:treated  coef %.4f   se %.4f   p %.3f   95%% CI [%.3f, %.3f]" % (
    m1.params["low:treated"], m1.bse["low:treated"], m1.pvalues["low:treated"],
    *m1.conf_int().loc["low:treated"]))
# %%
d2 = d.copy()
d2["y_dm"] = d2["log_yoy"] - d2.groupby("party_code")["log_yoy"].transform("mean")
d2["t_dm"] = d2["treated"] - d2.groupby("party_code")["treated"].transform("mean")
d2["i_dm"] = (d2["low"]*d2["treated"]) - \
             d2.groupby("party_code").apply(lambda g: (g["low"]*g["treated"]).mean()) \
               .reindex(d2["party_code"]).values

m2 = smf.ols("y_dm ~ i_dm + t_dm + C(festival)", data=d2) \
        .fit(cov_type="cluster", cov_kwds={"groups": d2["party_code"]})
print("i_dm coef %.4f  se %.4f  p %.3f  CI [%.3f, %.3f]" % (
    m2.params["i_dm"], m2.bse["i_dm"], m2.pvalues["i_dm"], *m2.conf_int().loc["i_dm"]))
print("\nn =", len(d2), " clusters =", d2['party_code'].nunique())
# %%
# 1. tier x festival interactions - does the estimate survive?
m3 = smf.ols("y_dm ~ i_dm + t_dm + C(festival)*low", data=d2) \
        .fit(cov_type="cluster", cov_kwds={"groups": d2["party_code"]})
print("with tier x festival: coef %.4f  CI [%.3f, %.3f]" % (
    m3.params["i_dm"], *m3.conf_int().loc["i_dm"]))

# 2. alternative treatment threshold (0.90 - includes Diwali 2021)
cons90 = (s["fill_rate"] < 0.90)
d3 = d2.copy()
d3["t90"] = d3["prior_season"].map(cons90).astype(int)
d3["i90"] = d3["low"]*d3["t90"]
for c in ["t90","i90"]:
    d3[c] = d3[c] - d3.groupby("party_code")[c].transform("mean")
m4 = smf.ols("y_dm ~ i90 + t90 + C(festival)", data=d3) \
        .fit(cov_type="cluster", cov_kwds={"groups": d3["party_code"]})
print("threshold 0.90:       coef %.4f  CI [%.3f, %.3f]" % (
    m4.params["i90"], *m4.conf_int().loc["i90"]))

# 3. tier 2 vs tier 1, as a dose check - should be smaller than tier 3
d2["mid"] = (d2["tier"]==2).astype(int)
d2["i_mid"] = d2["mid"]*d2["treated"]
d2["i_mid"] = d2["i_mid"] - d2.groupby("party_code")["i_mid"].transform("mean")
m5 = smf.ols("y_dm ~ i_dm + i_mid + t_dm + C(festival)", data=d2) \
        .fit(cov_type="cluster", cov_kwds={"groups": d2["party_code"]})
print("tier3 %.4f [%.3f, %.3f] | tier2 %.4f [%.3f, %.3f]" % (
    m5.params["i_dm"], *m5.conf_int().loc["i_dm"],
    m5.params["i_mid"], *m5.conf_int().loc["i_mid"]))

# %%
print(pd.crosstab(d3["prior_season"], [d3["t90"].round(2) != 0]))
print("\nseasons flagged by each threshold:")
print("0.80:", list(s.index[s['fill_rate'] < 0.80]))
print("0.90:", list(s.index[s['fill_rate'] < 0.90]))
print("\nin panel:", sorted(d2['prior_season'].unique()))
# %%
import numpy as np
rng = np.random.default_rng(0)

def estimate(dd, ycol="y_dm"):
    """Re-demean and fit. Returns coef, se, ci."""
    x = dd.copy()
    x["i_"] = x["low"]*x["treated"]
    x["t_"] = x["treated"]
    for c in ["i_","t_"]:
        x[c] = x[c] - x.groupby("party_code")[c].transform("mean")
    m = smf.ols(f"{ycol} ~ i_ + t_ + C(festival)", data=x) \
           .fit(cov_type="cluster", cov_kwds={"groups": x["party_code"]})
    lo, hi = m.conf_int().loc["i_"]
    return m.params["i_"], m.bse["i_"], lo, hi

# --- POSITIVE CONTROL: inject a known effect, see if we recover it
print("effect   recovered   se      CI                 detected")
for true_eff in [-0.05, -0.10, -0.15, -0.20, -0.30]:
    hits, coefs = 0, []
    for rep in range(200):
        x = d2.copy()
        noise = rng.normal(0, 0.0, len(x))          # no extra noise, just the injection
        x["y_inj"] = x["y_dm"] + true_eff * x["low"] * x["treated"]
        # re-demean the outcome after injection
        x["y_inj"] = x["y_inj"] - x.groupby("party_code")["y_inj"].transform("mean")
        c, se, lo, hi = estimate(x, "y_inj")
        coefs.append(c); hits += (hi < 0)
        if rep == 0: first = (c, se, lo, hi)
        break                                        # deterministic, one rep is enough
    c, se, lo, hi = first
    print(f"{true_eff:+.2f}    {c:+.4f}     {se:.4f}  [{lo:+.3f}, {hi:+.3f}]   "
          f"{'YES' if hi < 0 else 'no'}")
# %%
# --- PLACEBO B: shuffle priority across parties, keep seasons real
res = []
tiers = d2.groupby("party_code")["tier"].first()
for _ in range(1000):
    perm = pd.Series(rng.permutation(tiers.values), index=tiers.index)
    x = d2.copy()
    x["low"] = x["party_code"].map(perm).eq(3).astype(int)
    res.append(estimate(x)[0])
res = np.array(res)
print(f"PLACEBO B (shuffle priority): mean {res.mean():+.4f}  sd {res.std():.4f}")
print(f"  2.5/97.5 pct: [{np.percentile(res,2.5):+.3f}, {np.percentile(res,97.5):+.3f}]")
print(f"  real estimate -0.0232 sits at percentile {100*(res < -0.0232).mean():.1f}")
# %%
# --- PLACEBO A: shuffle which seasons are constrained (keep 4 of 8)
seas = sorted(d2["prior_season"].unique())
res = []
for _ in range(1000):
    fake = set(rng.choice(seas, 4, replace=False))
    x = d2.copy()
    x["treated"] = x["prior_season"].isin(fake).astype(int)
    try: res.append(estimate(x)[0])
    except Exception: pass
res = np.array(res)
print(f"PLACEBO A (shuffle seasons): mean {res.mean():+.4f}  sd {res.std():.4f}")
print(f"  2.5/97.5 pct: [{np.percentile(res,2.5):+.3f}, {np.percentile(res,97.5):+.3f}]")
print(f"  real estimate -0.0232 sits at percentile {100*(res < -0.0232).mean():.1f}")
# %%
# re-run placebos with retained arrays
resB, resA = [], []
tiers = d2.groupby("party_code")["tier"].first()
for _ in range(1000):
    perm = pd.Series(rng.permutation(tiers.values), index=tiers.index)
    x = d2.copy(); x["low"] = x["party_code"].map(perm).eq(3).astype(int)
    resB.append(estimate(x)[0])

seas = sorted(d2["prior_season"].unique())
for _ in range(1000):
    fake = set(rng.choice(seas, 4, replace=False))
    x = d2.copy(); x["treated"] = x["prior_season"].isin(fake).astype(int)
    try: resA.append(estimate(x)[0])
    except Exception: pass

pla = pd.DataFrame({"placebo_priority": pd.Series(resB),
                    "placebo_season":   pd.Series(resA)})

con = sqlite3.connect("garment.db")
d2.to_sql("component6_panel", con, if_exists="replace", index=False)
pla.to_sql("component6_placebo", con, if_exists="replace", index=False)
con.close()

os.makedirs("docs", exist_ok=True)
d2.to_csv("docs/component6_panel.csv", index=False)
pla.to_csv("docs/component6_placebo.csv", index=False)

# season-level fill rates and treatment definition
s.assign(constrained=s["fill_rate"] < 0.80).to_csv("docs/component6_seasons.csv")

# tier x season fill rate - the first stage
tab.to_csv("docs/component6_firststage.csv")
print("saved")
# %%
