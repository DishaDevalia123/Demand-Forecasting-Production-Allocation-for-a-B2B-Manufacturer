# %%
import sqlite3, os
import pandas as pd

con = sqlite3.connect("garment.db")

for f in sorted(os.listdir("data/clean")):
    name = f.replace(".csv", "")
    t = pd.read_csv(f"data/clean/{f}")
    t.to_sql(name, con, if_exists="replace", index=False)
    print(f"{name:<32} {t.shape}")

print("\ntables in db:")
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con))
con.close()
# %%
con = sqlite3.connect("garment.db")
print(pd.read_sql("""
    SELECT b.season, COUNT(*) AS lines, SUM(b.qty) AS booked_qty,
           COUNT(DISTINCT b.party_code) AS parties,
           COUNT(DISTINCT b.design_code) AS designs
    FROM fact_bookings b
    GROUP BY b.season
    ORDER BY MIN(b.order_date)
""", con))
con.close()
# %%
import sqlite3
con = sqlite3.connect("garment.db")

fill = pd.read_sql("""
WITH bk AS (
    SELECT order_no, design_code, party_code, season,
           SUM(qty) AS booked
    FROM fact_bookings
    GROUP BY order_no, design_code, party_code, season
),
dp AS (
    SELECT order_no, design_code, SUM(qty_dispatched) AS shipped
    FROM fact_dispatch
    WHERE order_no != 'GENERAL'
    GROUP BY order_no, design_code
)
SELECT bk.*, COALESCE(dp.shipped, 0) AS shipped
FROM bk LEFT JOIN dp
  ON bk.order_no = dp.order_no AND bk.design_code = dp.design_code
""", con)

fill["fill_rate"] = fill["shipped"] / fill["booked"]

print(len(fill), "order+design lines")
print("total booked:", fill["booked"].sum(), " shipped:", fill["shipped"].sum())
print("\naggregate fill rate:", round(fill["shipped"].sum()/fill["booked"].sum(), 4))
print("mean line-level fill rate:", round(fill["fill_rate"].mean(), 4))
print("\nfill_rate distribution:")
print(fill["fill_rate"].describe())
print("\nfully unfilled:", (fill["fill_rate"]==0).sum())
print("fully filled:", (fill["fill_rate"]>=1).sum())
# %%
# Fill rate
fill["size_band"] = pd.cut(fill["booked"], [0,50,200,1000,5000,10**9],
                           labels=["<50","50-200","200-1k","1k-5k","5k+"])

band = fill.groupby("size_band", observed=True).apply(
    lambda g: pd.Series({
        "lines":     len(g),
        "zero_rate": (g["fill_rate"] == 0).mean(),
        "mean_fill": g["fill_rate"].mean(),
        "agg_fill":  g["shipped"].sum() / g["booked"].sum(),
    }))
print(band.round(3))

# %%
f2 = fill.merge(pc[["party_code","priority"]], on="party_code", how="left")
prio = f2.groupby("priority").apply(
    lambda g: pd.Series({
        "lines":     len(g),
        "zero_rate": (g["fill_rate"] == 0).mean(),
        "mean_fill": g["fill_rate"].mean(),
        "agg_fill":  g["shipped"].sum() / g["booked"].sum(),
    }))
print(prio.round(3))
# %%
print(pd.crosstab(f2["priority"], f2["size_band"], normalize="index").round(3))

print("\nfill rate by priority WITHIN each size band:")
print(f2.groupby(["size_band","priority"], observed=True).apply(
    lambda g: pd.Series({"lines": len(g), "zero_rate": (g["fill_rate"]==0).mean()})
).round(3))
# %%
fill.drop(columns=["size_band"]).to_csv("data/clean/analysis_fill_rate.csv", index=False)
con = sqlite3.connect("garment.db")
fill.drop(columns=["size_band"]).to_sql("analysis_fill_rate", con, if_exists="replace", index=False)
con.close()
# %%
# bucketed demand distribution

con = sqlite3.connect("garment.db")
dem = pd.read_sql("""
    SELECT season, design_code, SUM(qty) AS demand
    FROM fact_bookings
    GROUP BY season, design_code
""", con)
con.close()

print(dem["demand"].describe())
print("\nrows:", len(dem))
# %%
con = sqlite3.connect("garment.db")
full = pd.read_sql("""
    SELECT s.season, s.design_code, s.brand,
           s.n_samples_carried,
           COALESCE(b.demand, 0) AS demand
    FROM fact_samples s
    LEFT JOIN (SELECT season, design_code, SUM(qty) AS demand
               FROM fact_bookings GROUP BY season, design_code) b
      ON s.season = b.season AND s.design_code = b.design_code
""", con)
con.close()

print("design-seasons carried:", len(full))
print("zero demand:", (full["demand"]==0).sum(), f"({(full['demand']==0).mean():.1%})")
print("\nwith zeros:")
print(full["demand"].describe())

buckets = pd.cut(full["demand"], [-1,0,25,50,75,100,10**9],
                 labels=["0","1-25","26-50","51-75","76-100","100+"])
print("\nbucketed (your Savla table):")
print(buckets.value_counts().sort_index())
# %%
con = sqlite3.connect("garment.db")
print(pd.read_sql("""
    SELECT COUNT(*) AS booked_not_carried FROM (
        SELECT DISTINCT season, design_code FROM fact_bookings
        EXCEPT
        SELECT season, design_code FROM fact_samples)
""", con))
con.close()
# %%
bk = pd.crosstab(full["brand"], buckets, normalize="index")
print((bk*100).round(1))
# %%
con = sqlite3.connect("garment.db")
full["demand_bucket"] = buckets
full.to_sql("analysis_demand", con, if_exists="replace", index=False)
full.to_csv("data/clean/analysis_demand.csv", index=False)
con.close()
print(full.shape)
# %%
con = sqlite3.connect("garment.db")
util = pd.read_sql("""
    SELECT season, brand,
           COUNT(*) AS carried,
           SUM(CASE WHEN demand > 0 THEN 1 ELSE 0 END) AS booked,
           ROUND(1.0*SUM(CASE WHEN demand > 0 THEN 1 ELSE 0 END)/COUNT(*), 3) AS util_ratio
    FROM analysis_demand
    GROUP BY season, brand
""", con)
con.close()
print(util.pivot(index="season", columns="brand", values="util_ratio"))
# %%
con = sqlite3.connect("garment.db")
util.to_sql("analysis_utilization", con, if_exists="replace", index=False)
con.close()
# %%
con = sqlite3.connect("garment.db")
sos = pd.read_sql("""
    SELECT b.season, d.main_group AS brand, SUM(b.qty) AS qty
    FROM fact_bookings b JOIN dim_design d ON b.design_code = d.design_code
    GROUP BY b.season, d.main_group
""", con)
con.close()

order = ["Summer 2021","Diwali 2021","Christmas 2021","Pongal 2022","Summer 2022",
         "Diwali 2022","Christmas 2022","Pongal 2023","Summer 2023","Diwali 2023"]
piv = sos.pivot(index="season", columns="brand", values="qty").reindex(order)
print(piv)
print("\nseason-over-season % change:")
print((piv.pct_change()*100).round(1))
# %%
con = sqlite3.connect("garment.db")
sos_val = pd.read_sql("""
    SELECT b.season, d.main_group AS brand, SUM(b.total_amt) AS value
    FROM fact_bookings b JOIN dim_design d ON b.design_code = d.design_code
    GROUP BY b.season, d.main_group
""", con)
con.close()

pv = sos_val.pivot(index="season", columns="brand", values="value").reindex(order)
print((pv/1e6).round(2))
print("\nshare of total revenue %:")
print((pv.div(pv.sum(axis=1), axis=0)*100).round(1))
# %%
fest = piv.copy()
fest["festival"] = [s.split()[0] for s in fest.index]
fest["year"] = [int(s.split()[1]) for s in fest.index]
for b in ["Fairdeal","Formals","Kids"]:
    t = fest.pivot(index="festival", columns="year", values=b)
    print(f"\n{b}:"); print(t)
# %%
con = sqlite3.connect("garment.db")
sos.to_sql("analysis_sos_units", con, if_exists="replace", index=False)
sos_val.to_sql("analysis_sos_value", con, if_exists="replace", index=False)
con.close()

# %%
con = sqlite3.connect("garment.db")

pat = pd.read_sql("""
    SELECT d.main_group AS brand, d.pattern,
           COUNT(DISTINCT a.design_code || a.season) AS carried,
           SUM(CASE WHEN a.demand > 0 THEN 1 ELSE 0 END) AS booked,
           SUM(a.demand) AS units
    FROM analysis_demand a
    JOIN dim_design d ON a.design_code = d.design_code
    GROUP BY d.main_group, d.pattern
""", con)
con.close()

pat["util"] = (pat["booked"] / pat["carried"]).round(3)
pat["units_per_carried"] = (pat["units"] / pat["carried"]).round(0)

print("utilization by brand x pattern:")
print(pat.pivot(index="pattern", columns="brand", values="util"))
print("\nunits per design carried:")
print(pat.pivot(index="pattern", columns="brand", values="units_per_carried"))
# %%
con = sqlite3.connect("garment.db")

pat = pd.read_sql("""
    SELECT d.main_group AS brand, d.pattern,
           COUNT(*) AS carried,
           SUM(CASE WHEN a.demand > 0 THEN 1 ELSE 0 END) AS booked,
           SUM(CASE WHEN a.demand = 0 THEN 1 ELSE 0 END) AS zero_designs,
           SUM(a.demand) AS units
    FROM analysis_demand a
    JOIN dim_design d ON a.design_code = d.design_code
    GROUP BY d.main_group, d.pattern
""", con)
con.close()

pat["util"]      = (pat["booked"] / pat["carried"]).round(3)
pat["zero_rate"] = (pat["zero_designs"] / pat["carried"]).round(3)
pat["units_per_carried"] = (pat["units"] / pat["carried"]).round(0)

print("ZERO RATE by brand x pattern:")
print(pat.pivot(index="pattern", columns="brand", values="zero_rate"))
print("\nUNITS PER DESIGN CARRIED (incl. zeros):")
print(pat.pivot(index="pattern", columns="brand", values="units_per_carried"))
print("\nDESIGNS CARRIED (sample size per cell):")
print(pat.pivot(index="pattern", columns="brand", values="carried"))
# %%
con = sqlite3.connect("garment.db")
pat.to_sql("analysis_brand_pattern", con, if_exists="replace", index=False)
con.close()
# %%
con = sqlite3.connect("garment.db")
reg = pd.read_sql("""
    SELECT b.season, p.region, c.region_relevance,
           SUM(b.qty) AS units,
           COUNT(DISTINCT b.party_code) AS parties
    FROM fact_bookings b
    JOIN dim_party p ON b.party_code = p.party_code
    JOIN dim_calendar c ON c.season = b.season AND c.region = p.region
    GROUP BY b.season, p.region, c.region_relevance
""", con)
con.close()

reg["units_per_party"] = (reg["units"] / reg["parties"]).round(0)
print(reg.groupby("region_relevance").agg(
    cells=("units","size"), total_units=("units","sum"),
    avg_units_per_party=("units_per_party","mean")).round(0))

print("\nunits per party, season x region:")
print(reg.pivot(index="season", columns="region", values="units_per_party"))
# %%
con = sqlite3.connect("garment.db")
psz = pd.read_sql("""
    SELECT p.region, p.party_code, p.priority, SUM(b.qty) AS total_units
    FROM fact_bookings b JOIN dim_party p ON b.party_code = p.party_code
    GROUP BY p.region, p.party_code, p.priority
""", con)
con.close()

print(psz.groupby("region").agg(
    parties=("party_code","size"),
    total=("total_units","sum"),
    median_party=("total_units","median"),
    max_party=("total_units","max")).round(0).sort_values("total", ascending=False))
# %%
w = reg.merge(reg.groupby("region")["units"].mean().rename("region_avg"), on="region")
w["vs_region_avg"] = (w["units"] / w["region_avg"]).round(2)
print(w.groupby("region_relevance")["vs_region_avg"].describe().round(2))
# %%
con = sqlite3.connect("garment.db")
reg.to_sql("analysis_region_season", con, if_exists="replace", index=False)
con.close()
# %%
con = sqlite3.connect("garment.db")
pty = pd.read_sql("""
    SELECT b.party_code, p.priority, p.region, p.first_season,
           SUM(b.qty) AS units, SUM(b.total_amt) AS value,
           COUNT(DISTINCT b.season) AS seasons_active
    FROM fact_bookings b JOIN dim_party p ON b.party_code = p.party_code
    GROUP BY b.party_code, p.priority, p.region, p.first_season
""", con)
con.close()

pty = pty.sort_values("value", ascending=False)
pty["cum_share"] = (pty["value"].cumsum() / pty["value"].sum()).round(3)

print("top 10 parties:")
print(pty.head(10)[["party_code","priority","region","units","value","seasons_active","cum_share"]].to_string(index=False))
print("\ntop 5 share:", pty["cum_share"].iloc[4])
print("top 10 share:", pty["cum_share"].iloc[9])
print("top 20 share:", pty["cum_share"].iloc[19])
# %%
con = sqlite3.connect("garment.db")
gaps = pd.read_sql("""
    SELECT a.party_code, a.season, a.attended,
           COALESCE(b.units,0) AS units
    FROM fact_fair_attendance a
    LEFT JOIN (SELECT party_code, season, SUM(qty) AS units
               FROM fact_bookings GROUP BY party_code, season) b
      ON a.party_code=b.party_code AND a.season=b.season
""", con)
con.close()

print(pd.crosstab(gaps["attended"], gaps["units"]>0))
# %%
n = gaps[gaps["attended"]=="N"].merge(pty[["party_code","priority","value"]], on="party_code", how="left")
print("non-attendance by season:")
print(n["season"].value_counts().sort_index())
print("\nby priority:")
print(n["priority"].value_counts().sort_index())
print("\nparties with multiple absences:")
print(n["party_code"].value_counts().head(10))
# %%
n2 = gaps[gaps["attended"]=="N"].merge(pc[["party_code","region"]], on="party_code")
n2 = n2.merge(cc[["season","region","region_relevance"]].drop_duplicates(), on=["season","region"], how="left")
print(n2["region_relevance"].value_counts())

# compare against the base rate of eligible party-seasons by relevance
allg = gaps.merge(pc[["party_code","region"]], on="party_code")
allg = allg.merge(cc[["season","region","region_relevance"]].drop_duplicates(), on=["season","region"], how="left")
print("\nabsence rate by region_relevance:")
print(allg.groupby("region_relevance")["attended"].apply(lambda s: (s=="N").mean()).round(3))
# %%
last = gaps[gaps["units"]>0].groupby("party_code")["season"].apply(
    lambda s: max(order.index(x) for x in s))
print("parties whose last booking was before Diwali 2023:")
print(last[last < 9].sort_values().to_dict())
# %%
con = sqlite3.connect("garment.db")
ps = pd.read_sql("""
    SELECT party_code, season, SUM(total_amt) AS value, SUM(qty) AS units
    FROM fact_bookings GROUP BY party_code, season
""", con)
con.close()

ps["si"] = ps["season"].map({s:i for i,s in enumerate(order)})
ps = ps.sort_values(["party_code","si"])
ps["prev_value"] = ps.groupby("party_code")["value"].shift()
ps["pct_change"] = (ps["value"]/ps["prev_value"] - 1)

print(ps["pct_change"].describe().round(3))
print("\npercentiles:")
print(ps["pct_change"].quantile([.05,.1,.25,.5,.75,.9,.95]).round(3))
print("\nshare of transitions below various thresholds:")
for t in [-0.7,-0.5,-0.3,-0.2]:
    print(f"  drop worse than {int(t*100)}%: {(ps['pct_change']<t).mean():.1%}")
# %%
ps["festival"] = ps["season"].str.split().str[0]
print(ps.groupby("festival")["pct_change"].agg(["count","median","mean"]).round(3))
print("\ndrops worse than -30%, by festival:")
print(ps[ps["pct_change"] < -0.3]["festival"].value_counts())
# %%
ps["year"] = ps["season"].str.split().str[1].astype(int)
yoy = ps.merge(ps.assign(year=ps["year"]+1)[["party_code","festival","year","value"]]
                 .rename(columns={"value":"prev_year_value"}),
               on=["party_code","festival","year"], how="inner")
yoy["yoy_change"] = yoy["value"]/yoy["prev_year_value"] - 1
print("\nYoY same-festival transitions:", len(yoy))
print(yoy["yoy_change"].quantile([.05,.1,.25,.5,.75,.9]).round(3))
for t in [-0.5,-0.3,-0.2]:
    print(f"  drop worse than {int(t*100)}%: {(yoy['yoy_change']<t).mean():.1%}  (n={(yoy['yoy_change']<t).sum()})")
# %%
con = sqlite3.connect("garment.db")
lines = pd.read_sql("""
    SELECT b.party_code, b.season, d.main_group AS brand, d.pattern,
           SUM(b.total_amt) AS value, SUM(b.qty) AS units
    FROM fact_bookings b JOIN dim_design d ON b.design_code = d.design_code
    GROUP BY b.party_code, b.season, d.main_group, d.pattern
""", con)
con.close()

def yoy_at(df, keys):
    x = df.copy()
    x["festival"] = x["season"].str.split().str[0]
    x["year"] = x["season"].str.split().str[1].astype(int)
    k = keys + ["festival", "year"]
    prev = x.assign(year=x["year"]+1)[k + ["value"]].rename(columns={"value":"prev"})
    m = x.merge(prev, on=k, how="inner")
    m["chg"] = m["value"]/m["prev"] - 1
    return m

for name, keys, src in [
    ("party-season",        ["party_code"],            lines.groupby(["party_code","season"], as_index=False)["value"].sum()),
    ("party-season-brand",  ["party_code","brand"],    lines.groupby(["party_code","season","brand"], as_index=False)["value"].sum()),
    ("party-season-pattern",["party_code","pattern"],  lines.groupby(["party_code","season","pattern"], as_index=False)["value"].sum()),
]:
    m = yoy_at(src, keys)
    print(f"\n{name}: {len(m)} YoY transitions")
    print("  median chg:", round(m["chg"].median(),3))
    for t in [-0.5,-0.3,-0.2]:
        print(f"  below {int(t*100)}%: {(m['chg']<t).sum():5d}  ({(m['chg']<t).mean():.1%})")
# %%
con = sqlite3.connect("garment.db")
lt = pd.read_sql("""
    SELECT stage, jobworker_type,
           COUNT(*) AS lots,
           ROUND(AVG(expected_days),2) AS exp_days,
           ROUND(AVG(actual_days),2) AS act_days,
           ROUND(AVG(actual_days - expected_days),2) AS overrun,
           ROUND(AVG(1.0*qty_out/qty_in),4) AS yield
    FROM fact_production
    GROUP BY stage, jobworker_type
""", con)
con.close()
print(lt.pivot(index="stage", columns="jobworker_type", values=["act_days","overrun","yield"]).round(3))
# %%
con = sqlite3.connect("garment.db")
print(pd.read_sql("""
    SELECT jobworker_type, stage,
           COUNT(*) AS n, ROUND(AVG(qty_in),0) AS avg_qty,
           ROUND(AVG(actual_days/expected_days),3) AS ratio
    FROM fact_production GROUP BY jobworker_type, stage
""", con).pivot(index="stage", columns="jobworker_type", values=["avg_qty","ratio"]))
con.close()
# %%
con = sqlite3.connect("garment.db")
p2 = pd.read_sql("SELECT * FROM fact_production", con)
con.close()

p2["ratio"] = p2["actual_days"] / p2["expected_days"]
p2["qband"] = pd.qcut(p2["qty_in"], 5, labels=["tiny","small","mid","large","huge"])
print(p2.groupby(["qband","jobworker_type"], observed=True)["ratio"].agg(["size","mean"]).round(3).unstack())
# %%
con = sqlite3.connect("garment.db")
lt.to_sql("analysis_leadtime", con, if_exists="replace", index=False)
con.close()
# %%
