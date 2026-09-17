# %%
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)

RAW = "data/raw"
p = pd.read_csv(f"{RAW}/dim_party.csv")
b = pd.read_csv(f"{RAW}/fact_bookings.csv")

print(p.shape, b.shape)
# %%
p.head()
# %%
b.head()
# %%
summary = pd.DataFrame({
    "dtype":    p.dtypes.astype(str),
    "nulls":    p.isna().sum(),
    "n_unique": p.nunique(),
    "sample":   [p[c].dropna().iloc[0] if p[c].notna().any() else None for c in p.columns],
})
summary

# %%
for c in p.columns:
    print(f"\n--- {c}  ({p[c].nunique()} distinct) ---")
    print(p[c].value_counts(dropna=False).head(25))

# %%
p["priority"].value_counts()

# %%
# is the encoding style random, or does it follow something?
p["enc"] = p["priority"].astype(str).str.strip().str.isdigit().map({True: "numeric", False: "text"})
pd.crosstab(p["first_season"], p["enc"])

# %%
p[p["party_code"].str.len() != 8]

# %%
base = ["PTY-0021", "PTY-0022", "PTY-0030"]
p[p["party_code"].str.replace("-DUP", "", regex=False).isin(base)].sort_values("party_code")
# %%
codes = base + [c + "-DUP" for c in base]
b[b["party_code"].isin(codes)]["party_code"].value_counts()

# %%
import glob, os
for f in sorted(glob.glob(f"{RAW}/*.csv")):
    cols = pd.read_csv(f, nrows=0).columns
    if "party_code" in cols:
        d = pd.read_csv(f)
        hits = d[d["party_code"].isin([c + "-DUP" for c in base])].shape[0]
        print(f"{os.path.basename(f):<30} rows referencing -DUP: {hits}")

# %%
p["name_clean"] = p["party_name"].str.strip()
dupes = p[p["name_clean"].duplicated(keep=False)].sort_values(["name_clean", "party_code"])
dupes[["party_code","party_name","priority","place","region","first_season"]]

# %%
act = (b.groupby("party_code")
         .agg(lines=("qty","size"), qty=("qty","sum"),
              seasons=("season", lambda s: sorted(set(s))))
         .reset_index())

chk = dupes.merge(act, on="party_code", how="left")
chk["lines"] = chk["lines"].fillna(0).astype(int)
chk["n_seasons"] = chk["seasons"].apply(lambda x: len(x) if isinstance(x, list) else 0)

for name, g in chk.groupby("name_clean"):
    print(f"\n=== {name} ===")
    for _, r in g.iterrows():
        print(f"  {r.party_code:<14} first={r.first_season:<15} prio={str(r.priority):<7} "
              f"lines={r.lines:<5} seasons={r.n_seasons}  qty={r.qty}")

# %%
p["region"].value_counts(dropna=False)

# %%
pd.crosstab(p["place"], p["region"].fillna("(blank)"))

# Cleaning dim_party.csv

# %%
REGION_MAP = {
    "delhi ncr": "Delhi NCR",
    "gujarat": "Gujarat",
    "gj": "Gujarat",
    "kerala": "Kerala",
    "kl": "Kerala",
    "karnataka": "Karnataka",
    "goa": "Goa",
    "maharashtra": "Maharashtra",
    "north east": "North East",
    "ne": "North East",
    "tamil nadu": "Tamil Nadu",
}

PRIORITY_MAP = {"High": 1, "Medium": 2, "Low": 3}

pc = p.copy()

# 1. strip whitespace from every text column
for c in pc.select_dtypes("object").columns:
    pc[c] = pc[c].str.strip()

# 2. drop the -DUP rows
pc = pc[~pc["party_code"].str.endswith("-DUP")]

# 3. priority -> numeric
pc["priority"] = pd.to_numeric(
    pc["priority"].replace(PRIORITY_MAP), errors="raise"
).astype("Int64")

# 4. region -> canonical, then fill blanks from place
pc["region"] = pc["region"].str.lower().map(REGION_MAP)
place_to_region = (pc.dropna(subset=["region"])
                     .groupby("place")["region"]
                     .agg(lambda s: s.mode().iloc[0]))
pc["region"] = pc["region"].fillna(pc["place"].map(place_to_region))

pc = pc.drop(columns=[c for c in ["enc", "name_clean"] if c in pc.columns])

# --- verify
print("rows:", len(pc))
print("priority:", sorted(pc["priority"].dropna().unique()))
print("regions:", sorted(pc["region"].unique()))
print("nulls:\n", pc.isna().sum())

# %%
import os
os.makedirs("data/clean", exist_ok=True)
pc.to_csv("data/clean/dim_party.csv", index=False)

# dim_design.csv
# %%
d = pd.read_csv(f"{RAW}/dim_design.csv")

pd.DataFrame({
    "dtype": d.dtypes.astype(str),
    "nulls": d.isna().sum(),
    "n_unique": d.nunique(),
    "sample": [d[c].dropna().iloc[0] if d[c].notna().any() else None for c in d.columns],
})

# %%
for c in ["item_group","main_group","pattern","sleeve","colour","season_introduced"]:
    print(f"\n--- {c} ({d[c].nunique()}) ---")
    print(d[c].value_counts(dropna=False))

# %%
print(d["rate"].describe())
print("\ndesign_code lengths:")
print(d["design_code"].str.len().value_counts())
print("\nduplicate design_code:", d["design_code"].duplicated().sum())

# %%
print(d[d["design_code"].str.len() == 9]["design_code"].head(10).tolist())
print(d[d["design_code"].str.len() == 10]["design_code"].head(10).tolist())

# prefix (letters before the dash) by code length
d["prefix"] = d["design_code"].str.split("-").str[0]
print(pd.crosstab(d["prefix"], d["design_code"].str.len()))

# %%
d.groupby("main_group")["rate"].describe()[["min","25%","50%","75%","max"]]

# %%
d.drop(columns=["prefix"]).to_csv("data/clean/dim_design.csv", index=False)

# dim_calender.csv
# %%
cal = pd.read_csv(f"{RAW}/dim_calendar.csv")
print(cal.shape)
cal.head(20)

# %%
for c in cal.columns:
    print(f"\n--- {c} ({cal[c].nunique()}) ---")
    print(cal[c].value_counts(dropna=False).head(20))

# %%
cc = cal.copy()

cc["fair_date"] = pd.to_datetime(cc["fair_date"], errors="raise")

# split "Mar 2021-May 2021" into two halves
parts = cc["delivery_window"].str.split("-", expand=True)
cc["delivery_start"] = pd.to_datetime(parts[0].str.strip(), format="%b %Y", errors="raise")
cc["delivery_end"]   = pd.to_datetime(parts[1].str.strip(), format="%b %Y", errors="raise")

# %%
bad = cc["delivery_end"] < cc["delivery_start"]
cc.loc[bad, "delivery_start"] = cc.loc[bad, "delivery_start"] - pd.DateOffset(years=1)

# rebuild the text label so it matches the corrected dates
cc["delivery_window"] = (cc["delivery_start"].dt.strftime("%b %Y") + "-"
                         + cc["delivery_end"].dt.strftime("%b %Y"))
bad = cc["delivery_end"] < cc["delivery_start"]
print("rows with end < start:", bad.sum())
print(cc.loc[bad, ["season","delivery_window","delivery_start","delivery_end"]].drop_duplicates())

# %%
assert (cc["delivery_end"] >= cc["delivery_start"]).all()
assert (cc["fair_date"] < cc["delivery_start"]).all()
print(cc[["season","fair_date","delivery_start","delivery_end"]].drop_duplicates().to_string(index=False))

# %%
assert (cc["delivery_end"] >= cc["delivery_start"]).all()
assert (cc["fair_date"] < cc["delivery_start"]).all()
print(cc[["season","fair_date","delivery_window","delivery_start","delivery_end"]].drop_duplicates().to_string(index=False))

cc.to_csv("data/clean/dim_calendar.csv", index=False)

# fact_bookings.csv

# %%
b = pd.read_csv(f"{RAW}/fact_bookings.csv")
print(b.shape)

pd.DataFrame({
    "dtype": b.dtypes.astype(str),
    "nulls": b.isna().sum(),
    "n_unique": b.nunique(),
    "sample": [b[c].dropna().iloc[0] if b[c].notna().any() else None for c in b.columns],
})
# %%
# %%
for c in ["sleeve","size","process_status","season"]:
    print(f"\n--- {c} ({b[c].nunique()}) ---")
    print(b[c].value_counts(dropna=False))

# %%
b[["qty","rate","total_amt"]].describe()

# %%
# what's non-numeric in total_amt?
amt = pd.to_numeric(b["total_amt"], errors="coerce")
print("unparseable:", amt.isna().sum())
print(b.loc[amt.isna(), "total_amt"].value_counts().head(20))

# %%
# grain check: is (order_no, design_code, size) unique?
g = ["order_no","design_code","size"]
print("rows:", len(b), " distinct triples:", b[g].drop_duplicates().shape[0])
print("dupe rows on grain:", b.duplicated(subset=g).sum())

# %%
# orphans: design codes in bookings that aren't in dim_design
missing = set(b["design_code"]) - set(d["design_code"])
print("orphan design codes:", len(missing), list(missing)[:10])


# %%
# how many BOOKING ROWS are affected, not just how many codes
orph = b[b["design_code"].isin(missing)]
print("orphan rows:", len(orph), f"({100*len(orph)/len(b):.1f}%)")
print("\nby season:")
print(orph["season"].value_counts().sort_index())

# %%
# what shapes do the orphan codes take?
import re
def shape(c):
    return re.sub(r"\d", "#", c)
print(orph["design_code"].map(shape).value_counts())
print("\nclean table shapes:")
print(d["design_code"].map(shape).value_counts())

# %%
# the 448 grain duplicates - are they identical rows or conflicting?
dupmask = b.duplicated(subset=["order_no","design_code","size"], keep=False)
dups = b[dupmask].sort_values(["order_no","design_code","size"])
print("full-row duplicates:", b.duplicated().sum())
dups.head(12)

# %%
orph = b[b["design_code"].isin(missing)].copy()
orph["shape"] = orph["design_code"].map(shape)
orph["kind"] = orph["shape"].str.contains("X").map({True: "X-suffix", False: "hyphen-sep"})
print(pd.crosstab(orph["season"], orph["kind"]))
# %%
fixed = orph.loc[orph["kind"]=="hyphen-sep", "design_code"].str.replace(
    r"-(\d)$", r"/\1", regex=True)
print("recovered:", fixed.isin(set(d["design_code"])).sum(), "of", len(fixed))

xs = orph.loc[orph["kind"]=="X-suffix", "design_code"]
print("\nX-suffix examples:", xs.head(10).tolist())
base_match = xs.str.replace(r"/\dX$", "", regex=True)
print("base code exists in dim_design:", base_match.isin(
    d["design_code"].str.replace(r"/\d$", "", regex=True).values).sum(), "of", len(xs))
# %%
orph["kind"] = orph["design_code"].str.contains(r"/\dX$", regex=True).map(
    {True: "X-suffix", False: "hyphen-sep"})
print(pd.crosstab(orph["season"], orph["kind"]))

xs = orph.loc[orph["kind"]=="X-suffix", "design_code"]
print("\nX-suffix rows:", len(xs))
print(xs.head(15).tolist())

# does the base code (before the suffix) exist in dim_design?
base = xs.str.extract(r"^(.+)/\dX$")[0]
dim_base = set(d["design_code"].str.extract(r"^(.+)/\d$")[0])
print("\nbase exists in dim_design:", base.isin(dim_base).sum(), "of", len(xs))
print("distinct bad codes:", xs.nunique())
# %%
d2 = d.copy()
d2["family"] = d2["design_code"].str.extract(r"^(.+)/\d$")[0]

varies = (d2.groupby("family")[["item_group","main_group","pattern","sleeve","colour","rate"]]
            .nunique().max())
print("max distinct values within a family:")
print(varies)
# %%
v = d2.groupby("family")[["pattern","sleeve","colour","rate"]].nunique()
print((v > 1).mean().round(3))          # share of families where each attr varies
print("\nfamilies with any variation:", (v > 1).any(axis=1).mean().round(3))
# %%
xrows = b[b["design_code"].str.contains(r"/9X$", regex=True)]
print(xrows[["design_code","sleeve","rate","qty","season"]].head(10))
print("\nnull sleeve:", xrows["sleeve"].isna().sum(), " null rate:", xrows["rate"].isna().sum())
# %%
d3 = d.copy()
d3["family"] = d3["design_code"].str.extract(r"^(.+)/\d$")[0]

x = b[b["design_code"].str.contains(r"/9X$", regex=True)].copy()
x["family"] = x["design_code"].str.extract(r"^(.+)/\dX$")[0]

cand = (x.merge(d3[["family","sleeve","rate","design_code","pattern","colour"]],
                on=["family","sleeve","rate"], how="left", suffixes=("","_dim")))

n = cand.groupby(cand.index // 1).size()  # placeholder
matches = cand.groupby(["design_code"])["design_code_dim"].nunique()
print("resolved to exactly 1 variant:", (matches == 1).sum())
print("ambiguous (2+ variants):", (matches > 1).sum())
print("no match:", (matches == 0).sum())
print("\ntotal distinct /9X codes:", x["design_code"].nunique())
# %%
dupmask = b.duplicated(subset=["order_no","design_code","size"], keep=False)
dups = b[dupmask]
pairs = (dups.groupby(["order_no","design_code","size"])["process_status"]
             .apply(lambda s: tuple(sorted(s))))
print(pairs.value_counts().head(20))
print("\ngroup sizes:", dups.groupby(["order_no","design_code","size"]).size().value_counts().to_dict())
# %%
sh = b["order_date"].map(lambda s: __import__("re").sub(r"\d", "#", str(s)))
print(sh.value_counts())
# %%
import glob, os
for f in sorted(glob.glob(f"{RAW}/*.csv")):
    print(f"\n{os.path.basename(f)}")
    print("   ", list(pd.read_csv(f, nrows=0).columns))
# %%
amb = b.loc[b["order_date"].str.contains(r"[/-]") & ~b["order_date"].str.match(r"^\d{4}-"), "order_date"]
parts = amb.str.split(r"[/-]", regex=True, expand=True).astype(int)
print("first component max:", parts[0].max(), " min:", parts[0].min())
print("second component max:", parts[1].max(), " min:", parts[1].min())
print("rows where first > 12:", (parts[0] > 12).sum())
print("rows where second > 12:", (parts[1] > 12).sum())
# %%
mask_amb = b["order_date"].str.contains(r"[/-]") & ~b["order_date"].str.match(r"^\d{4}-")
amb = b.loc[mask_amb, "order_date"]
parts = amb.str.split(r"[/-]", regex=True, expand=True).astype(int)

both_le12 = (parts[0] <= 12) & (parts[1] <= 12)
print("ambiguous rows total:", len(amb))
print("unresolvable (both <= 12):", both_le12.sum())
print("resolvable:", (~both_le12).sum())
print("\nas share of all bookings:", f"{100*both_le12.sum()/len(b):.1f}%")
# %%
cal_fair = cc[["season","fair_date"]].drop_duplicates().set_index("season")["fair_date"]
sample = b.loc[mask_amb, ["order_date","season"]].head(20).copy()
sample["fair_date"] = sample["season"].map(cal_fair)
print(sample.to_string(index=False))
# %%
fairmap = cc.drop_duplicates("season").set_index("season")["fair_date"]

iso = b[~b["order_date"].str.contains(r"[/-]") | b["order_date"].str.match(r"^\d{4}-")].copy()
iso = iso[iso["order_date"].str.match(r"^\d{4}-")]
iso["od"] = pd.to_datetime(iso["order_date"])
iso["gap"] = (iso["od"] - iso["season"].map(fairmap)).dt.days
print(iso["gap"].describe())
print("\npercentiles:")
print(iso["gap"].quantile([0, .01, .5, .99, 1]))

# %%
fairmap = cc.drop_duplicates("season").set_index("season")["fair_date"]
mask_amb = ~b["order_date"].str.match(r"^\d{4}-")

dayfirst   = pd.to_datetime(b.loc[mask_amb,"order_date"], dayfirst=True,  format="mixed", errors="coerce")
monthfirst = pd.to_datetime(b.loc[mask_amb,"order_date"], dayfirst=False, format="mixed", errors="coerce")
fair = b.loc[mask_amb,"season"].map(fairmap)

WINDOW_DAYS = 17   # <- your number

ok_d = (dayfirst   >= fair) & (dayfirst   <= fair + pd.Timedelta(days=WINDOW_DAYS))
ok_m = (monthfirst >= fair) & (monthfirst <= fair + pd.Timedelta(days=WINDOW_DAYS))

print("only dayfirst valid:  ", (ok_d & ~ok_m).sum())
print("only monthfirst valid:", (ok_m & ~ok_d).sum())
print("both valid:           ", (ok_d & ok_m).sum())
print("neither valid:        ", (~ok_d & ~ok_m).sum())
# %%
same = (dayfirst == monthfirst)
print("readings identical:", same.sum())
print("readings differ but both in window:", (ok_d & ok_m & ~same).sum())

diff = b.loc[mask_amb][ok_d & ok_m & ~same]
print("\nexamples where they genuinely differ:")
tmp = pd.DataFrame({"raw": diff["order_date"], "season": diff["season"],
                    "dayfirst": dayfirst[ok_d & ok_m & ~same],
                    "monthfirst": monthfirst[ok_d & ok_m & ~same]})
print(tmp.head(15).to_string(index=False))
print("\ntruly ambiguous rows:", len(tmp), f"({100*len(tmp)/len(b):.2f}% of bookings)")
# %%
bt = b.copy()
bt["amt"] = pd.to_numeric(bt["total_amt"].astype(str).str.replace(",", ""), errors="coerce")
bt["calc"] = bt["qty"] * bt["rate"]
bt["ratio"] = bt["amt"] / bt["calc"]

print("rows where amt == qty*rate:", (bt["amt"] == bt["calc"]).sum(), "of", len(bt))
print("\nratio distribution:")
print(bt["ratio"].describe())
print("\nmost common ratios:")
print(bt["ratio"].round(3).value_counts().head(15))
# %%
bad = bt[bt["ratio"] != 1].copy()
print("exactly 12x:", (bad["ratio"] == 12).sum())
print("other:", (bad["ratio"] != 12).sum())

print("\n--- the 12x rows ---")
print(bad[bad["ratio"]==12]["season"].value_counts().sort_index())
print(bad[bad["ratio"]==12][["qty","rate","amt","calc"]].describe().loc[["min","50%","max"]])

print("\n--- the non-12x rows ---")
print(bad[bad["ratio"]!=12]["season"].value_counts().sort_index())
print("\nratio range:", bad[bad['ratio']!=12]["ratio"].min(), "to", bad[bad['ratio']!=12]["ratio"].max())
print(bad[bad["ratio"]!=12][["qty","rate","amt","calc","ratio"]].head(10).to_string())
# %%
bad = bt[bt["ratio"] != 1].copy()
bad["implied_pieces"] = bad["amt"] / bad["rate"]
print("implied pieces are whole numbers:", (bad["implied_pieces"] % 1 == 0).sum(), "of", len(bad))
print(bad["implied_pieces"].describe())

bad["dozens"] = bad["implied_pieces"] / 12
print("\nimplied pieces / 12 vs recorded qty:")
print(bad[["qty","rate","amt","implied_pieces","dozens"]].head(15).to_string())
print("\nqty == round(implied/12):", (bad["qty"] == bad["dozens"].round()).sum(), "of", len(bad))
# %%
odd = bad[bad["qty"] != bad["dozens"].round()]
print(odd[["qty","rate","amt","implied_pieces","dozens","ratio","season"]].to_string())
# %%
fairmap = cc.drop_duplicates("season").set_index("season")["fair_date"]

bb = b.copy()
mask_amb = ~bb["order_date"].str.match(r"^\d{4}-")

iso_parsed = pd.to_datetime(bb.loc[~mask_amb, "order_date"], format="%Y-%m-%d")

df_p = pd.to_datetime(bb.loc[mask_amb, "order_date"], dayfirst=True,  format="mixed", errors="coerce")
mf_p = pd.to_datetime(bb.loc[mask_amb, "order_date"], dayfirst=False, format="mixed", errors="coerce")
fair = bb.loc[mask_amb, "season"].map(fairmap)
ok_d = (df_p >= fair) & (df_p <= fair + pd.Timedelta(days=17))

bb["order_date"] = pd.concat([iso_parsed, df_p.where(ok_d, mf_p)]).sort_index()

assert bb["order_date"].notna().all()
assert (bb["order_date"] - bb["season"].map(fairmap)).dt.days.between(1, 17).all()
print(bb["order_date"].min(), "->", bb["order_date"].max())
# %%
bb["amt"] = pd.to_numeric(bb["total_amt"].astype(str).str.replace(",", ""), errors="raise")
mismatch = bb["amt"] != bb["qty"] * bb["rate"]
print("rows to fix:", mismatch.sum())

bb.loc[mismatch, "qty"] = (bb.loc[mismatch, "amt"] / bb.loc[mismatch, "rate"]).astype(int)

assert (bb["amt"] == bb["qty"] * bb["rate"]).all()
bb = bb.drop(columns=["total_amt"]).rename(columns={"amt": "total_amt"})
print(bb["qty"].describe())
# %%
bb["amt"] = pd.to_numeric(bb["total_amt"].astype(str).str.replace(",", ""), errors="raise")

mismatch = bb["amt"] != bb["qty"] * bb["rate"]
print("rows to fix:", mismatch.sum())

bb.loc[mismatch, "qty"] = (bb.loc[mismatch, "amt"] / bb.loc[mismatch, "rate"]).round().astype(int)

assert (bb["amt"] == bb["qty"] * bb["rate"]).all()
bb = bb.drop(columns=["total_amt"]).rename(columns={"amt": "total_amt"})
print(bb["qty"].describe())
# %%
top = bb.nlargest(20, "qty")[["order_no","party_code","design_code","size","qty","rate","total_amt","season"]]
print(top.to_string(index=False))

print("\nqty percentiles:")
print(bb["qty"].quantile([.5,.9,.99,.999,1]))

print("\nlarge orders by party:")
print(bb[bb["qty"] > 5000]["party_code"].value_counts().head(10))

# cleaning fact_bookings.csv
# %%
# --- design code repair
d3 = d.copy()
d3["family"] = d3["design_code"].str.extract(r"^(.+)/\d$")[0]

# 1. hyphen separator -> slash
bb["design_code"] = bb["design_code"].str.replace(r"-(\d)$", r"/\1", regex=True)

# 2. /9X placeholder -> resolve via family + sleeve + rate
xmask = bb["design_code"].str.contains(r"/\dX$", regex=True)
xf = bb.loc[xmask].copy()
xf["family"] = xf["design_code"].str.extract(r"^(.+)/\dX$")[0]
lookup = (d3[["family","sleeve","rate","design_code"]]
            .drop_duplicates(subset=["family","sleeve","rate"])
            .rename(columns={"design_code":"resolved"}))
xf = xf.merge(lookup, on=["family","sleeve","rate"], how="left")
assert xf["resolved"].notna().all()
bb.loc[xmask, "design_code"] = xf["resolved"].values

# verify: zero orphans
orphans = set(bb["design_code"]) - set(d["design_code"])
print("orphan design codes:", len(orphans))

# --- drop process_status and dedupe
before = len(bb)
bb = bb.drop(columns=["process_status"]).drop_duplicates()
print(f"rows: {before} -> {len(bb)}  (removed {before-len(bb)})")

g = ["order_no","design_code","size"]
print("still duplicated on grain:", bb.duplicated(subset=g).sum())
# %%
print(sorted(orphans))
print("\nrows affected:", bb["design_code"].isin(orphans).sum())
print(bb[bb["design_code"].isin(orphans)][["design_code","sleeve","rate","qty","season"]].to_string(index=False))
# %%
# rebuild from the raw table
bb2 = bb.copy()

# normalise ANY trailing "-<digit>" or "-<digit>X" to slash form
bb2["design_code"] = bb2["design_code"].str.replace(r"-(\d X?)$", r"/\1", regex=True)
bb2["design_code"] = bb2["design_code"].str.replace(r"-(\dX)$", r"/\1", regex=True)

still = set(bb2["design_code"]) - set(d["design_code"])
print("after separator fix, non-matching:", len(still), sorted(still)[:10])
# %%
xmask = bb2["design_code"].str.contains(r"/\dX$", regex=True)
print("X-suffix rows to resolve:", xmask.sum())

xf = bb2.loc[xmask].copy()
xf["family"] = xf["design_code"].str.extract(r"^(.+)/\dX$")[0]
xf = xf.merge(lookup, on=["family","sleeve","rate"], how="left")
print("unresolved:", xf["resolved"].isna().sum())
bb2.loc[xmask, "design_code"] = xf["resolved"].values

orphans2 = set(bb2["design_code"]) - set(d["design_code"])
print("orphan design codes:", len(orphans2))
# %%
bb2.to_csv("data/clean/fact_bookings.csv", index=False)
print(bb2.shape)
print(bb2.dtypes)
# %%
bb2["total_amt"] = bb2["total_amt"].astype("int64")
bb2.to_csv("data/clean/fact_bookings.csv", index=False)
# %%
# fact_dispatch.csv
disp = pd.read_csv(f"{RAW}/fact_dispatch.csv")
print(disp.shape)
pd.DataFrame({
    "dtype": disp.dtypes.astype(str),
    "nulls": disp.isna().sum(),
    "n_unique": disp.nunique(),
    "sample": [disp[c].dropna().iloc[0] if disp[c].notna().any() else None for c in disp.columns],
})
# %%
# orphan orders and designs
bad_orders = set(disp["order_no"]) - set(bb2["order_no"])
bad_designs = set(disp["design_code"]) - set(d["design_code"])
print("orphan order_no:", len(bad_orders), sorted(bad_orders)[:10])
print("rows affected:", disp["order_no"].isin(bad_orders).sum())
print("\norphan design_code:", len(bad_designs), sorted(bad_designs)[:10])
print("rows affected:", disp["design_code"].isin(bad_designs).sum())
# %%
# date formats
import re
print(disp["dispatch_date"].map(lambda s: re.sub(r"\d", "#", str(s))).value_counts())
# %%
# qty sanity
print(disp["qty_dispatched"].describe())
print("zeros:", (disp["qty_dispatched"] == 0).sum(), " negatives:", (disp["qty_dispatched"] < 0).sum())
# %%
orph_d = disp[disp["order_no"].isin(bad_orders)]
print(orph_d.to_string(index=False))
# %%
dd = disp[~disp["order_no"].isin(bad_orders)].copy()
dd["dispatch_date"] = pd.to_datetime(dd["dispatch_date"])

shipped = dd.groupby(["order_no","design_code"])["qty_dispatched"].sum().rename("shipped")
booked = bb2.groupby(["order_no","design_code"])["qty"].sum().rename("booked")

cmp = pd.concat([booked, shipped], axis=1)
print("order+design pairs:", len(cmp))
print("booked but never shipped:", cmp["shipped"].isna().sum())
print("shipped but never booked:", cmp["booked"].isna().sum())

over = cmp.dropna()
over = over[over["shipped"] > over["booked"]]
print("OVER-SHIPPED (shipped > booked):", len(over))
print(over.head(10))
# %%
prod = pd.read_csv(f"{RAW}/fact_production.csv")
codes = orph_d["design_code"].unique()
sub = prod[prod["design_code"].isin(codes)]
print(sub["order_ref"].value_counts().head())
# %%
dc = disp.copy()
dc["dispatch_date"] = pd.to_datetime(dc["dispatch_date"], format="%Y-%m-%d")
dc["order_no"] = dc["order_no"].str.replace(r"^ORD-\w{4}-99999$", "GENERAL", regex=True)

print(dc["order_no"].str.startswith("GENERAL").sum(), "general rows")
print("orphans now:", len(set(dc["order_no"]) - set(bb2["order_no"]) - {"GENERAL"}))

# dispatch must not precede the order
chk = dc[dc["order_no"] != "GENERAL"].merge(
    bb2[["order_no","order_date"]].drop_duplicates("order_no"), on="order_no", how="left")
print("dispatch before order:", (chk["dispatch_date"] < chk["order_date"]).sum())

dc.to_csv("data/clean/fact_dispatch.csv", index=False)
# %%
chk2 = dc[dc["order_no"] != "GENERAL"].merge(
    bb2.groupby("order_no")["order_date"].min().rename("first_order_date"),
    on="order_no", how="left")
print("dispatch before EARLIEST line on the order:", (chk2["dispatch_date"] < chk2["first_order_date"]).sum())
# %%
bad_t = chk2[chk2["dispatch_date"] < chk2["first_order_date"]]
print(bad_t[["order_no","design_code","dispatch_date","first_order_date"]].head(15).to_string(index=False))
print("\nby season:", bad_t["order_no"].str[4:8].value_counts().to_dict())
print("gap in days:", (bad_t["first_order_date"] - bad_t["dispatch_date"]).dt.days.describe())
# %%
fairmap2 = cc.drop_duplicates("season").set_index("season")["fair_date"]
bad_t = chk2[chk2["dispatch_date"] < chk2["first_order_date"]].copy()
bad_t["season"] = bad_t["order_no"].str[4:6].map(
    {"SU":"Summer","DI":"Diwali","CH":"Christmas","PO":"Pongal"}) + " 20" + bad_t["order_no"].str[6:8]
bad_t["fair_date"] = bad_t["season"].map(fairmap2)
print("dispatched before the fair:", (bad_t["dispatch_date"] < bad_t["fair_date"]).sum(), "of", len(bad_t))
# %%
# do these designs have GENERAL production lots?
pcodes = bad_t["design_code"].unique()
ps = prod[prod["design_code"].isin(pcodes)]
print(ps["order_ref"].apply(lambda x: "GENERAL" if x=="GENERAL" else "specific").value_counts())
# %%
first_ord = bb2.groupby("order_no")["order_date"].min().rename("first_order_date")
dc = dc.merge(first_ord, on="order_no", how="left")
dc["from_stock"] = dc["dispatch_date"] < dc["first_order_date"]
dc = dc.drop(columns=["first_order_date"])
print(dc["from_stock"].value_counts())
dc.to_csv("data/clean/fact_dispatch.csv", index=False)
# %%
odd5 = bad_t[bad_t["dispatch_date"] >= bad_t["fair_date"]]
print(odd5[["order_no","design_code","dispatch_date","first_order_date","fair_date"]].to_string(index=False))
# %%

# fact_production
prod = pd.read_csv(f"{RAW}/fact_production.csv")
print(prod.shape)
pd.DataFrame({
    "dtype": prod.dtypes.astype(str),
    "nulls": prod.isna().sum(),
    "n_unique": prod.nunique(),
    "sample": [prod[c].dropna().iloc[0] if prod[c].notna().any() else None for c in prod.columns],
})
# %%
for c in ["stage","jobworker_type"]:
    print(f"\n--- {c} ---")
    print(prod[c].value_counts(dropna=False))
print("\norder_ref GENERAL vs specific:")
print((prod["order_ref"] == "GENERAL").value_counts())
# %%
print(prod[["qty_in","qty_out","expected_days","actual_days"]].describe())
# %%
print(prod.groupby("jobworker_type")["jobworker_id"].apply(lambda s: s.isna().sum()))
print(prod[prod["jobworker_id"].isna()]["stage"].value_counts())
# %%
pp = prod.copy()
pp["start_date"] = pd.to_datetime(pp["start_date"])
pp["end_date"] = pd.to_datetime(pp["end_date"])
pp["date_gap"] = (pp["end_date"] - pp["start_date"]).dt.days

print("end < start:", (pp["date_gap"] < 0).sum())
print(pp["date_gap"].describe())
print("\ndate_gap vs actual_days — do they agree?")
print((pp["date_gap"] == pp["actual_days"].round()).sum(), "of", len(pp))
print(pp[["qty_in","expected_days","actual_days","date_gap"]].head(10).to_string())
# %%
# qty_out > qty_in anywhere?
print("gains:", (prod["qty_out"] > prod["qty_in"]).sum())

# does qty flow between stages?
order = ["Cutting","Stitching","Washing","Checking","Packing","Ready"]
pp["stage_n"] = pp["stage"].map({s:i for i,s in enumerate(order)})
pp = pp.sort_values(["lot_id","stage_n"])
pp["prev_out"] = pp.groupby("lot_id")["qty_out"].shift()
mismatch = pp["prev_out"].notna() & (pp["qty_in"] != pp["prev_out"])
print("stage-to-stage qty breaks:", mismatch.sum())
# %%
import numpy as np
print("floor(actual_days) == date_gap:", (np.floor(pp["actual_days"]) == pp["date_gap"]).sum(), "of", len(pp))
# %%
bad_p = pp[pp["date_gap"] < 0]
print(bad_p[["lot_id","stage","start_date","end_date","date_gap","actual_days"]].head(15).to_string(index=False))
print("\nby stage:", bad_p["stage"].value_counts().to_dict())
print("by season:", bad_p["lot_id"].str[4:8].value_counts().to_dict())
print("\nactual_days on these rows:")
print(bad_p["actual_days"].describe())
# %%
pp["prev_end"] = pp.groupby("lot_id")["end_date"].shift()
print("stage starts before previous stage ended:", (pp["start_date"] < pp["prev_end"]).sum())
# %%
neg = pp["date_gap"] < 0
pp.loc[neg, ["start_date","end_date"]] = pp.loc[neg, ["end_date","start_date"]].values

pp["date_gap"] = (pp["end_date"] - pp["start_date"]).dt.days
assert (pp["date_gap"] >= 0).all()

pp = pp.sort_values(["lot_id","stage_n"])
pp["prev_end"] = pp.groupby("lot_id")["end_date"].shift()
print("stage starts before previous ended:", (pp["start_date"] < pp["prev_end"]).sum())
print("floor(actual_days) == date_gap:", (np.floor(pp["actual_days"]) == pp["date_gap"]).sum(), "of", len(pp))
# %%
print("orphan design codes:", len(set(pp["design_code"]) - set(d["design_code"])))

specific = pp[pp["order_ref"] != "GENERAL"]["order_ref"]
print("distinct specific order_refs:", specific.nunique())
print("orphan order_refs:", len(set(specific) - set(bb2["order_no"])))
# %%
print(pp[pp["stage"]=="Cutting"]["qty_in"].describe())
print("\nlots over 50k:", (pp[pp["stage"]=="Cutting"]["qty_in"] > 50000).sum())
# %%
missing_p = set(pp["design_code"]) - set(d["design_code"])
import re
shp = pd.Series(list(missing_p)).map(lambda c: re.sub(r"\d","#",c))
print(shp.value_counts())
print("\nrows affected:", pp["design_code"].isin(missing_p).sum())
# %%
big = pp[(pp["stage"]=="Cutting") & (pp["qty_in"] > 50000)]
print(big["design_code"].value_counts().head(10))

# does lot qty roughly match total booked for that design+season?
big2 = big.copy()
big2["season"] = big2["lot_id"].str[4:6].map({"SU":"Summer","DI":"Diwali","CH":"Christmas","PO":"Pongal"}) + " 20" + big2["lot_id"].str[6:8]
booked_ds = bb2.groupby(["season","design_code"])["qty"].sum().rename("booked")
big2 = big2.merge(booked_ds, on=["season","design_code"], how="left")
print(big2[["lot_id","design_code","season","qty_in","booked"]].head(15).to_string(index=False))
# %%
pp["design_code"] = pp["design_code"].str.replace(r"-(\d)$", r"/\1", regex=True)
print("orphans now:", len(set(pp["design_code"]) - set(d["design_code"])))
# %%
pp["jobworker_id"] = pp["jobworker_id"].fillna("OUT-UNKNOWN")

pp = pp.drop(columns=[c for c in ["stage_n","prev_out","prev_end","date_gap"] if c in pp.columns])
pp.to_csv("data/clean/fact_production.csv", index=False)
print(pp.shape)
print(pp.isna().sum())
# %%
samp = pd.read_csv(f"{RAW}/fact_samples.csv")
att  = pd.read_csv(f"{RAW}/fact_fair_attendance.csv")
mkd  = pd.read_csv(f"{RAW}/fact_inventory_markdown.csv")

for name, t in [("fact_samples", samp), ("fact_fair_attendance", att), ("fact_inventory_markdown", mkd)]:
    print(f"\n{'='*60}\n{name}  {t.shape}\n{'='*60}")
    print(pd.DataFrame({
        "dtype": t.dtypes.astype(str),
        "nulls": t.isna().sum(),
        "n_unique": t.nunique(),
        "sample": [t[c].dropna().iloc[0] if t[c].notna().any() else None for c in t.columns],
    }))
# %%
# samples: are the repeats across seasons or within?
rep = samp[samp.duplicated("design_code", keep=False)]
print("designs appearing >1 time:", rep["design_code"].nunique())
print("of those, dupes within a single season:", rep.duplicated(["season","design_code"]).sum())

# attendance: does eligibility explain the 600?
elig = pc[["party_code","first_season"]].merge(
    cc[["season","fair_date"]].drop_duplicates(), how="cross")
fseason = cc.drop_duplicates("season").set_index("season")["fair_date"]
elig["ok"] = elig["season"].map(fseason) >= elig["first_season"].map(fseason)
print("\nexpected eligible rows:", elig["ok"].sum(), " actual:", len(att))

# markdown: what are the 53 nulls?
print("\nmarkdown seasons:", sorted(mkd["season"].unique()))
print(mkd[mkd["realised_value"].isna()][["season","qty_unsold","markdown_pct"]].head())
print("\nrealised_value vs qty_unsold * rate * (1-markdown):")
print(mkd[["qty_unsold","markdown_pct","realised_value"]].describe())
# %%
# grain + referential integrity, all three at once
print("samples grain (season, design):", samp.duplicated(["season","design_code"]).sum(), "dupes")
print("samples orphan designs:", len(set(samp["design_code"]) - set(d["design_code"])))

print("\nattendance grain (season, party):", att.duplicated(["season","party_code"]).sum(), "dupes")
print("attendance orphan parties:", len(set(att["party_code"]) - set(pc["party_code"])))
print(att["attended"].value_counts(dropna=False))

print("\nmarkdown grain (season, design):", mkd.duplicated(["season","design_code"]).sum(), "dupes")
print("markdown orphan designs:", len(set(mkd["design_code"]) - set(d["design_code"])))
# %%
m = mkd.merge(d[["design_code","rate"]], on="design_code", how="left")
m["calc"] = m["qty_unsold"] * m["rate"] * (1 - m["markdown_pct"]/100)
ok = m["realised_value"].notna()
m["ratio"] = m["realised_value"] / m["calc"]
print(m.loc[ok,"ratio"].describe())
print("\nwithin 1% of calc:", (m.loc[ok,"ratio"].between(0.99,1.01)).sum(), "of", ok.sum())
# %%
m["realised_value"] = m["realised_value"].fillna(m["calc"].round(2))
m = m.drop(columns=["rate","calc","ratio"])
assert m["realised_value"].notna().all()
m.to_csv("data/clean/fact_inventory_markdown.csv", index=False)

samp.to_csv("data/clean/fact_samples.csv", index=False)
att.to_csv("data/clean/fact_fair_attendance.csv", index=False)

import os
for f in sorted(os.listdir("data/clean")):
    print(f"{f:<32} {pd.read_csv(f'data/clean/{f}').shape}")
# %%
