import pandas as pd, numpy as np

D = "6af458274c1e8_data_pack/Data Pack/"
bbb  = pd.read_csv(D + "ball_by_ball.csv", low_memory=False)
m    = pd.read_csv(D + "matches.csv")
comp = pd.read_csv(D + "competitions.csv")

bbb = bbb.sort_values(["match_id", "innings", "innings_ball_num"])

# 1. Ordering and duplicates
dups = bbb.duplicated(["match_id", "innings", "innings_ball_num"]).sum()
print("1. duplicate ball keys:", dups)

# 2. What do runs / extras columns mean?
print("\n2. Extras semantics (mean/max by extras_type):")
print(bbb.groupby(bbb["extras_type"].fillna("none"))
         [["runs", "extras1_runs", "extras_2_runs"]].agg(["mean", "max", "count"]))

# 3. Legal balls per over (should mostly be 6)
bbb["is_legal"] = ~bbb["extras_type"].isin(["wide", "Overstep"])
legal_per_over = (bbb[bbb.innings <= 2]
                  .groupby(["match_id", "innings", "overid"])["is_legal"].sum())
print("\n3. Legal balls per over:\n", legal_per_over.value_counts().head(8))

# 4. Innings totals (using runs + extras as a first guess)
bbb["total_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
tot = bbb[bbb.innings <= 2].groupby(["match_id", "innings"])["total_runs"].sum()
print("\n4. Innings total summary:\n", tot.groupby("innings").describe())

# 5. Which leagues are in the data?
mm = m.merge(comp, on="comp_id", how="left")
print("\n5. Competitions (matches per comp):")
print(mm.groupby(["comp_name", "season", "match_format"]).size()
        .sort_values(ascending=False).head(40))

# 6. Result strings, stage, knockout
print("\n6. Sample results:\n", m["result"].sample(15, random_state=1).tolist())
print(m["stage"].value_counts())
print("knock_out counts:\n", m["knock_out"].value_counts())

# 7. Shot types and dismissal types
print("\n7. stroke values:\n", bbb["stroke"].value_counts().head(30))
print("\n   delivery_style values:\n", bbb["delivery_style"].value_counts().head(15))

# 8. Coverage
print("\n8. Matches in bbb but not matches.csv:",
      len(set(bbb.match_id) - set(m.match_id)))
print("   Matches without winner:", m["win_team_id"].isna().sum())