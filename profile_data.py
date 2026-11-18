import os
import pandas as pd

ROOT = "6af458274c1e8_data_pack"   # extracted folder name
OUT = "data_dictionary.txt"

lines = []
for dirpath, _, files in os.walk(ROOT):
    for f in files:
        path = os.path.join(dirpath, f)
        ext = f.lower().rsplit(".", 1)[-1]
        try:
            if ext == "csv":
                df = pd.read_csv(path, low_memory=False)
            elif ext in ("xlsx", "xls"):
                df = pd.read_excel(path)
            elif ext == "json":
                df = pd.read_json(path)
            else:
                lines.append(f"\n### {path}  (skipped: .{ext})")
                continue
        except Exception as e:
            lines.append(f"\n### {path}  (error: {e})")
            continue

        lines.append(f"\n### {path}")
        lines.append(f"rows={len(df)}, cols={df.shape[1]}")
        for c in df.columns:
            s = df[c]
            info = f"- {c} | {s.dtype} | null%={s.isna().mean()*100:.1f} | unique={s.nunique()}"
            if s.dtype == "object" and s.nunique() <= 12:
                info += f" | values={list(s.dropna().unique())}"
            elif pd.api.types.is_numeric_dtype(s):
                info += f" | min={s.min()} max={s.max()}"
            lines.append(info)

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n".join(lines))
