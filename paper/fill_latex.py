"""
paper/fill_latex.py
====================
Reads results/datasets/cross_case_summary.csv and patches draft_manuscript.tex:
  1. Fills \\todo{run} cells in tab:crosscase with real CAR/p/Verdict values.
  2. Replaces the CS-01 framebox placeholder with \\includegraphics.
  3. Appends CS-02..CS-06 figure environments after the CS-01 figure.

Run from project root:
    python paper/fill_latex.py
Output: paper/draft_manuscript.tex (in-place update)
Backup: paper/draft_manuscript.tex.bak
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper" / "draft_manuscript.tex"
CSV_PATH = ROOT / "results" / "datasets" / "cross_case_summary.csv"

# ── Load results ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.set_index("case_id")

def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True

def fmt_car(v) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return r"n/a"
        return f"${f:+.2f}$"
    except (TypeError, ValueError):
        return r"n/a"

def fmt_p(v) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return r"n/a"
        if f < 0.01:
            return r"$<0.01$"
        return f"${f:.2f}$"
    except (TypeError, ValueError):
        return r"n/a"

def fmt_verdict(row) -> str:
    v = row.get("validation", "Insufficient")
    if pd.isna(v) or v in ("Insufficient",):
        return r"n/a"
    car = row.get("CAR_pct")
    p   = row.get("p_value")
    if v == "Null":
        return r"Null"
    # Confirmed + or Confirmed -
    try:
        if float(car) > 0:
            return r"Confirmed $+$"
        else:
            return r"Confirmed $-$"
    except (TypeError, ValueError):
        return r"Confirmed"

# Map case_id -> (CAR_str, p_str, verdict_str)
replacements: dict[str, tuple[str, str, str]] = {}
for cid, row in df.iterrows():
    replacements[cid] = (
        fmt_car(row.get("CAR_pct")),
        fmt_p(row.get("p_value")),
        fmt_verdict(row),
    )
    print(f"  {cid}: CAR={replacements[cid][0]}  p={replacements[cid][1]}  verdict={replacements[cid][2]}")

# ── Read LaTeX ────────────────────────────────────────────────────────────────
shutil.copy(TEX_PATH, str(TEX_PATH) + ".bak")
tex = TEX_PATH.read_text(encoding="utf-8")

# ── Replace \\todo{run} cells ─────────────────────────────────────────────────
# Each row ends with:  & \todo{run} & \todo{run} & \todo{run} \\
# We match by the row prefix (case_id) and replace three \todo{run}s.

CASE_ROW_PREFIXES = {
    "CS-01": r"CS-01 & Red Sea / Houthi crisis",
    "CS-02": r"CS-02 & US semiconductor export controls",
    "CS-03": r"CS-03 & Iran oil sanctions tightening",
    "CS-04": r"CS-04 & China rare-earth export curbs",
    "CS-05": r"CS-05 & Russia Black Sea energy disruption",
    "CS-06": r"CS-06 & India pharma API supply shock",
}

def replace_todos_in_row(tex: str, prefix: str, car: str, p: str, verdict: str) -> str:
    """
    Find the table row starting with `prefix`, then replace its three
    \\todo{run} occurrences (left-to-right: CAR, p, verdict).
    """
    pattern = re.compile(
        r'(' + re.escape(prefix) + r'[^\n]*?)'   # row prefix + fixed cols
        r'\\todo\{run\}'                           # CAR placeholder
        r'([^\n]*?)'                               # & separator
        r'\\todo\{run\}'                           # p placeholder
        r'([^\n]*?)'                               # & separator
        r'\\todo\{run\}'                           # verdict placeholder
        r'([^\n]*?\\\\)',                          # trailing \\
        re.DOTALL
    )
    def sub(m):
        return (m.group(1) + car +
                m.group(2) + p +
                m.group(3) + verdict +
                m.group(4))
    result, n = pattern.subn(sub, tex)
    if n == 0:
        print(f"  WARNING: pattern not found for prefix: {prefix!r}")
    return result

for cid, prefix in CASE_ROW_PREFIXES.items():
    if cid in replacements:
        car, p, verdict = replacements[cid]
        tex = replace_todos_in_row(tex, prefix, car, p, verdict)
        print(f"  Filled {cid}: CAR={car}  p={p}  verdict={verdict}")
    else:
        print(f"  WARNING: {cid} not in CSV, todo{{run}} left unfilled.")

# ── Build all 6 figure environments ──────────────────────────────────────────

def make_figure(case_id: str, label_id: str, caption: str) -> str:
    fig_dir = case_id.lower().replace("-", "")
    return (
        f"\\begin{{figure}}[t]\n"
        f"\\centering\n"
        f"\\includegraphics[width=\\columnwidth]{{../results/figures/{fig_dir}/propagation_map_sm.png}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{fig:propmap_{label_id}}}\n"
        f"\\end{{figure}}"
    )

cs01_fig = make_figure(
    "CS-01", "cs01",
    r"CS-01 propagation map: Red Sea / Houthi crisis (12 Jan 2024). "
    r"Red star = event origin (Yemen). Orange circles = directly affected countries "
    r"(size $\propto$ mean $|$Goldstein$|$). Dashed arrow to India coloured by "
    r"event-study verdict. Pipeline-detected pathway only; not a causal claim."
)

cs02_fig = make_figure(
    "CS-02", "cs02",
    r"CS-02 propagation map: US semiconductor export controls (29 Jan 2024). "
    r"Red star = event origin (USA). Orange circles = directly affected countries. "
    r"Dashed arrow to India coloured by event-study verdict. "
    r"Pipeline-detected pathway only; not a causal claim."
)

cs03_fig = make_figure(
    "CS-03", "cs03",
    r"CS-03 propagation map: Iran oil sanctions tightening (16 Feb 2024). "
    r"Red star = event origin (Iran). Orange circles = directly affected countries. "
    r"Dashed arrow to India coloured by event-study verdict. "
    r"Pipeline-detected pathway only; not a causal claim."
)

cs04_fig = make_figure(
    "CS-04", "cs04",
    r"CS-04 propagation map: China rare-earth export curbs (01 Feb 2024). "
    r"Red star = event origin (China). Orange circles = directly affected countries. "
    r"Dashed arrow to India coloured by event-study verdict. "
    r"Pipeline-detected pathway only; not a causal claim."
)

cs05_fig = make_figure(
    "CS-05", "cs05",
    r"CS-05 propagation map: Russia Black Sea energy disruption (18 Jan 2024). "
    r"Red star = event origin (Russia). Orange circles = directly affected countries. "
    r"Dashed arrow to India coloured by event-study verdict. "
    r"Pipeline-detected pathway only; not a causal claim."
)

cs06_fig = make_figure(
    "CS-06", "cs06",
    r"CS-06 propagation map: India pharma API supply shock (08 Mar 2024). "
    r"Red star = event origin (China). Orange circles = directly affected countries. "
    r"Dashed arrow to India coloured by event-study verdict. "
    r"Pipeline-detected pathway only; not a causal claim."
)

# ── Replace the CS-01 framebox block and insert all 6 figures ─────────────────
# The original block to find (exact match):
FRAMEBOX_BLOCK = (
    r"\begin{figure}[t]" + "\n"
    r"\centering" + "\n"
    r"% Uncomment after running run_all_cases.py:" + "\n"
    r"% \includegraphics[width=\columnwidth]{../results/figures/cs01/propagation_map_sm.png}" + "\n"
    r"\framebox{\parbox{\columnwidth}{\centering\vspace{1.2cm}" + "\n"
    r"\small\textit{Figure placeholder.}\\[4pt]" + "\n"
    r"Run \texttt{python experiments/case\_studies/run\_all\_cases.py}\\" + "\n"
    r"then uncomment the \texttt{\textbackslash{}includegraphics} line above.\\[4pt]" + "\n"
    r"Output: \texttt{results/figures/cs01/propagation\_map\_sm.png}" + "\n"
    r"\vspace{1.2cm}}}" + "\n"
    r"\caption{CS-01 propagation map: Red Sea / Houthi crisis (12 Jan 2024). Red star = event origin (Yemen). Orange circles = directly affected countries (size $\propto$ mean $|$Goldstein$|$). Dashed arrow to India coloured by event-study verdict. Pipeline-detected pathway only; not a causal claim.}" + "\n"
    r"\label{fig:propmap_cs01}" + "\n"
    r"\end{figure}"
)

all_figures = (
    cs01_fig + "\n\n" +
    cs02_fig + "\n\n" +
    cs03_fig + "\n\n" +
    cs04_fig + "\n\n" +
    cs05_fig + "\n\n" +
    cs06_fig
)

if FRAMEBOX_BLOCK in tex:
    tex = tex.replace(FRAMEBOX_BLOCK, all_figures)
    print("  Replaced CS-01 framebox with all 6 figure blocks (exact match).")
else:
    # Fallback: regex approach
    framebox_pat = re.compile(
        r'\\begin\{figure\}\[t\]\s*\\centering\s*'
        r'%[^\n]*\n%[^\n]*\n'
        r'\\framebox\{.*?\}'
        r'\s*\\caption\{CS-01[^}]*\}'
        r'\s*\\label\{fig:propmap_cs01\}'
        r'\s*\\end\{figure\}',
        re.DOTALL
    )
    tex, n_subs = framebox_pat.subn(all_figures, tex)
    if n_subs:
        print(f"  Replaced CS-01 framebox via regex ({n_subs} substitution).")
    else:
        # Last resort: insert before \section{Discussion}
        print("  WARNING: CS-01 framebox not found by regex. Inserting figures before \\section{Discussion}.")
        tex = tex.replace(
            r"\section{Discussion}",
            all_figures + "\n\n" + r"\section{Discussion}"
        )

# ── Write patched LaTeX ───────────────────────────────────────────────────────
TEX_PATH.write_text(tex, encoding="utf-8")
print(f"\nPatched LaTeX written to {TEX_PATH}")
print(f"Backup at {str(TEX_PATH) + '.bak'}")

# ── Quick sanity check ────────────────────────────────────────────────────────
remaining_todos = tex.count(r"\todo{run}")
print(f"\nRemaining \\todo{{run}} occurrences: {remaining_todos}")
propmap_count = tex.count("propagation_map_sm.png")
print(f"\\includegraphics lines for propagation maps: {propmap_count}")
