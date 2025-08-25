# app.py — Refugee Camp Site Feasibility AHP Tool (EN UI + DXF download)
# Run: streamlit run app.py

import os, json, time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

APP_TITLE = "Refugee Camp Site Feasibility — AHP Tool"
DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RI_TABLE = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}

# ---------- helpers

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def load_docx_lines(path):
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            lines.append(t)
    return lines

def load_categories_from_excel(path):
    df = pd.read_excel(path)
    cols = {c.strip().lower(): c for c in df.columns}
    cat_col = cols.get("category")
    sub_col = cols.get("subcriterion") or cols.get("sub-criterion")
    base_col = cols.get("baseweight") or cols.get("base_weight")
    if not cat_col or not sub_col:
        raise ValueError("Excel must contain 'Category' and 'Subcriterion' columns.")
    keep = [cat_col, sub_col] + ([base_col] if base_col else [])
    df = df[keep].copy()
    df.columns = ["Category", "Subcriterion"] + (["BaseWeight"] if base_col else [])
    return df

def normalize(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    return v if s == 0 else v / s

def ahp_from_pairs(names: List[str], pairs: Dict[Tuple[int,int], Tuple[int,int]]):
    n = len(names)
    A = np.ones((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            better, intensity = pairs.get((i,j), (i,1))
            k = int(max(1, min(9, intensity)))
            val = 1.0 if k == 1 else float(k)
            if better == i:
                A[i,j] = val; A[j,i] = 1.0/val
            elif better == j:
                A[i,j] = 1.0/val; A[j,i] = val
            else:
                A[i,j] = 1.0; A[j,i] = 1.0

    gm = np.prod(A, axis=1) ** (1.0/n)
    w = normalize(gm)

    Aw = A @ w
    lam = float(np.sum(Aw / w) / n)
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = (CI / RI) if RI > 0 else 0.0
    return A, {names[i]: float(w[i]) for i in range(n)}, float(CR)

def build_scaled_subcriterion_weights(df_cat: pd.DataFrame, ranks_cfg: dict):
    rank_map = {c["name"]: float(c["rank"]) for c in ranks_cfg["category_ranks"]}
    rows = []
    for cat, g in df_cat.groupby("Category"):
        subs = g["Subcriterion"].tolist()
        if "BaseWeight" in g.columns and not g["BaseWeight"].isna().all():
            bw = g["BaseWeight"].fillna(0).astype(float).to_numpy()
            if bw.sum() <= 0: bw = np.ones(len(subs))
        else:
            bw = np.ones(len(subs))
        bw = normalize(bw)
        k = rank_map.get(cat, 1.0)
        for s, w in zip(subs, bw * k):
            rows.append((cat, s, float(w)))
    total = sum(w for _,_,w in rows) or 1.0
    tree: Dict[str, Dict[str, float]] = {}
    for c, s, w in rows:
        tree.setdefault(c, {})[s] = w/total
    return tree

def ensure_state():
    if "S" not in st.session_state:
        st.session_state.S = {
            "project_name": "",
            "locations": [],
            "mandatory": {},
            "active_locations": [],
            "pairwise": {},
            "local_weights": {},
            "global_scores": {},
            "category_breakdown": {},
            "last_saved": None
        }

# ---------- boot

st.set_page_config(page_title=APP_TITLE, layout="wide")
ensure_state()
S = st.session_state.S

meta = load_json(os.path.join(DATA_DIR, "report_meta.json"), {
    "project_default_name": "Refugee Camp Site Feasibility",
    "org_name": "", "author": "",
    "include_conditional_notes": True,
    "language": "en", "date_format": "YYYY-MM-DD",
    "footer_note": "", "references": []
})

ranks_cfg = load_json(os.path.join(DATA_DIR, "category_ranks.json"))
if not ranks_cfg:
    st.error("data/category_ranks.json not found.")
    st.stop()

map_cfg = load_json(os.path.join(DATA_DIR, "map_settings.json"), {
    "enable_map": True, "size_by": "global_score",
    "export_geojson": True, "crs": "EPSG:4326", "skip_map_option": True
})

try:
    df_cat = load_categories_from_excel(os.path.join(DATA_DIR, "Main categories and sub criteria.xlsx"))
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

mandatory_docx = os.path.join(DATA_DIR, "Mandatory Criteria Questions.docx")
try:
    mandatory_lines = load_docx_lines(mandatory_docx)
except Exception as e:
    st.error(f"Failed to read mandatory DOCX: {e}")
    st.stop()

weight_tree = build_scaled_subcriterion_weights(df_cat, ranks_cfg)

# ---------- sidebar

st.sidebar.title("Navigation")
step = st.sidebar.radio(
    "Go to step",
    ["1) Project Setup",
     "2) Mandatory Screening",
     "3) Pairwise (by Sub-criterion)",
     "4) Aggregate & Rank",
     "5) Outputs & Report",
     "⚙️ Save / Load / Inspect"]
)

with st.sidebar.expander("💾 Save / Export", expanded=False):
    if st.button("Save current session"):
        S["last_saved"] = datetime.utcnow().isoformat() + "Z"
        payload = json.dumps(S, indent=2)
        st.download_button("Download state.json", data=payload,
                           file_name=f"ahp_state_{int(time.time())}.json",
                           mime="application/json")

with st.sidebar.expander("📥 Load / Import", expanded=False):
    up = st.file_uploader("Upload a previously saved state.json", type=["json"])
    if up is not None:
        try:
            st.session_state.S = json.load(up)
            S = st.session_state.S
            st.success("State loaded successfully.")
        except Exception as e:
            st.error(f"Load failed: {e}")

# ---------- STEP 1

if step.startswith("1"):
    st.title("1) User & Project Setup")
    S["project_name"] = st.text_input(
        "Project name (optional)",
        value=S.get("project_name") or meta.get("project_default_name","")
    )

    st.subheader("Locations")
    count = st.number_input("How many locations do you want to compare?",
                            min_value=2, max_value=12,
                            value=max(2, len(S["locations"]) or 2), step=1)

    while len(S["locations"]) < count:
        S["locations"].append({"name": f"Camp {chr(65+len(S['locations']))}", "lat": None, "lon": None})
    if len(S["locations"]) > count:
        S["locations"] = S["locations"][:count]

    for i, loc in enumerate(S["locations"]):
        with st.container(border=True):
            S["locations"][i]["name"] = st.text_input(f"Location {i+1} name", value=loc["name"], key=f"nm_{i}")
            c1, c2 = st.columns(2)
            with c1:
                lat_in = st.text_input(f"{S['locations'][i]['name']} latitude (optional)",
                                       value="" if loc["lat"] is None else str(loc["lat"]), key=f"lat_{i}")
            with c2:
                lon_in = st.text_input(f"{S['locations'][i]['name']} longitude (optional)",
                                       value="" if loc["lon"] is None else str(loc["lon"]), key=f"lon_{i}")

            try:
                S["locations"][i]["lat"] = float(lat_in.strip()) if lat_in.strip() != "" else None
            except Exception:
                S["locations"][i]["lat"] = None
            try:
                S["locations"][i]["lon"] = float(lon_in.strip()) if lon_in.strip() != "" else None
            except Exception:
                S["locations"][i]["lon"] = None

    st.info("Coordinates are optional and used only for the map in Step 5.")

# ---------- STEP 2

elif step.startswith("2"):
    st.title("2) Mandatory Criteria Screening (YES / NO / CONDITIONAL)")
    st.caption("Locations with any 'NO' will be eliminated. 'CONDITIONAL' locations remain but will be flagged in the report.")

    if not S["locations"]:
        st.warning("Please add at least two locations in Step 1 first.")
        st.stop()

    st.write("**Mandatory Questions (from DOCX):**")
    for i, q in enumerate(mandatory_lines, start=1):
        st.write(f"{i}. {q}")

    for loc in S["locations"]:
        lname = loc["name"]
        S["mandatory"].setdefault(lname, {})
        with st.expander(f"{lname}"):
            for idx, q in enumerate(mandatory_lines):
                qid = f"q{idx+1}"
                default = S["mandatory"][lname].get(qid, "YES")
                ans = st.radio(q, ["YES","NO","CONDITIONAL"], horizontal=True,
                               index=["YES","NO","CONDITIONAL"].index(default), key=f"{lname}_{qid}")
                S["mandatory"][lname][qid] = ans

    active, eliminated, conditional = [], [], []
    for loc in S["locations"]:
        lname = loc["name"]
        answers = S["mandatory"].get(lname, {})
        if any(a == "NO" for a in answers.values()) or len(answers)==0:
            eliminated.append(lname)
        else:
            active.append(lname)
            if any(a == "CONDITIONAL" for a in answers.values()):
                conditional.append(lname)
    S["active_locations"] = active

    c1,c2,c3 = st.columns(3)
    with c1: st.success(f"Active: {', '.join(active) if active else 'None'}")
    with c2: st.warning(f"Eliminated (NO): {', '.join(eliminated) if eliminated else 'None'}")
    with c3:
        if conditional:
            st.info(f"Conditional: {', '.join(conditional)} (will be highlighted in the report)")

# ---------- STEP 3

elif step.startswith("3"):
    st.title("3) Pairwise Comparison — by Sub-criterion (Saaty 1–9)")
    names = S.get("active_locations") or [l["name"] for l in S["locations"]]
    if len(names) < 2:
        st.warning("At least two ACTIVE locations are required. Check Step 2.")
        st.stop()

    sublist = []
    for cat, subs in weight_tree.items():
        for sc, w in subs.items():
            sublist.append((cat, sc, w))

    done_count = sum(1 for (cat, sc, _) in sublist if f"{cat} :: {sc}" in S["local_weights"])
    st.progress(done_count / max(1,len(sublist)))
    st.caption(f"Completed sub-criteria: {done_count}/{len(sublist)}")

    with st.expander("Saaty scale (1–9) quick reference", expanded=False):
        st.markdown(
            "- **1** = Equal importance\n"
            "- **3** = Slightly more important\n"
            "- **5** = Clearly more important\n"
            "- **7** = Very strongly more important\n"
            "- **9** = Absolutely dominant\n"
            "- **2,4,6,8** = Intermediate values"
        )

    for (cat, sc, fixed_w) in sublist:
        key = f"{cat} :: {sc}"
        with st.container(border=True):
            st.subheader(key)
            st.caption(f"Fixed sub-weight (scaled): {fixed_w:.4f}")
            S["pairwise"].setdefault(key, {"pairs": {}})

            n = len(names)
            for i in range(n):
                for j in range(i+1, n):
                    left, right = names[i], names[j]
                    cols = st.columns([2,2,2,2])
                    with cols[0]:
                        choice = st.radio(
                            f"Which location performs better on '{sc}'?",
                            [left,"Equal",right],
                            key=f"{key}_c_{i}_{j}", horizontal=True
                        )
                    eq = (choice == "Equal")
                    default_k = 1 if eq else 3
                    with cols[1]:
                        inten = st.slider(
                            "Intensity (1–9, Saaty)",
                            1, 9, default_k,
                            key=f"{key}_s_{i}_{j}",
                            disabled=eq
                        )
                    if eq:
                        S["pairwise"][key]["pairs"][(i,j)] = (i, 1)
                    elif choice == left:
                        S["pairwise"][key]["pairs"][(i,j)] = (i, inten)
                    else:
                        S["pairwise"][key]["pairs"][(i,j)] = (j, inten)

            A, wloc, CR = ahp_from_pairs(names, S["pairwise"][key]["pairs"])
            S["local_weights"][key] = wloc
            S["pairwise"][key]["CR"] = CR

            st.write(pd.DataFrame(A, index=names, columns=names))
            st.write(pd.DataFrame({"Local Weight": wloc}).T)
            if CR > 0.10:
                st.error(f"CR = {CR:.3f} (> 0.10). It is recommended to review your judgments.")
            else:
                st.success(f"CR = {CR:.3f} (acceptable)")

# ---------- STEP 4

elif step.startswith("4"):
    st.title("4) Aggregate & Rank")

    names = S.get("active_locations") or [l["name"] for l in S["locations"]]
    if len(names) < 2:
        st.warning("At least two ACTIVE locations are required.")
        st.stop()

    subs = []
    for cat, subs_dict in weight_tree.items():
        for sc, fixed_w in subs_dict.items():
            subs.append((cat, sc, fixed_w, f"{cat} :: {sc}"))

    global_scores = {n: 0.0 for n in names}
    category_breakdown = {n: {cat: 0.0 for cat in weight_tree.keys()} for n in names}

    for (cat, sc, fixed_w, key) in subs:
        local = S["local_weights"].get(key)
        if not local:
            continue
        for n in names:
            lw = float(local.get(n, 0.0))
            global_scores[n] += lw * fixed_w
            category_breakdown[n][cat] += lw * fixed_w

    total = sum(global_scores.values()) or 1.0
    for k in global_scores: global_scores[k] /= total
    for n in category_breakdown:
        for cat in category_breakdown[n]:
            category_breakdown[n][cat] /= total

    S["global_scores"] = global_scores
    S["category_breakdown"] = category_breakdown

    df_rank = pd.DataFrame({"Global Score": global_scores}).sort_values("Global Score", ascending=False)
    st.dataframe(df_rank.style.format({"Global Score":"{:.4f}"}))
    top3 = list(df_rank.index)[:3]
    st.success(f"Top 3 Locations: {', '.join(top3)}")

# ---------- STEP 5

elif step.startswith("5"):
    st.title("5) Outputs & Visualisation")

    scores = S.get("global_scores", {})
    breakdown = S.get("category_breakdown", {})
    if not scores:
        st.warning("Please compute scores in Step 4 first.")
        st.stop()

    st.subheader("Global Score Comparison (Bar)")
    fig_bar = go.Figure(go.Bar(x=list(scores.keys()), y=list(scores.values())))
    fig_bar.update_layout(xaxis_title="Location", yaxis_title="Global Score")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Category Performance (Radar)")
    cats = list(weight_tree.keys())
    fig_rad = go.Figure()
    for n in scores.keys():
        r_vals = [breakdown[n].get(c, 0.0) for c in cats]
        r_vals.append(r_vals[0])
        cats_close = cats + [cats[0]]
        fig_rad.add_trace(go.Scatterpolar(r=r_vals, theta=cats_close, fill="toself", name=n))
    fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_rad, use_container_width=True)

    st.subheader("Mandatory Criteria Summary")
    mand_rows = []
    for loc in S["locations"]:
        lname = loc["name"]
        row = {"Location": lname}
        for idx, _ in enumerate(load_docx_lines(os.path.join(DATA_DIR, "Mandatory Criteria Questions.docx"))):
            qid = f"q{idx+1}"
            row[qid] = S["mandatory"].get(lname, {}).get(qid, "-")
        mand_rows.append(row)
    df_mand = pd.DataFrame(mand_rows)
    st.dataframe(df_mand)

    if map_cfg.get("enable_map", True):
        st.subheader("Geovisualization (optional)")
        has_lat = any(l.get("lat") is not None for l in S["locations"])
        has_lon = any(l.get("lon") is not None for l in S["locations"])
        if has_lat and has_lon:
            map_df = pd.DataFrame([{
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
                "name": loc.get("name"),
                "score": scores.get(loc.get("name"), 0.0)
            } for loc in S["locations"] if loc.get("lat") is not None and loc.get("lon") is not None])
            st.map(map_df, latitude="lat", longitude="lon", size="score")
            if map_cfg.get("export_geojson", True):
                gj = {
                    "type": "FeatureCollection",
                    "features": [
                        {"type":"Feature",
                         "properties":{"name": r["name"], "score": r["score"]},
                         "geometry":{"type":"Point","coordinates":[r["lon"], r["lat"]]}}
                        for _, r in map_df.iterrows()
                    ]
                }
                st.download_button("Download GeoJSON", data=json.dumps(gj, indent=2),
                                   file_name="locations.geojson", mime="application/geo+json")
        else:
            st.caption("Add coordinates in Step 1 to enable the map.")

    st.subheader("PDF Report")
    bar_path = os.path.join(OUT_DIR, f"bar_{int(time.time())}.png")
    rad_path = os.path.join(OUT_DIR, f"rad_{int(time.time())}.png")
    fig_bar.write_image(bar_path, engine="kaleido", scale=2)
    fig_rad.write_image(rad_path, engine="kaleido", scale=2)

    pdf_path = os.path.join(OUT_DIR, f"report_{int(time.time())}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4; y = H - 40
    c.setFont("Helvetica-Bold", 14); c.drawString(40, y, meta.get("project_default_name", APP_TITLE)); y -= 18
    c.setFont("Helvetica", 10); c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()}Z"); y -= 14
    c.drawString(40, y, f"Author: {meta.get('author','')}   Org: {meta.get('org_name','')}"); y -= 18
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Global Scores"); y -= 14
    c.setFont("Helvetica", 9)
    for name, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        c.drawString(50, y, f"{name}: {sc:.4f}"); y -= 12
    y -= 10
    try:
        c.drawImage(ImageReader(bar_path), 40, max(60, y-220), width=520, height=220, preserveAspectRatio=True, anchor='nw'); y -= 240
    except: pass
    try:
        c.drawImage(ImageReader(rad_path), 40, max(60, y-220), width=520, height=220, preserveAspectRatio=True, anchor='nw'); y -= 240
    except: pass
    y -= 10
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Mandatory Summary"); y -= 14
    c.setFont("Helvetica", 8)
    for _, r in df_mand.iterrows():
        loc = r["Location"]; vals = [v for k,v in r.items() if k!="Location"]
        c.drawString(50, y, f"{loc}: " + ", ".join(vals[:6]) + ("..." if len(vals)>6 else "")); y -= 10
        if y < 80: c.showPage(); y = H - 40
    c.setFont("Helvetica", 8); c.drawString(40, 40, meta.get("footer_note","")); c.save()
    st.success("PDF generated.")
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF", data=f.read(),
                           file_name=os.path.basename(pdf_path), mime="application/pdf")

    # ---- DXF download (hexagonal layout) ----
    st.subheader("Spatial Layout (DXF) — download")
    DXF_DIR = os.path.join(DATA_DIR, "layouts")
    os.makedirs(DXF_DIR, exist_ok=True)

    # Varsayılan olarak bu dosyayı öne çıkaralım:
    preferred_dxf = "Hexagonal spatial layout plan.dxf"

    up_dxf = st.file_uploader("Upload a DXF to include in this session (optional)", type=["dxf"])
    uploaded_path = None
    if up_dxf is not None:
        uploaded_path = os.path.join(OUT_DIR, up_dxf.name)
        with open(uploaded_path, "wb") as g:
            g.write(up_dxf.read())
        st.success(f"Uploaded: {up_dxf.name}")

    available_dxf = [f for f in os.listdir(DXF_DIR) if f.lower().endswith(".dxf")]
    pref_index = 0
    if available_dxf:
        if preferred_dxf in available_dxf:
            pref_index = available_dxf.index(preferred_dxf)
        elif S.get("active_locations"):
            for i, f in enumerate(available_dxf):
                base = os.path.splitext(f)[0].strip().lower()
                if base in [n.strip().lower() for n in S["active_locations"]]:
                    pref_index = i; break

    if not available_dxf and uploaded_path is None:
        st.info("Place your AutoCAD drawings (.dxf) into data/layouts/ to enable downloads.")
    else:
        choices = []
        if uploaded_path is not None:
            choices.append(f"[uploaded] {os.path.basename(uploaded_path)}")
        choices += available_dxf
        sel = st.selectbox("Select a DXF file to download", choices, index=0 if uploaded_path else pref_index)
        path = uploaded_path if sel.startswith("[uploaded]") else os.path.join(DXF_DIR, sel)
        fname = os.path.basename(path)
        with open(path, "rb") as fh:
            st.download_button("📥 Download selected DXF", data=fh, file_name=fname, mime="application/octet-stream")

# ---------- SETTINGS

else:
    st.title("⚙️ Save / Load / Inspect")
    st.write("**Fixed sub-criterion weights (scaled by category ranks)**")
    rows = []
    for cat, subs in weight_tree.items():
        for sc, w in subs.items():
            rows.append({"Category": cat, "Sub-criterion": sc, "Weight": w})
    st.dataframe(pd.DataFrame(rows).sort_values(["Category","Sub-criterion"]))

    st.write("**Current session keys**")
    st.json(S)
