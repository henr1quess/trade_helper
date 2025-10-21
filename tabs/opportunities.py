from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = Path.cwd()
RAW_DIR = PROJECT_DIR / "raw"
LATEST_SNAPSHOT_PATH = RAW_DIR / "latest_snapshot.json"
WATCHLIST_PATH = PROJECT_DIR / "watchlist.json"

# --------------------------- IO utils ---------------------------
def _read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------------------------- data loading ---------------------------
def _load_snapshot_df(snapshot_path: Path) -> pd.DataFrame:
    snap = _read_json(snapshot_path, {})
    recs = snap.get("records") or []
    df = pd.DataFrame(recs)

    need = ["item_id","item_name","trading_category","trading_family","trading_group","top_buy","low_sell"]
    for c in need:
        if c not in df.columns:
            df[c] = None

    df["top_buy"] = pd.to_numeric(df["top_buy"], errors="coerce")
    df["low_sell"] = pd.to_numeric(df["low_sell"], errors="coerce")

    def _roi(row):
        tb, ls = row["top_buy"], row["low_sell"]
        if pd.isna(tb) or pd.isna(ls):
            return np.nan
        buy = tb + 0.01
        sell = ls - 0.01
        if buy <= 0:
            return np.nan
        return (sell - buy) / buy * 100.0

    df["roi_pct"] = df.apply(_roi, axis=1)
    return df

# --------------------------- formatting ---------------------------
def _fmt_money(x):
    if pd.isna(x):
        return "-"
    return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _roi_badge(v):
    if pd.isna(v):
        return "-"
    pct = float(v)
    if pct < 0:   dot = "🔴"
    elif pct < 10: dot = "🟡"
    elif pct < 15: dot = "🟠"
    else:          dot = "🟢"
    return f"{dot} {pct:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")

# --------------------------- watchlist ---------------------------
def _load_watchlist_ids() -> set[str]:
    data = _read_json(WATCHLIST_PATH, {"items": []})
    ids = set()
    if isinstance(data, list):  # legado
        for el in data:
            if isinstance(el, dict):
                iid = el.get("item_id") or el.get("id")
                if iid: ids.add(str(iid))
    elif isinstance(data, dict):
        for el in data.get("items", []):
            if isinstance(el, dict):
                iid = el.get("item_id") or el.get("id")
                if iid: ids.add(str(iid))
    return ids

def _save_watchlist_ids(ids: set[str], df_lookup: pd.DataFrame):
    look = df_lookup.loc[:, ["item_id","item_name"]].drop_duplicates()
    look["item_id"] = look["item_id"].astype(str)
    look = look[look["item_id"].isin(ids)]
    payload = {"items": look.to_dict(orient="records")}
    _write_json(WATCHLIST_PATH, payload)

# --------------------------- filtros UI ---------------------------
def _filters_cascade(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### Filtros por Category / Family / Group")

    # ---------------- Category ----------------
    cats_all = sorted(df["trading_category"].dropna().astype(str).unique().tolist())
    default_cats = [c for c in ["Resources","Utilities"] if c in cats_all]
    col_c1, col_c2 = st.columns((1,1), vertical_alignment="center")
    with col_c1:
        sel_cats = st.multiselect("Category (incluir)", cats_all, default=default_cats,
                                  placeholder="Selecione 1+ categorias")
    with col_c2:
        ex_cats = st.multiselect("Category (excluir)", cats_all, placeholder="Opcional: excluir categorias")

    df_cat = df.copy()
    if sel_cats:
        df_cat = df_cat[df_cat["trading_category"].astype(str).isin(sel_cats)]
    if ex_cats:
        df_cat = df_cat[~df_cat["trading_category"].astype(str).isin(ex_cats)]

    # ---------------- Family ----------------
    fam_opts = sorted(df_cat["trading_family"].dropna().astype(str).unique().tolist())
    fam_defaults = [f for f in ["RefinedResources","RawResources"] if f in fam_opts]
    col_f1, col_f2 = st.columns((1,1), vertical_alignment="center")
    with col_f1:
        sel_fams = st.multiselect("Family (incluir)", fam_opts, default=fam_defaults,
                                  placeholder="Selecione 1+ families")
    with col_f2:
        ex_fams = st.multiselect("Family (excluir)", fam_opts, placeholder="Opcional: excluir families")

    df_fam = df_cat.copy()
    if sel_fams:
        df_fam = df_fam[df_fam["trading_family"].astype(str).isin(sel_fams)]
    if ex_fams:
        df_fam = df_fam[~df_fam["trading_family"].astype(str).isin(ex_fams)]

    # ---------------- Group ----------------
    grp_opts = sorted(df_fam["trading_group"].dropna().astype(str).unique().tolist())
    col_g1, col_g2 = st.columns((1,1), vertical_alignment="center")
    with col_g1:
        sel_grps = st.multiselect("Group (incluir)", grp_opts, placeholder="Selecione 1+ groups")
    with col_g2:
        ex_grps = st.multiselect("Group (excluir)", grp_opts, placeholder="Opcional: excluir groups")

    df_grp = df_fam.copy()
    if sel_grps:
        df_grp = df_grp[df_grp["trading_group"].astype(str).isin(sel_grps)]
    if ex_grps:
        df_grp = df_grp[~df_grp["trading_group"].astype(str).isin(ex_grps)]

    return df_grp


def _roi_slider_bounds(df: pd.DataFrame):
    vals = pd.to_numeric(df["roi_pct"], errors="coerce").dropna()
    if vals.empty:
        return (0.0, 0.0, (0.0, 0.0))
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return (float(np.floor(lo)), float(np.ceil(hi)), (float(np.floor(lo)), float(np.ceil(hi))))

# --------------------------- main render ---------------------------
def render(latest_snapshot_path: Path = LATEST_SNAPSHOT_PATH, last_meta_path: Path | None = None):
    st.header("Oportunidades")

    df = _load_snapshot_df(latest_snapshot_path)
    if df.empty:
        st.info("Snapshot vazio. Coleta/Processamento necessário.")
        return

    df_filt = _filters_cascade(df)

    rmin, rmax, rdefault = _roi_slider_bounds(df_filt)
    roi_sel = st.slider("ROI (%)",
                        min_value=rmin, max_value=rmax, value=rdefault, step=1.0,
                        help="Faixa baseada nos percentis (1–99%) do conjunto filtrado.")
    df_filt = df_filt[df_filt["roi_pct"].between(roi_sel[0], roi_sel[1], inclusive="both")]

    wl_ids = _load_watchlist_ids()

    show_mode = st.radio("Exibir", ["Todos","Somente watchlist"], horizontal=True)
    if show_mode == "Somente watchlist":
        df_filt = df_filt[df_filt["item_id"].astype(str).isin(wl_ids)]

    st.write(f"Oportunidades: **{len(df_filt):,}**".replace(",", "."))

    # -------- tabela ----------
    tbl = pd.DataFrame({
        "★": df_filt["item_id"].astype(str).isin(wl_ids),
        "Item": df_filt["item_name"].fillna(""),
        "Category": df_filt["trading_category"].fillna(""),
        "Family": df_filt["trading_family"].fillna(""),
        "Group": df_filt["trading_group"].fillna(""),
        "Top Buy": df_filt["top_buy"].map(_fmt_money),
        "Low Sell": df_filt["low_sell"].map(_fmt_money),
        "ROI %": df_filt["roi_pct"].apply(_roi_badge),
    })
    keys = df_filt.loc[:, ["item_id","item_name"]].copy()
    keys["item_id"] = keys["item_id"].astype(str)

    edited = st.data_editor(
        tbl,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",  # ← remove a coluna de operações/drag; ⭐ vira a 1ª coluna
        column_order=["★","Item","Category","Family","Group","Top Buy","Low Sell","ROI %"],
        column_config={
            "★": st.column_config.CheckboxColumn("★", help="Favoritar / remover", width=36),  # largura real pequena
            "ROI %": st.column_config.TextColumn("ROI %", help="ROI com sinalização por cor (emoji)"),
        },
        disabled=["Item","Category","Family","Group","Top Buy","Low Sell","ROI %"],
    )

    # salva automático ao detectar mudança
    if "★" in edited.columns:
        prev = tbl["★"].astype(bool).to_list()
        new  = edited["★"].astype(bool).to_list()
        if new != prev:
            for iid, flag in zip(keys["item_id"].tolist(), new):
                if flag: wl_ids.add(iid)
                else:    wl_ids.discard(iid)
            _save_watchlist_ids(wl_ids, df)
            st.toast("Watchlist atualizada.", icon="⭐")

    st.caption("Clique na ⭐ (primeira coluna) para favoritar — salva automaticamente. "
               "ROI é calculado com Top Buy + 0.01 e Low Sell − 0.01.")

if __name__ == "__main__":
    st.set_page_config(page_title="Oportunidades", layout="wide")
    render(LATEST_SNAPSHOT_PATH)
