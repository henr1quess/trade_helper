# tabs/opportunities.py
from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd
import streamlit as st

# ---------- helpers de IO ----------
def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _load_latest_records(latest_snapshot_path: Path) -> pd.DataFrame:
    """
    Carrega raw/latest_snapshot.json e devolve um DataFrame com:
      item_id, item_name, trading_category, trading_family, trading_group,
      top_buy, low_sell, buy_qty, sell_qty
    """
    snap = _read_json(latest_snapshot_path)
    if not isinstance(snap, dict):
        return pd.DataFrame()
    records = snap.get("records", [])
    if not isinstance(records, list):
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)

    # normalizações numéricas
    for col in ["top_buy", "low_sell"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["buy_qty", "sell_qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # garantir colunas
    for col in [
        "item_id", "item_name",
        "trading_category", "trading_family", "trading_group",
        "top_buy", "low_sell", "buy_qty", "sell_qty"
    ]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df

def _prep_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula campos de flip (usados só para cálculo, não exibidos):
      flip_buy  = top_buy + 0.01
      flip_sell = low_sell - 0.01
      spread    = flip_sell - flip_buy
      roi_pct   = spread / flip_buy
      lucro_un  = spread
    Retorna apenas linhas com spread > 0.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "item_name","trading_category","trading_family","trading_group",
            "top_buy","low_sell","roi_pct","item_id",
            "flip_buy","flip_sell","spread","lucro_un"  # backend only
        ])

    x = df.copy()

    x["flip_buy"]  = pd.to_numeric(x["top_buy"], errors="coerce") + 0.01
    x["flip_sell"] = pd.to_numeric(x["low_sell"], errors="coerce") - 0.01

    x["spread"]   = x["flip_sell"] - x["flip_buy"]
    x["roi_pct"]  = x["spread"] / x["flip_buy"]
    x["lucro_un"] = x["spread"]

    cols = [
        "item_name",
        "trading_category","trading_family","trading_group",
        "top_buy","low_sell",
        "roi_pct",
        # backend (não exibidas):
        "flip_buy","flip_sell","spread","lucro_un",
        "item_id",
    ]
    x = x.reindex(columns=cols)

    x = x.dropna(subset=["top_buy", "low_sell", "roi_pct"])
    x = x[x["spread"].notna() & (x["spread"] > 0)]

    x = x.sort_values(["roi_pct", "spread"], ascending=[False, False]).reset_index(drop=True)
    return x

def _winsorized_bounds(s: pd.Series, low_q=0.01, high_q=0.99) -> tuple[float, float]:
    """Retorna (min,max) por percentis para ignorar outliers (em FRAÇÃO)."""
    s = pd.to_numeric(s, errors="coerce")
    s = s[~s.isna() & ~s.isin([math.inf, -math.inf])]
    if s.empty:
        return (0.0, 1.0)
    lo = float(s.quantile(low_q))
    hi = float(s.quantile(high_q))
    if lo == hi:
        lo -= 0.01
        hi += 0.01
    return (lo, hi)

# ---- badge de ROI (círculo colorido + número) ----
def _roi_badge(v: float) -> str:
    """
    Converte ROI fração (0.15=15%) em um badge colorido + número.
    Regras:
      < 0%  -> 🔴
      0–10% -> 🟡
      10–15%-> 🟠
      ≥ 15% -> 🟢
    """
    try:
        pct = float(v) * 100.0
    except Exception:
        return ""
    if pct < 0:
        dot = "🔴"
    elif pct < 10:
        dot = "🟡"
    elif pct < 15:
        dot = "🟠"
    else:
        dot = "🟢"
    return f"{dot} {pct:,.2f}%"

# ---------- UI principal ----------
def render(latest_snapshot_path: Path, last_meta_path: Path | None = None):
    st.header("Oportunidades")

    # carregar dados e preparar oportunidades
    df = _load_latest_records(Path(latest_snapshot_path))
    opp_all = _prep_opportunities(df)

    # ==========================
    #  Filtros em Cascata (UI)
    # ==========================
    DEFAULT_CATS = ["Resources", "Utilities"]

    with st.expander("Filtros por Category / Family / Group", expanded=False):
        c1, c2, c3 = st.columns(3)

        # Opções de Category
        all_cats = sorted([v for v in opp_all["trading_category"].dropna().astype(str).unique()])
        default_cats_available = [c for c in DEFAULT_CATS if c in all_cats]

        # CATEGORY (pré-seleciona Resources/Utilities quando existirem)
        sel_cats = st.multiselect(
            "Category",
            options=all_cats,
            default=default_cats_available,
            placeholder="Selecione 1+ categorias",
            key="filter_cats",
            help="Pré-selecionado com Resources e Utilities (quando existirem).",
        )

        # Limita DF para derivar famílias apenas das categorias escolhidas
        df_for_fams = opp_all if not sel_cats else opp_all[opp_all["trading_category"].astype(str).isin(sel_cats)]
        fam_options = sorted([v for v in df_for_fams["trading_family"].dropna().astype(str).unique()])

        # Se havia families selecionadas que não existem mais, limpamos
        if "filter_fams" in st.session_state:
            st.session_state.filter_fams = [f for f in st.session_state.filter_fams if f in fam_options]

        # FAMILY (mostra só families das categories)
        sel_fams = st.multiselect(
            "Family",
            options=fam_options,
            default=st.session_state.get("filter_fams", []),
            placeholder="Selecione 1+ families",
            key="filter_fams",
            help="Mostra apenas families dentro das categories selecionadas.",
        )

        # Limita DF para derivar groups apenas das families escolhidas (e categories)
        df_for_groups = df_for_fams if not sel_fams else df_for_fams[df_for_fams["trading_family"].astype(str).isin(sel_fams)]
        group_options = sorted([v for v in df_for_groups["trading_group"].dropna().astype(str).unique()])

        # Limpa groups inválidos quando options mudam
        if "filter_groups" in st.session_state:
            st.session_state.filter_groups = [g for g in st.session_state.filter_groups if g in group_options]

        # GROUP (mostra só groups das families/category atuais)
        sel_groups = st.multiselect(
            "Group",
            options=group_options,
            default=st.session_state.get("filter_groups", []),
            placeholder="Selecione 1+ groups",
            key="filter_groups",
            help="Mostra apenas groups dentro das families selecionadas (e das categories).",
        )

    # aplica filtros de dimensão ao dataset principal
    opp = opp_all.copy()
    if sel_cats:
        opp = opp[opp["trading_category"].astype(str).isin(sel_cats)]
    if sel_fams:
        opp = opp[opp["trading_family"].astype(str).isin(sel_fams)]
    if sel_groups:
        opp = opp[opp["trading_group"].astype(str).isin(sel_groups)]

    # ==========================
    #  Filtros Numéricos (ROI)
    # ==========================
    roi_lo, roi_hi = _winsorized_bounds(opp["roi_pct"] if not opp.empty else pd.Series([0.0, 1.0]))
    col1, col2 = st.columns([2, 1])
    with col1:
        roi_rng = st.slider(
            "ROI (%)",
            min_value=float(round(roi_lo * 100, 2)),
            max_value=float(round(roi_hi * 100, 2)),
            value=(float(round(roi_lo * 100, 2)), float(round(roi_hi * 100, 2))),
            step=0.25,
            help="Faixa de ROI (em %) baseada na seleção atual, cortando outliers (1º–99º percentil).",
        )
    with col2:
        lucro_min = st.number_input(
            "Lucro mínimo (por un.)",
            min_value=0.0,
            value=0.0,
            step=0.10,
            format="%.2f",
            help="Mantém itens com lucro por unidade >= valor informado.",
        )

    roi_min_f, roi_max_f = roi_rng[0] / 100.0, roi_rng[1] / 100.0
    opp = opp[(opp["roi_pct"] >= roi_min_f) & (opp["roi_pct"] <= roi_max_f)]
    if lucro_min and lucro_min > 0:
        opp = opp[pd.to_numeric(opp["lucro_un"], errors="coerce") >= float(lucro_min)]

    # ==========================
    #  Exibição
    # ==========================
    opp_display = opp[[
        "item_name",
        "trading_category", "trading_family", "trading_group",
        "top_buy", "low_sell",
        "roi_pct",
    ]].copy()

    opp_display = opp_display.rename(columns={
        "item_name": "Item",
        "trading_category": "Category",
        "trading_family": "Family",
        "trading_group": "Group",
        "top_buy": "Top Buy",
        "low_sell": "Low Sell",
        "roi_pct": "ROI %",
    })

    # formatadores
    def _fmt_money(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return ""

    def _fmt_roi_cell(v):
        return _roi_badge(v)  # círculo colorido + número já formatado

    if opp_display.empty:
        st.metric("Oportunidades", 0)
        st.warning("Nenhum item atendeu aos filtros.")
        return

    st.metric("Oportunidades", len(opp_display))

    # Estética: cabeçalhos maiores, pesos e alinhamentos
    styler = (
        opp_display
        .style
        .format({
            "Top Buy": _fmt_money,
            "Low Sell": _fmt_money,
            "ROI %": _fmt_roi_cell,
        })
        .set_properties(subset=["Item", "Category", "Family", "Group"], **{
            "text-align": "left",
            "white-space": "nowrap",
        })
        .set_properties(subset=["Top Buy", "Low Sell", "ROI %"], **{
            "text-align": "right",
            "white-space": "nowrap",
        })
        .set_table_styles([
            {"selector": "th.col_heading", "props": [("font-size", "1.1rem"), ("font-weight", "800"), ("padding", "10px 14px")]},
            {"selector": "tbody td",       "props": [("padding", "9px 14px"), ("font-size", "0.95rem")]},
        ])
    )

    st.dataframe(styler, use_container_width=True, hide_index=True)

    st.caption(
        "ROI (badge): 🔴 <0% • 🟡 0–10% • 🟠 10–15% • 🟢 ≥15%. "
        "Slider usa percentis (1–99%) da seleção atual. "
        "ROI é calculado internamente com Top Buy + 0.01 e Low Sell − 0.01."
    )
