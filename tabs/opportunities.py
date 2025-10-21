# tabs/opportunities.py
from __future__ import annotations
from pathlib import Path
import json
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
    snap = _read_json(latest_snapshot_path)
    if not snap or "records" not in snap:
        return pd.DataFrame(columns=[
            "item_name","top_buy","low_sell","buy_qty","sell_qty","timestamp"
        ])
    df = pd.DataFrame(snap["records"])
    # garante campos essenciais
    keep = ["item_name","top_buy","low_sell","buy_qty","sell_qty","timestamp"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    return df[keep]

# ---------- cálculo das oportunidades ----------
def _compute_opps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    # numeric
    for c in ["top_buy","low_sell","buy_qty","sell_qty"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # precisa ter as duas pontas
    x = x[x["top_buy"].notna() & x["low_sell"].notna()]
    if x.empty:
        return x

    x["flip_buy"]  = x["top_buy"] + 0.01
    x["flip_sell"] = x["low_sell"] - 0.01
    x["spread"]    = x["flip_sell"] - x["flip_buy"]
    x["roi_pct"]   = x["spread"] / x["flip_buy"]
    x["lucro_un"]  = x["spread"]

    # colunas finais (em ordem)
    cols = [
        "item_name",            # → Item
        "flip_buy",             # → Comprar (flip)
        "flip_sell",            # → Vender (flip)
        "spread",               # → Spread
        "roi_pct",              # → ROI %
        "lucro_un",             # → Lucro / un
        "top_buy",              # → Top Buy (mercado)
        "low_sell",             # → Low Sell (mercado)
        "buy_qty",              # → Quantidade na compra
        "sell_qty",             # → Quantidade na venda
        "timestamp",            # → Timestamp do snapshot
    ]
    for c in cols:
        if c not in x.columns:
            x[c] = None
    return x[cols].reset_index(drop=True)

# ---------- UI ----------
def render(latest_snapshot_path: Path, last_meta_path: Path):
    st.subheader("📈 Oportunidades")

    # CSS leve para aumentar legibilidade
    st.markdown("""
        <style>
        .stDataFrame {font-size: 0.95rem;}
        div[data-testid="stMetricValue"] {font-size: 1.1rem;}
        </style>
    """, unsafe_allow_html=True)

    raw = _load_latest_records(latest_snapshot_path)
    if raw.empty:
        st.info("Nenhum snapshot processado ainda. Vá em **☁️ Coletar snapshot** e depois **⚙️ Processar snapshot**.")
        return

    opp = _compute_opps(raw)

    # Filtros e ordenação
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        min_roi = st.number_input("ROI mínimo (%)", value=0.0, step=0.5, format="%.2f")
    with c2:
        min_lucro = st.number_input("Lucro mínimo (por un.)", value=0.0, step=0.1, format="%.2f")
    with c3:
        order_by = st.selectbox("Ordenar por", ["ROI %", "Spread", "Lucro/un"], index=0)
    with c4:
        busca = st.text_input("Buscar por nome do item", "")

    if not opp.empty:
        if min_roi:
            opp = opp[opp["roi_pct"] >= (min_roi/100.0)]
        if min_lucro:
            opp = opp[opp["lucro_un"] >= min_lucro]
        if busca.strip():
            s = busca.lower().strip()
            opp = opp[opp["item_name"].fillna("").str.lower().str.contains(s)]

        # Ordenação
        if order_by == "ROI %":
            opp = opp.sort_values(["roi_pct","spread"], ascending=[False, False], kind="mergesort")
        elif order_by == "Spread":
            opp = opp.sort_values(["spread","roi_pct"], ascending=[False, False], kind="mergesort")
        else:  # Lucro/un
            opp = opp.sort_values(["lucro_un","roi_pct"], ascending=[False, False], kind="mergesort")

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Oportunidades", f"{len(opp)}")
    if not opp.empty:
        # melhor ROI
        top_roi = opp.iloc[0]
        k2.metric("Melhor ROI", f"{float(top_roi['roi_pct']*100):.2f}%")
        # melhor lucro unitário
        best_profit = opp.iloc[opp["lucro_un"].idxmax()]
        k3.metric("Melhor lucro/un.", f"{float(best_profit['lucro_un']):.2f}")
    else:
        k2.metric("Melhor ROI", "—")
        k3.metric("Melhor lucro/un.", "—")

    # Configuração de colunas (PT-BR)
    col_cfg = {
        "item_name": st.column_config.TextColumn("Item", help="Nome do item"),
        "flip_buy":  st.column_config.NumberColumn("Comprar (flip)", help="top_buy + 0.01", format="%.2f"),
        "flip_sell": st.column_config.NumberColumn("Vender (flip)", help="low_sell - 0.01", format="%.2f"),
        "spread":    st.column_config.NumberColumn("Spread", help="Vender (flip) − Comprar (flip)", format="%.2f"),
        "roi_pct":   st.column_config.NumberColumn("ROI %", help="Spread ÷ Comprar (flip)", format="%.2f%%"),
        "lucro_un":  st.column_config.NumberColumn("Lucro por unidade", format="%.2f"),
        "top_buy":   st.column_config.NumberColumn("Top Buy (mercado)", help="Maior preço de compra", format="%.2f"),
        "low_sell":  st.column_config.NumberColumn("Low Sell (mercado)", help="Menor preço de venda", format="%.2f"),
        "buy_qty":   st.column_config.NumberColumn("Qtd. na compra", help="Soma das quantidades em buy orders", format="%d"),
        "sell_qty":  st.column_config.NumberColumn("Qtd. na venda", help="Soma das quantidades em sell orders", format="%d"),
        "timestamp": st.column_config.TextColumn("Snapshot"),
    }

    # Renderização
    if opp.empty:
        st.warning("Nenhum item atendeu aos filtros.")
        return

    st.dataframe(
        opp,
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
    )

    # rodapé enxuto
    st.caption("Dica: 'Comprar (flip)' = Top Buy + 0.01 | 'Vender (flip)' = Low Sell − 0.01. Ajuste filtros acima para refinar.")
