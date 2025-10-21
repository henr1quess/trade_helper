# tabs/calculadora.py
from __future__ import annotations
import streamlit as st

def render():
    st.subheader("🧮 Calculadora de margem (com taxas)")

    c1, c2, c3 = st.columns(3)
    with c1:
        buy_price = st.number_input(
            "Preço de compra (Buy Order)", min_value=0.0, value=1.00, step=0.01, format="%.2f", key="calc_buy_price"
        )
    with c2:
        sell_price = st.number_input(
            "Preço de venda (Sell Order)", min_value=0.0, value=1.50, step=0.01, format="%.2f", key="calc_sell_price"
        )
    with c3:
        qty = st.number_input("Quantidade", min_value=1, value=100, step=1, key="calc_qty")

    st.markdown("**Taxas (em %)**")
    c4, c5, c6 = st.columns(3)
    with c4:
        buy_fee_pct = st.number_input("Taxa de compra (%)", min_value=0.0, value=0.00, step=0.05, format="%.2f", key="calc_buy_fee")
    with c5:
        sell_fee_pct = st.number_input("Taxa de venda (%)", min_value=0.0, value=6.50, step=0.10, format="%.2f", key="calc_sell_fee")
    with c6:
        extra_fee_pct = st.number_input("Outras taxas (%)", min_value=0.0, value=0.00, step=0.05, format="%.2f", key="calc_extra_fee")

    buy_cost_unit  = buy_price * (1 + buy_fee_pct/100.0)
    sell_recv_unit = sell_price * (1 - (sell_fee_pct + extra_fee_pct)/100.0)

    lucro_unit = sell_recv_unit - buy_cost_unit
    roi_pct = (lucro_unit / buy_cost_unit) if buy_cost_unit > 0 else 0.0

    st.divider()
    st.metric("Lucro por unidade", f"{lucro_unit:.2f}")
    st.metric("ROI", f"{roi_pct*100:.2f}%")
    st.metric("Lucro total (qtd)", f"{(lucro_unit*qty):.2f}")
    st.caption("Dica: se usar flip (buy = top_buy+0.01, sell = low_sell-0.01), aplique esses ajustes nos campos.")
