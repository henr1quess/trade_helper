# New World Helper — Histórico | Oportunidades | Importar preços | Cadastro | Calculadora
# Run: streamlit run streamlit_app.py

import json
import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="New World Helper", page_icon="🪙", layout="wide")

# --------------------------------------------------------------------------------------
# Paths & persistence
# --------------------------------------------------------------------------------------
SCRIPT_DIR = Path(os.getcwd())
HOME_CFG  = Path.home() / ".nw_flip_config.json"
LOCAL_CFG = SCRIPT_DIR / "nw_flip_config.json"
CFG_CANDIDATES = [LOCAL_CFG, HOME_CFG]

HISTORY_PATH = SCRIPT_DIR / "history.json"
LEGACY_WATCHLIST = SCRIPT_DIR / "watchlist.json"
HISTORY_READ_CANDIDATES = [HISTORY_PATH, LEGACY_WATCHLIST]

ITEMS_PATH = SCRIPT_DIR / "items.json"  # master data of items (cadastro)

# --------------------------------------------------------------------------------------
# Helpers (I/O)
# --------------------------------------------------------------------------------------
def load_json_records(path: Path, cols=None):
    try:
        df = pd.read_json(path, orient="records")
        if cols:
            for c in cols:
                if c not in df.columns:
                    df[c] = None
        return df
    except Exception:
        return pd.DataFrame(columns=cols or [])

def load_history():
    # agora inclui colunas para liquidez (timestamps de fill)
    base_cols = ["timestamp","item","buy_price","sell_price","buy_duration","sell_duration",
                 "buy_placed_ts","buy_filled_ts","sell_filled_ts"]
    for p in HISTORY_READ_CANDIDATES:
        if p.exists():
            return load_json_records(p, base_cols), p
    return pd.DataFrame(columns=base_cols), None

def save_history(df: pd.DataFrame):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(HISTORY_PATH, orient="records", indent=2)

# Tags helpers
def ensure_list_tags(x):
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [s.strip() for s in str(x).split(",") if s.strip()]

def stringify_tags(x):
    lst = ensure_list_tags(x)
    return ", ".join(lst)

def load_items():
    return load_json_records(ITEMS_PATH, ["item","categoria","peso","stack_max","tags"])

def save_items(df: pd.DataFrame):
    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["item","categoria","peso","stack_max","tags"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df["tags"] = df["tags"].apply(ensure_list_tags)
    df.to_json(ITEMS_PATH, orient="records", indent=2, force_ascii=False)

def load_first_existing(paths):
    for p in paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p
            except Exception:
                pass
    return {}, None

def save_to_all(cfg, paths):
    saved = []
    for p in paths:
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            saved.append(str(p))
        except Exception:
            pass
    return saved

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def to_utc_iso(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def parse_iso(s):
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

# --------------------------------------------------------------------------------------
# Config (calibração + taxa)
# --------------------------------------------------------------------------------------
sell_defaults = {1: {"S": 1.0, "Q": 1844, "F": 28.65000},
                 3: {"S": 1.0, "Q": 1844, "F": 36.11000},
                 7: {"S": 1.0, "Q": 1844, "F": 43.94000},
                 14: {"S": 1.0, "Q": 1844, "F": 52.54000}}
buy_defaults  = {1: {"B": 1.0, "Q": 1000, "F": 15.72000},
                 3: {"B": 1.0, "Q": 1000, "F": 19.93000},
                 7: {"B": 1.0, "Q": 1000, "F": 24.53000},
                 14: {"B": 1.0, "Q": 1000, "F": 29.90000}}

def rate(F, P, Q): return (F/(P*Q)) if P*Q>0 else 0.0
default_sell_rates = {str(k): rate(v["F"], v["S"], v["Q"]) for k, v in sell_defaults.items()}
default_buy_rates  = {str(k): rate(v["F"], v["B"], v["Q"]) for k, v in buy_defaults.items()}

cfg, cfg_path = load_first_existing(CFG_CANDIDATES)
tax_pct = float(cfg.get("tax_pct", 5.0))
sell_rates_cfg = cfg.get("sell_rates") or default_sell_rates
buy_rates_cfg  = cfg.get("buy_rates")  or default_buy_rates

if "sell_rates" not in st.session_state:
    st.session_state.sell_rates = {int(k): float(v) for k, v in sell_rates_cfg.items()}
if "buy_rates" not in st.session_state:
    st.session_state.buy_rates = {int(k): float(v) for k, v in buy_rates_cfg.items()}
if "tax_pct" not in st.session_state:
    st.session_state.tax_pct = tax_pct

# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------
DURATIONS = [1,3,7,14]
LIST_COL_AVAILABLE = hasattr(st.column_config, "ListColumn")

def auto_fee(total_value, rate_dict, duration):
    r = None
    if isinstance(rate_dict, dict):
        r = rate_dict.get(duration, rate_dict.get(int(duration), rate_dict.get(str(duration))))
    return total_value * (r or 0.0)

def compute_metrics(buy_price, sell_price, buy_duration=3, sell_duration=3, tax_pct=5.0):
    rtax = tax_pct/100.0
    Fb = auto_fee(buy_price,  st.session_state.buy_rates,  int(buy_duration))  # Q=1
    Fs = auto_fee(sell_price, st.session_state.sell_rates, int(sell_duration))
    profit_per_unit = sell_price*(1-rtax) - buy_price - (Fb) - (Fs)
    cost_basis = buy_price + Fb
    roi = (profit_per_unit / cost_basis) if cost_basis>0 else float("nan")
    return profit_per_unit, roi, Fb, Fs

def fmt(x, p=4):
    try: return f"{x:,.{p}f}"
    except: return str(x)

def median_timedelta_hours(series):
    if series.empty: return None
    s_sorted = series.sort_values()
    mid = len(s_sorted)//2
    if len(s_sorted)%2==1:
        return s_sorted.iloc[mid].total_seconds()/3600.0
    else:
        return (s_sorted.iloc[mid-1] + s_sorted.iloc[mid]).total_seconds()/7200.0

def liquidity_label(hours):
    if hours is None: return "—"
    if hours <= 24: return "⚡⚡⚡"
    if hours <= 72: return "⚡⚡"
    return "⚡"

# --------------------------------------------------------------------------------------
# Sidebar Config (colapsada)
# --------------------------------------------------------------------------------------
with st.sidebar.expander("⚙️ Config (calibração & taxa) — clique para expandir", expanded=False):
    st.write(f"**Config lida de:** {cfg_path if cfg_path else 'padrões internos'}")
    st.write(f"HOME: {HOME_CFG}")
    st.write(f"LOCAL: {LOCAL_CFG}")
    st.session_state.tax_pct = st.number_input("Sales tax r (%)", 0.0, 15.0, float(st.session_state.tax_pct), 0.25)

    st.markdown("**Calibrar SELL (por duração)**")
    cols = st.columns(4); changed=False
    for d, c in zip(DURATIONS, cols):
        up = c.number_input(f"S{d}d price", key=f"sell_up_{d}", value=float(sell_defaults[d]['S']), format="%.5f")
        qy = c.number_input(f"Qs{d}d qty", key=f"sell_q_{d}", value=int(sell_defaults[d]['Q']), step=1)
        fe = c.number_input(f"Fs{d}d fee", key=f"sell_fee_{d}", value=float(sell_defaults[d]['F']), format="%.5f")
        if c.button(f"Salvar {d}d", key=f"save_sell_{d}"):
            r_eff = (fe/(up*qy)) if up*qy>0 else 0.0
            if r_eff>0: st.session_state.sell_rates[d]=r_eff; changed=True

    st.markdown("**Calibrar BUY (por duração)**")
    cols = st.columns(4)
    for d, c in zip(DURATIONS, cols):
        up = c.number_input(f"B{d}d price", key=f"buy_up_{d}", value=float(buy_defaults[d]['B']), format="%.5f")
        qy = c.number_input(f"Qb{d}d qty", key=f"buy_q_{d}", value=int(buy_defaults[d]['Q']), step=1)
        fe = c.number_input(f"Fb{d}d fee", key=f"buy_fee_{d}", value=float(buy_defaults[d]['F']), format="%.5f")
        if c.button(f"Salvar {d}d", key=f"save_buy_{d}"):
            r_eff = (fe/(up*qy)) if up*qy>0 else 0.0
            if r_eff>0: st.session_state.buy_rates[d]=r_eff; changed=True

    if changed or (float(cfg.get("tax_pct",-1)) != float(st.session_state.tax_pct)):
        new_cfg = {
            "tax_pct": float(st.session_state.tax_pct),
            "sell_rates": {str(k): float(v) for k,v in st.session_state.sell_rates.items()},
            "buy_rates":  {str(k): float(v) for k,v in st.session_state.buy_rates.items()},
        }
        for p in save_to_all(new_cfg, CFG_CANDIDATES):
            st.success(f"Config salva em: {p}")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab_hist, tab_best, tab_import, tab_cad, tab_calc = st.tabs(["Histórico", "Oportunidades", "Importar preços", "Cadastro", "Calculadora"])

# --------------------------------------------------------------------------------------
# Histórico
# --------------------------------------------------------------------------------------
with tab_hist:
    st.markdown("## Histórico")
    hist_df, src_path = load_history()
    st.caption(f"Lendo de: `{(src_path or HISTORY_PATH).resolve()}` (salva em `{HISTORY_PATH.resolve()}`)")

    if not hist_df.empty:
        # Conversões de tempo
        for col in ["timestamp","buy_placed_ts","buy_filled_ts","sell_filled_ts"]:
            if col in hist_df.columns:
                hist_df[col] = pd.to_datetime(hist_df[col], utc=True, errors="coerce")

        rows = []
        for idx, r in hist_df.reset_index().iterrows():
            pp, roi, Fb, Fs = compute_metrics(r["buy_price"], r["sell_price"], r["buy_duration"], r["sell_duration"], st.session_state.tax_pct)
            rows.append({
                "row_id": int(r["index"]),
                "timestamp": r["timestamp"],
                "item": r["item"],
                "buy_price": r["buy_price"],
                "sell_price": r["sell_price"],
                "buy_duration": int(r["buy_duration"]),
                "sell_duration": int(r["sell_duration"]),
                "profit_per_unit": pp,
                "roi_pct": (roi*100.0) if pd.notna(roi) else None,
                "buy_placed_ts": r.get("buy_placed_ts", pd.NaT),
                "buy_filled_ts": r.get("buy_filled_ts", pd.NaT),
                "sell_filled_ts": r.get("sell_filled_ts", pd.NaT)
            })
        table = pd.DataFrame(rows).sort_values("timestamp", ascending=False)

        st.caption("Dica: segure Ctrl/Cmd para seleção múltipla")
        df_key = "hist_tbl"
        try:
            st.dataframe(
                table.set_index("row_id"),
                use_container_width=True,
                on_select="rerun",
                selection_mode="multi-row",
                key=df_key,
            )
            sel_row_ids = []
            state = st.session_state.get(df_key, {})
            sel = state.get("selection", {})
            rows_sel = sel.get("rows", sel.get("indices", []))
            for r in rows_sel or []:
                if isinstance(r, dict):
                    if "row" in r: sel_row_ids.append(int(r["row"]))
                    elif "index" in r: sel_row_ids.append(int(r["index"]))
                elif isinstance(r, int):
                    sel_row_ids.append(int(r))
        except Exception:
            st.dataframe(table.set_index("row_id"), use_container_width=True)
            sel_row_ids = []

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.download_button("Baixar histórico (JSON)", data=table.to_json(orient="records", indent=2, date_format="iso"), file_name="history_with_roi.json", mime="application/json")
        with c2:
            disabled = len(sel_row_ids) == 0
            if st.button(f"🗑️ Apagar selecionados ({len(sel_row_ids)})", disabled=disabled):
                if sel_row_ids:
                    new_df = hist_df.drop(index=sel_row_ids, errors="ignore")
                    save_history(new_df)
                    st.success(f"Removidas {len(sel_row_ids)} linha(s) do histórico.")
                    st.rerun()
        with c3:
            if "confirm_clear" not in st.session_state:
                st.session_state.confirm_clear = False
            if not st.session_state.confirm_clear:
                if st.button("⚠️ Limpar histórico (todos)"):
                    st.session_state.confirm_clear = True
                    st.experimental_rerun()
            else:
                st.warning("Tem certeza que deseja **apagar TODO o histórico**? Essa ação não pode ser desfeita.")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✅ Confirmar limpeza"):
                        HISTORY_PATH.unlink(missing_ok=True)
                        st.session_state.confirm_clear = False
                        st.success("Histórico limpo.")
                        st.rerun()
                with b2:
                    if st.button("Cancelar"):
                        st.session_state.confirm_clear = False
                        st.info("Cancelado. Nada foi apagado.")
        with c4:
            disabled = len(sel_row_ids) == 0
            if st.button("✔️ Marcar BUY filled (selecionados)", disabled=disabled):
                if sel_row_ids:
                    new_df = hist_df.copy()
                    ts_now = to_utc_iso()
                    for rid in sel_row_ids:
                        if rid in new_df.index:
                            if pd.isna(new_df.at[rid, "buy_placed_ts"]):
                                new_df.at[rid, "buy_placed_ts"] = new_df.at[rid, "timestamp"]
                            new_df.at[rid, "buy_filled_ts"] = ts_now
                    save_history(new_df)
                    st.success(f"BUY filled marcado em {len(sel_row_ids)} registro(s).")
                    st.rerun()
        with c5:
            disabled = len(sel_row_ids) == 0
            if st.button("✔️ Marcar SELL filled (selecionados)", disabled=disabled):
                if sel_row_ids:
                    new_df = hist_df.copy()
                    ts_now = to_utc_iso()
                    for rid in sel_row_ids:
                        if rid in new_df.index:
                            # se não tinha buy_placed, considere timestamp como placed
                            if pd.isna(new_df.at[rid, "buy_placed_ts"]):
                                new_df.at[rid, "buy_placed_ts"] = new_df.at[rid, "timestamp"]
                            # se não tinha buy_filled, permite ainda assim registrar sell_filled
                            new_df.at[rid, "sell_filled_ts"] = ts_now
                    save_history(new_df)
                    st.success(f"SELL filled marcado em {len(sel_row_ids)} registro(s).")
                    st.rerun()
    else:
        st.info("Seu histórico está vazio. Vá na aba **Importar preços** para adicionar itens.")

# --------------------------------------------------------------------------------------
# Oportunidades
# --------------------------------------------------------------------------------------
with tab_best:
    st.markdown("## Oportunidades")

    # Controles novos (sem allocator/banca)
    c0, c1, c2 = st.columns([1,1,2])
    slots_por_ordem = c0.selectbox("Slots/ordem", [2,1], index=0, help="2 = buy+sell; 1 = apenas sell (estoque).")
    tamanho_pref = c1.number_input("Tamanho preferido (cap por ordem)", min_value=0, value=0, step=1,
                                   help="0 = usar stack_max; se >0, limita a este máximo por ordem")
    min_roi = c2.slider("ROI mínimo", 0.0, 0.5, 0.15, 0.01)

    hist_df, _ = load_history()
    items_df = load_items()

    if hist_df.empty:
        st.info("Ainda não há histórico suficiente. Importe alguns itens primeiro.")
    else:
        # últimos preços por item
        tmp = hist_df.copy()
        tmp["ts"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
        tmp = tmp.sort_values("ts").groupby("item", as_index=False).tail(1)

        rows = []
        for _, r in tmp.iterrows():
            pp, roi, Fb, Fs = compute_metrics(r["buy_price"], r["sell_price"], r["buy_duration"], r["sell_duration"], st.session_state.tax_pct)
            rows.append({
                "item": r["item"],
                "timestamp": r["timestamp"],
                "buy_price": r["buy_price"],
                "sell_price": r["sell_price"],
                "buy_duration": int(r["buy_duration"]),
                "sell_duration": int(r["sell_duration"]),
                "profit_per_unit": pp,
                "roi": roi,
                "roi_pct": (roi*100.0) if pd.notna(roi) else None
            })
        best = pd.DataFrame(rows).sort_values("roi", ascending=False)

        # Merge com cadastro
        items_df = items_df.drop_duplicates(subset=["item"])
        enriched = best.merge(items_df, on="item", how="left")

        # Normaliza tags
        if "tags" not in enriched.columns:
            enriched["tags"] = [[] for _ in range(len(enriched))]
        else:
            enriched["tags"] = enriched["tags"].apply(ensure_list_tags)

        # lucro/peso & lucro/100peso
        enriched["lucro_por_peso"] = None
        mask = enriched["peso"].apply(lambda x: isinstance(x, (int, float))) & (enriched["peso"]>0)
        enriched.loc[mask, "lucro_por_peso"] = enriched.loc[mask, "profit_per_unit"] / enriched.loc[mask, "peso"]
        enriched["lucro_100_peso"] = enriched["lucro_por_peso"] * 100.0

        # qty_por_ordem (sem allocator): usa stack_max ou tamanho_pref se >0
        def qty_for_row(row):
            sm = row.get("stack_max", None)
            if pd.isna(sm) or sm is None or sm <= 0:
                sm = 1000  # fallback razoável
            if tamanho_pref and tamanho_pref > 0:
                sm = min(int(sm), int(tamanho_pref))
            return int(sm)

        enriched["qty_por_ordem"] = enriched.apply(qty_for_row, axis=1)

        # lucro por slot
        def lucro_slot(row):
            if pd.isna(row["profit_per_unit"]) or row["qty_por_ordem"]<=0 or slots_por_ordem<=0:
                return None
            return (row["profit_per_unit"] * row["qty_por_ordem"]) / slots_por_ordem
        enriched["lucro_por_slot"] = enriched.apply(lucro_slot, axis=1)

        # Liquidez (dos seus fills)
        full_hist, _ = load_history()
        for col in ["buy_placed_ts","buy_filled_ts","sell_filled_ts"]:
            if col in full_hist.columns:
                full_hist[col] = pd.to_datetime(full_hist[col], utc=True, errors="coerce")
        # Por item: mediana do tempo de round-trip quando tivermos buy_placed e sell_filled
        liq = []
        for it, g in full_hist.groupby("item"):
            g = g.copy()
            mask_rt = g["buy_placed_ts"].notna() & g["sell_filled_ts"].notna()
            rt = (g.loc[mask_rt, "sell_filled_ts"] - g.loc[mask_rt, "buy_placed_ts"]).dropna()
            med_h = median_timedelta_hours(rt) if not rt.empty else None
            liq.append({"item": it, "median_hours": med_h, "liquidez": liquidity_label(med_h)})
        liq_df = pd.DataFrame(liq)
        enriched = enriched.merge(liq_df, on="item", how="left")

        # Filtros e view
        filt = (enriched["roi"] >= min_roi) & enriched["roi"].notna()
        view = enriched.loc[filt].copy().sort_values(["lucro_por_slot","roi"], ascending=[False, False])

        # Exibição
        if "timestamp" in view.columns:
            view["timestamp"] = pd.to_datetime(view["timestamp"], utc=True, errors="coerce")

        if LIST_COL_AVAILABLE:
            colcfg = {
                "item": st.column_config.TextColumn("item"),
                "categoria": st.column_config.TextColumn("categoria"),
                "tags": st.column_config.ListColumn("tags"),
                "peso": st.column_config.NumberColumn("peso", format="%.3f"),
                "timestamp": st.column_config.DatetimeColumn("timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "buy_price": st.column_config.NumberColumn("buy", format="%.2f"),
                "sell_price": st.column_config.NumberColumn("sell", format="%.2f"),
                "profit_per_unit": st.column_config.NumberColumn("lucro/u", format="%.4f"),
                "roi_pct": st.column_config.ProgressColumn("ROI", format="%.2f%%", min_value=0, max_value=100),
                "lucro_por_peso": st.column_config.NumberColumn("lucro/peso", format="%.4f"),
                "lucro_100_peso": st.column_config.NumberColumn("lucro/100 peso", format="%.2f"),
                "qty_por_ordem": st.column_config.NumberColumn("qty/ordem"),
                "lucro_por_slot": st.column_config.NumberColumn("lucro/slot", format="%.2f"),
                "liquidez": st.column_config.TextColumn("⚡ liquidez")
            }
        else:
            view["tags"] = view["tags"].apply(stringify_tags)
            colcfg = {
                "item": st.column_config.TextColumn("item"),
                "categoria": st.column_config.TextColumn("categoria"),
                "tags": st.column_config.TextColumn("tags"),
                "peso": st.column_config.NumberColumn("peso", format="%.3f"),
                "timestamp": st.column_config.DatetimeColumn("timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "buy_price": st.column_config.NumberColumn("buy", format="%.2f"),
                "sell_price": st.column_config.NumberColumn("sell", format="%.2f"),
                "profit_per_unit": st.column_config.NumberColumn("lucro/u", format="%.4f"),
                "roi_pct": st.column_config.ProgressColumn("ROI", format="%.2f%%", min_value=0, max_value=100),
                "lucro_por_peso": st.column_config.NumberColumn("lucro/peso", format="%.4f"),
                "lucro_100_peso": st.column_config.NumberColumn("lucro/100 peso", format="%.2f"),
                "qty_por_ordem": st.column_config.NumberColumn("qty/ordem"),
                "lucro_por_slot": st.column_config.NumberColumn("lucro/slot", format="%.2f"),
                "liquidez": st.column_config.TextColumn("⚡ liquidez")
            }

        st.data_editor(
            view[["item","categoria","tags","peso","timestamp","buy_price","sell_price",
                  "profit_per_unit","roi_pct","lucro_por_peso","lucro_100_peso","qty_por_ordem",
                  "lucro_por_slot","liquidez"]],
            column_config=colcfg,
            hide_index=True, use_container_width=True, disabled=True,
            height=min(560, 90 + 38*max(1, len(view)))
        )

# --------------------------------------------------------------------------------------
# Importar preços
# --------------------------------------------------------------------------------------
with tab_import:
    st.markdown("## Importar preços")
    st.caption("Use `item, top_buy, low_sell, buy_duration, sell_duration, timestamp`. Assumo **buy = top_buy + 0.01** e **sell = low_sell - 0.01**.")

    PROMPT_TEXT = r"""
Você é uma IA que recebe **várias capturas de tela** (prints) do Trading Post do jogo *New World* com:
- **Current Buy Orders** e **Current Sell Orders**
- O **nome do item** visível no topo
- Às vezes a **duração** selecionada para a ordem (ex.: 1d, 3d, 7d, 14d)

Seu objetivo é produzir **um JSON único (array)** com um objeto por item, seguindo **exatamente** este formato:

[
  {"item":"NOME DO ITEM","top_buy":4.03,"low_sell":5.40,"buy_duration":3,"sell_duration":3,"timestamp":"2025-10-18T12:34:56Z"},
  {"item":"Outro Item","top_buy":0.62,"low_sell":0.71,"buy_duration":1,"sell_duration":3}
]

Regras:
1) Para cada print, identifique o **nome exato** do item e use no campo `"item"` (sem tier/raridade).
2) Em **Current Buy Orders**, pegue **o maior preço** (topo). Grave como `"top_buy"` (número com ponto).
3) Em **Current Sell Orders**, pegue **o menor preço** (topo). Grave como `"low_sell"`.
4) Arredonde para **2 casas decimais** (ex.: 5.399 → 5.40).
5) Se a **duração** (1d/3d/7d/14d) estiver clara no print, preencha `"buy_duration"` e `"sell_duration"` (em dias, inteiro). Se não aparecer, use **3**.
6) Inclua `"timestamp"` ISO **se** disponível; caso contrário pode **omitir**.
7) **Não** aplique +0.01/−0.01; apenas extraia **top_buy** e **low_sell**. O app fará os ajustes.
8) Saída final: **um único array JSON** com **todos os itens** dos prints, sem duplicatas (se repetir, mantêm-se **o último**).

Validação:
- Use **ponto** como separador decimal.
- Mínimo por objeto: `"item"`, `"top_buy"`, `"low_sell"`.
- Se houver dúvida, **ignore** o item.
Retorne **apenas** o JSON, sem comentários.
"""
    components.html(
        f"""
        <div>
          <button id="copyPrompt" style="padding:8px 12px; border:1px solid #ccc; border-radius:6px; background:#f3f4f6; cursor:pointer;">
            📋 Copiar prompt p/ IA
          </button>
          <textarea id="promptPayload" style="position:absolute; left:-10000px; top:-10000px;">{PROMPT_TEXT}</textarea>
        </div>
        <script>
          const btn = document.getElementById('copyPrompt');
          btn.addEventListener('click', async () => {{
            const txt = document.getElementById('promptPayload').value;
            try {{ await navigator.clipboard.writeText(txt); btn.innerText = '✅ Copiado!'; }}
            catch(e) {{
              const ta = document.getElementById('promptPayload');
              ta.focus(); ta.select(); document.execCommand('copy'); btn.innerText = '✅ Copiado!';
            }}
            setTimeout(()=>btn.innerText='📋 Copiar prompt p/ IA', 1500);
          }});
        </script>
        """,
        height=60
    )
    with st.expander("Ver prompt (opcional)"):
        st.code(PROMPT_TEXT, language="markdown")

    pasted = st.text_area("Colar JSON/CSV", height=140, placeholder='[\n  {"item":"Infused Weapon Fragment","top_buy":4.03,"low_sell":5.40,"buy_duration":3,"sell_duration":3}\n]')
    upload = st.file_uploader("...ou enviar arquivo", type=["json","csv"])

    raw = upload.read().decode("utf-8", errors="ignore") if upload is not None else pasted if pasted.strip() else None

    def parse_rows(txt: str) -> pd.DataFrame:
        if not txt: return pd.DataFrame()
        txt = txt.strip()
        try:
            return pd.DataFrame(json.loads(txt))
        except Exception:
            pass
        try:
            return pd.read_csv(StringIO(txt))
        except Exception:
            return pd.DataFrame()

    def add_to_history(preview_df: pd.DataFrame):
        cur, _ = load_history()
        new_rows = preview_df[["timestamp","item","buy_price","sell_price","buy_duration","sell_duration","buy_placed_ts"]].copy()
        # Normaliza timestamps para UTC ISO
        for col in ["timestamp","buy_placed_ts"]:
            new_rows[col] = pd.to_datetime(new_rows[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = pd.concat([cur, new_rows], ignore_index=True)
        save_history(cur)

    df_in = parse_rows(raw) if raw else pd.DataFrame()
    items_df = load_items().drop_duplicates(subset=["item"])

    if not df_in.empty:
        for c in ["buy_duration","sell_duration"]:
            if c not in df_in.columns: df_in[c] = 3
        # timestamp: se não vier, usar agora
        if "timestamp" not in df_in.columns:
            df_in["timestamp"] = now_iso()
        df_in["timestamp"] = df_in["timestamp"].fillna(now_iso())
        # buy_placed_ts = timestamp da coleta (pode editar depois)
        df_in["buy_placed_ts"] = df_in["timestamp"]

        # preços efetivos
        df_in["buy_price"]  = (df_in["top_buy"] + 0.01).round(2)
        df_in["sell_price"] = (df_in["low_sell"] - 0.01).round(2)

        # Prévia com ROI
        rows = []
        for _, r in df_in.iterrows():
            pp, roi, Fb, Fs = compute_metrics(r["buy_price"], r["sell_price"], r["buy_duration"], r["sell_duration"], st.session_state.tax_pct)
            rows.append({
                "timestamp": r["timestamp"],
                "buy_placed_ts": r["buy_placed_ts"],
                "item": r["item"],
                "buy_price": r["buy_price"], "sell_price": r["sell_price"],
                "buy_duration": int(r["buy_duration"]), "sell_duration": int(r["sell_duration"]),
                "profit_per_unit": pp, "roi": roi, "roi_pct": roi*100.0
            })
        preview = pd.DataFrame(rows).sort_values("roi", ascending=False)
        for col in ["timestamp","buy_placed_ts"]:
            preview[col] = pd.to_datetime(preview[col], utc=True, errors="coerce")

        # Itens não cadastrados
        missing_mask = ~preview["item"].isin(items_df["item"])
        missing_items = preview.loc[missing_mask, "item"].unique().tolist()
        hide_missing = st.toggle("Ocultar não cadastrados", value=False)
        if missing_items:
            st.warning(f"Há {len(missing_items)} item(ns) **não cadastrados**: " + ", ".join(missing_items))
        # Badge na prévia
        preview["status"] = preview["item"].apply(lambda x: "🚩 não cadastrado" if x in missing_items else "—")

        # Exibição
        st.subheader("Prévia (ordenada por ROI)")
        st.data_editor(
            preview.loc[~(hide_missing & missing_mask), ["status","timestamp","item","buy_price","sell_price","buy_duration","sell_duration","profit_per_unit","roi_pct"]],
            column_config={
                "status": st.column_config.TextColumn("status"),
                "timestamp": st.column_config.DatetimeColumn("timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "item": st.column_config.TextColumn("item"),
                "buy_price": st.column_config.NumberColumn("buy", format="%.2f"),
                "sell_price": st.column_config.NumberColumn("sell", format="%.2f"),
                "buy_duration": st.column_config.NumberColumn("buy d"),
                "sell_duration": st.column_config.NumberColumn("sell d"),
                "profit_per_unit": st.column_config.NumberColumn("lucro/u", format="%.4f"),
                "roi_pct": st.column_config.ProgressColumn("ROI", format="%.2f%%", min_value=0, max_value=100)
            },
            hide_index=True, use_container_width=True, disabled=True
        )

        # Cadastro rápido inline
        if missing_items:
            st.markdown("### Cadastro rápido (itens não cadastrados)")
            stub = pd.DataFrame({"item": missing_items, "categoria": "", "peso": 0.0, "stack_max": pd.Series([None]*len(missing_items), dtype="Int64"), "tags": [[] for _ in missing_items]})
            if not LIST_COL_AVAILABLE:
                stub["tags"] = [""]*len(stub)
            colcfg = {
                "item": st.column_config.TextColumn("item", help="Nome do item", required=True),
                "categoria": st.column_config.TextColumn("categoria", help="Ex.: Wood, Ore, Hide..."),
                "peso": st.column_config.NumberColumn("peso", format="%.3f", help="Peso por unidade", required=True),
                "stack_max": st.column_config.NumberColumn("stack_max", help="Opcional", min_value=1, step=1)
            }
            if LIST_COL_AVAILABLE:
                colcfg["tags"] = st.column_config.ListColumn("tags", help="Tags livres", default=[])
            else:
                colcfg["tags"] = st.column_config.TextColumn("tags", help="Separadas por vírgula")

            quick = st.data_editor(stub, num_rows="dynamic", column_config=colcfg, hide_index=True, use_container_width=True)
            if st.button("💾 Salvar cadastro rápido"):
                quick = quick.dropna(subset=["item"]).copy()
                if not LIST_COL_AVAILABLE:
                    quick["tags"] = quick["tags"].apply(ensure_list_tags)
                try:
                    quick["peso"] = pd.to_numeric(quick["peso"], errors="coerce")
                    if "stack_max" in quick.columns:
                        quick["stack_max"] = pd.to_numeric(quick["stack_max"], errors="coerce").astype("Int64")
                except Exception:
                    pass
                base = load_items()
                mask_new = ~base["item"].isin(quick["item"])
                merged = pd.concat([base[mask_new], quick], ignore_index=True)
                save_items(merged)
                st.success(f"{len(quick)} item(ns) cadastrados. Recarregando prévia…")
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Adicionar ao histórico (append)"):
                # adiciona também buy_placed_ts
                df_to_add = preview.copy()
                add_to_history(df_to_add)
                st.success(f"{len(preview)} registro(s) adicionados ao histórico.")
                st.rerun()
        with c2:
            st.download_button("Baixar processado (JSON)",
                               data=preview.to_json(orient="records", indent=2, date_format="iso"),
                               file_name="import_preview.json", mime="application/json")
    else:
        st.info("Cole ou envie um arquivo para ver a prévia e adicionar ao histórico.")

# --------------------------------------------------------------------------------------
# Cadastro (com Tags)
# --------------------------------------------------------------------------------------
with tab_cad:
    st.markdown("## Cadastro")
    st.caption(f"Arquivo: `{ITEMS_PATH.resolve()}`")

    IA_PROMPT = r"""
Você é uma IA que recebe **imagens** contendo **nomes de vários itens** do jogo *New World*.
Para cada item, você deve **consultar o NWDB** (https://nwdb.info) e produzir um **JSON** de cadastro com os campos abaixo.

### Saída (um único array JSON):
[
  {"item":"Dark Hide","categoria":"Raw Hide","peso":0.100,"stack_max":1000},
  {"item":"Iron Ore","categoria":"Ore","peso":0.100,"stack_max":1000}
]

### Regras de extração
1) **Nome do item (`item`)**: use o nome **exato** encontrado no NWDB.
2) **Peso (`peso`)**: no NWDB é exibido como **Weight** (ou equivalente). Grave como número decimal com **ponto** e **3 casas** (ex.: 0.100).
3) **Stack máximo (`stack_max`)**: no NWDB é exibido como **Max Stack** (ou equivalente). Grave como inteiro (ex.: 1000). Se não houver, omita o campo.
4) **Categoria (`categoria`)**:
   - A categoria NÃO está claramente na página do item. Então você deve localizar uma **página de listagem** onde esse item aparece (ex.: `https://nwdb.info/db/items/resources/raw-hide/page/1`).
   - Pegue a **última parte legível do caminho** (no exemplo: `raw-hide` → **"Raw Hide"**), substituindo **hífens por espaços** e usando **Title Case**.
   - Exemplo: se **Dark Hide** aparece em `/db/items/resources/raw-hide/page/1`, a categoria deve ser **"Raw Hide"**.
5) **Um único array** JSON com **todos os itens** detectados nas imagens. **Sem duplicatas**; se houver conflito, mantenha a versão com dados mais completos.
6) **Formatação**:
   - Use **ponto** como separador decimal em `peso`.
   - `stack_max` apenas se encontrado.
   - Não inclua campos extras.
7) Se algum item não puder ser validado com confiança no NWDB, **ignore**.

Retorne **apenas** o JSON (sem comentários).
"""
    components.html(
        f"""
        <div>
          <button id="copyCadPrompt" style="padding:8px 12px; border:1px solid #ccc; border-radius:6px; background:#f3f4f6; cursor:pointer;">
            📋 Copiar prompt p/ IA (Cadastro)
          </button>
          <textarea id="cadPromptPayload" style="position:absolute; left:-10000px; top:-10000px;">{IA_PROMPT}</textarea>
        </div>
        <script>
          const btn = document.getElementById('copyCadPrompt');
          btn.addEventListener('click', async () => {{
            const txt = document.getElementById('cadPromptPayload').value;
            try {{ await navigator.clipboard.writeText(txt); btn.innerText = '✅ Copiado!'; }}
            catch(e) {{
              const ta = document.getElementById('cadPromptPayload');
              ta.focus(); ta.select(); document.execCommand('copy'); btn.innerText = '✅ Copiado!';
            }}
            setTimeout(()=>btn.innerText='📋 Copiar prompt p/ IA (Cadastro)', 1500);
          }});
        </script>
        """,
        height=60
    )
    with st.expander("Ver prompt (opcional)"):
        st.code(IA_PROMPT, language="markdown")

    st.subheader("Importar cadastro (JSON/CSV)")
    st.caption("Campos: `item` (obrig.), `categoria` (obrig.), `peso` (obrig.), `stack_max` (opcional), `tags` (opcional).")
    pasted_items = st.text_area("Colar JSON/CSV do cadastro", height=140, placeholder='[\n  {"item":"Dark Hide","categoria":"Raw Hide","peso":0.100,"stack_max":1000,"tags":["leve","pvp"]}\n]')
    upload_items = st.file_uploader("...ou enviar arquivo", type=["json","csv"], key="items_upload")

    def parse_items_payload(txt: str) -> pd.DataFrame:
        if not txt: return pd.DataFrame()
        txt = txt.strip()
        try:
            return pd.DataFrame(json.loads(txt))
        except Exception:
            pass
        try:
            return pd.read_csv(StringIO(txt))
        except Exception:
            return pd.DataFrame()

    raw_items = upload_items.read().decode("utf-8", errors="ignore") if upload_items is not None else pasted_items if pasted_items.strip() else None
    df_items_in = parse_items_payload(raw_items) if raw_items else pd.DataFrame()

    items_df = load_items()

    if not df_items_in.empty:
        need = ["item","categoria","peso"]
        missing = [c for c in need if c not in df_items_in.columns]
        if missing:
            st.error(f"Campos obrigatórios ausentes: {', '.join(missing)}")
        else:
            try:
                df_items_in["peso"] = pd.to_numeric(df_items_in["peso"], errors="coerce")
            except Exception:
                pass
            if "stack_max" in df_items_in.columns:
                try:
                    df_items_in["stack_max"] = pd.to_numeric(df_items_in["stack_max"], errors="coerce").astype('Int64')
                except Exception:
                    pass
            if "tags" in df_items_in.columns:
                df_items_in["tags"] = df_items_in["tags"].apply(ensure_list_tags)
            else:
                df_items_in["tags"] = [[] for _ in range(len(df_items_in))]

            prev = df_items_in.copy()
            if LIST_COL_AVAILABLE:
                colcfg_prev = {
                    "item": st.column_config.TextColumn("item"),
                    "categoria": st.column_config.TextColumn("categoria"),
                    "peso": st.column_config.NumberColumn("peso", format="%.3f"),
                    **({"stack_max": st.column_config.NumberColumn("stack_max", min_value=1, step=1)} if "stack_max" in prev.columns else {}),
                    "tags": st.column_config.ListColumn("tags")
                }
            else:
                prev["tags"] = prev["tags"].apply(stringify_tags)
                colcfg_prev = {
                    "item": st.column_config.TextColumn("item"),
                    "categoria": st.column_config.TextColumn("categoria"),
                    "peso": st.column_config.NumberColumn("peso", format="%.3f"),
                    **({"stack_max": st.column_config.NumberColumn("stack_max", min_value=1, step=1)} if "stack_max" in prev.columns else {}),
                    "tags": st.column_config.TextColumn("tags")
                }

            st.subheader("Prévia do cadastro")
            st.data_editor(
                prev[["item","categoria","peso"] + (["stack_max"] if "stack_max" in prev.columns else []) + ["tags"]],
                column_config=colcfg_prev,
                hide_index=True, use_container_width=True, disabled=True
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Adicionar/atualizar cadastro"):
                    base = load_items()
                    mask = ~base["item"].isin(df_items_in["item"])
                    merged = pd.concat([base[mask], df_items_in], ignore_index=True)
                    save_items(merged)
                    st.success(f"{len(df_items_in)} item(ns) adicionados/atualizados.")
                    st.rerun()
            with c2:
                st.download_button(
                    "Baixar prévia (JSON)",
                    data=df_items_in.to_json(orient="records", indent=2, force_ascii=False),
                    file_name="items_preview.json",
                    mime="application/json"
                )

    st.markdown("### Editar cadastro existente")
    if "peso" in items_df.columns:
        try:
            items_df["peso"] = pd.to_numeric(items_df["peso"], errors="coerce")
        except Exception:
            pass
    if "stack_max" in items_df.columns:
        try:
            items_df["stack_max"] = pd.to_numeric(items_df["stack_max"], errors="coerce").astype('Int64')
        except Exception:
            pass
    if "tags" not in items_df.columns:
        items_df["tags"] = [[] for _ in range(len(items_df))]
    else:
        items_df["tags"] = items_df["tags"].apply(ensure_list_tags)

    if LIST_COL_AVAILABLE:
        colcfg_edit = {
            "item": st.column_config.TextColumn("item", help="Nome do item", required=True),
            "categoria": st.column_config.TextColumn("categoria", help="Ex.: Wood, Ore, Hide, Gem, Consumable..."),
            "peso": st.column_config.NumberColumn("peso", format="%.3f", help="Peso por unidade", required=True),
            "stack_max": st.column_config.NumberColumn("stack_max", help="Opcional", min_value=1, step=1),
            "tags": st.column_config.ListColumn("tags", help="Tags livres", default=[]),
        }
    else:
        items_df = items_df.copy()
        items_df["tags"] = items_df["tags"].apply(stringify_tags)
        colcfg_edit = {
            "item": st.column_config.TextColumn("item", required=True),
            "categoria": st.column_config.TextColumn("categoria"),
            "peso": st.column_config.NumberColumn("peso", format="%.3f", help="Peso por unidade"),
            "stack_max": st.column_config.NumberColumn("stack_max", help="Opcional", min_value=1, step=1),
            "tags": st.column_config.TextColumn("tags", help="Separadas por vírgula"),
        }

    edited = st.data_editor(
        items_df if not items_df.empty else pd.DataFrame(columns=["item","categoria","peso","stack_max","tags"]),
        num_rows="dynamic",
        column_config=colcfg_edit,
        hide_index=True, use_container_width=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💾 Salvar cadastro"):
            if "item" not in edited or "peso" not in edited:
                st.error("Campos obrigatórios ausentes (item, peso).")
            else:
                edited = edited.dropna(subset=["item"]).copy()
                edited["tags"] = edited["tags"].apply(ensure_list_tags)
                if (edited["peso"].fillna(0) <= 0).any():
                    st.error("Há linhas com peso ≤ 0. Corrija e salve novamente.")
                else:
                    save_items(edited)
                    st.success(f"{len(edited)} item(ns) salvos no cadastro.")
                    st.rerun()
    with c2:
        st.download_button(
            "⬇️ Exportar cadastro (JSON)",
            data=(load_items()).to_json(orient="records", indent=2, force_ascii=False),
            file_name="items.json", mime="application/json"
        )
    with c3:
        upload_items_file = st.file_uploader("Importar cadastro (JSON/CSV)", type=["json","csv"], key="items_upload_2")
        if upload_items_file is not None:
            raw = upload_items_file.read().decode("utf-8", errors="ignore")
            try:
                dfu = pd.DataFrame(json.loads(raw))
            except Exception:
                try:
                    dfu = pd.read_csv(StringIO(raw))
                except Exception:
                    dfu = pd.DataFrame()
            needed = ["item","categoria","peso"]
            if dfu.empty or any(c not in dfu.columns for c in needed):
                st.error("Arquivo inválido. É preciso ao menos: item, categoria, peso.")
            else:
                if "tags" in dfu.columns:
                    dfu["tags"] = dfu["tags"].apply(ensure_list_tags)
                else:
                    dfu["tags"] = [[] for _ in range(len(dfu))]
                base = load_items()
                mask = ~base["item"].isin(dfu["item"])
                merged = pd.concat([base[mask], dfu], ignore_index=True)
                save_items(merged)
                st.success(f"Cadastro importado: {len(dfu)} item(ns) atualizados/adicionados.")
                st.rerun()

# --------------------------------------------------------------------------------------
# Calculadora
# --------------------------------------------------------------------------------------
with tab_calc:
    st.markdown("## Calculadora")
    col1, col2 = st.columns(2)
    with col1:
        B = st.number_input("Preço de compra (B)", min_value=0.0, value=0.16, step=0.01)
        dur_buy = st.selectbox("Duração (compra)", DURATIONS, index=1)
    with col2:
        S = st.number_input("Preço de venda (S)", min_value=0.0, value=0.20, step=0.01)
        dur_sell = st.selectbox("Duração (venda)", DURATIONS, index=1)

    Q = st.number_input("Quantidade (Q)", min_value=1, value=1000, step=1, help="Quantidade comprada/vendida (é a mesma para compra e venda)")

    rtax = st.session_state.tax_pct/100.0
    Fb = auto_fee(B*Q,  st.session_state.buy_rates,  dur_buy)
    Fs = auto_fee(S*Q,  st.session_state.sell_rates, dur_sell)
    f_bu = Fb/Q if Q else 0.0
    f_se = Fs/Q if Q else 0.0
    profit_per_unit = S*(1-rtax) - B - f_bu - f_se
    net_profit_total = Q*profit_per_unit
    cost_basis = Q*(B+f_bu)
    roi = (net_profit_total/cost_basis) if cost_basis>0 else float("nan")
    S_be = (B + f_bu + f_se) / (1-rtax) if (1-rtax)!=0 else float("inf")
    B_be = S*(1-rtax) - (f_bu + f_se)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantidade", f"{Q:,}")
    c2.metric("Lucro / unidade", fmt(profit_per_unit))
    c3.metric("Lucro total", fmt(net_profit_total,2))
    c4.metric("ROI", f"{fmt(roi*100,2)}%")

    st.markdown("#### Fees (auto)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Fee compra total", fmt(Fb,4))
    d2.metric("Fee compra/u", fmt(f_bu))
    d3.metric("Fee venda total", fmt(Fs,4))
    d4.metric("Fee venda/u", fmt(f_se))

    st.markdown("#### Break-even")
    b1, b2 = st.columns(2)
    b1.metric("Preço de venda S*", fmt(S_be))
    b2.metric("Preço de compra B*", fmt(B_be))

    st.markdown("#### Meta de ROI")
    target_roi = st.slider("ROI alvo", 0.0, 1.0, 0.15, 0.01)
    S_for_target = (B + f_bu + f_se + target_roi*(B + f_bu)) / (1-rtax) if (1-rtax)!=0 else float("inf")
    B_max_for_target = (S*(1-rtax) - (f_bu + f_se) - target_roi*f_bu) / (1+target_roi) if (1+target_roi)!=0 else float("inf")
    t1, t2 = st.columns(2)
    t1.success(f"Vender a **{S_for_target:,.4f}** para {target_roi*100:.0f}% de ROI")
    t2.info(f"Máx. comprar a **{B_max_for_target:,.4f}** para {target_roi*100:.0f}% de ROI")
