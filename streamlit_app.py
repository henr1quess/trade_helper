# New World Helper — Oportunidades | Cadastro | Calculadora
# Run: streamlit run streamlit_app.py

import json
import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

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

HISTORY_PATH = SCRIPT_DIR / "history_local.json"
LEGACY_HISTORY_PATH = SCRIPT_DIR / "history.json"
LEGACY_WATCHLIST = SCRIPT_DIR / "watchlist.json"
HISTORY_READ_CANDIDATES = [HISTORY_PATH, LEGACY_HISTORY_PATH, LEGACY_WATCHLIST]

ITEMS_PATH = SCRIPT_DIR / "items.json"  # master data of items (cadastro)

# --------------------------------------------------------------------------------------
# Default sources for Devaloka snapshot sync
# --------------------------------------------------------------------------------------
DEFAULT_NWMP_SERVER = os.getenv("NWMP_SERVER", "devaloka")
DEFAULT_NWMP_BUY_SRC = os.getenv(
    "NWMP_BUY_SRC",
    "https://nwmpdata.gaming.tools/buy-orders2/devaloka.json",
)
DEFAULT_NWMP_SELL_SRC = os.getenv(
    "NWMP_SELL_SRC",
    "https://nwmpdata.gaming.tools/auctions2/devaloka.json",
)
DEFAULT_NWMP_RAW_ROOT = os.getenv("NWMP_RAW_ROOT", "raw")
DEFAULT_NWMP_BUY_CSV = os.getenv("NWMP_BUY_CSV_PATH", "data/history_devaloka_buy.csv")
DEFAULT_NWMP_SELL_CSV = os.getenv("NWMP_SELL_CSV_PATH", "data/history_devaloka_sell.csv")
DEFAULT_HISTORY_JSON = "history_local.json"


def _resolve_default_local_dir(env_var: str, default_candidate: str, fallback: Path) -> str:
    from pathlib import Path as _Path

    env_val = os.getenv(env_var)
    if env_val:
        return env_val

    candidate_path = _Path(default_candidate)
    if candidate_path.exists():
        return str(candidate_path)

    return str(fallback)


DEFAULT_LOCAL_BUYORDERS_DIR = _resolve_default_local_dir(
    "NWMP_LOCAL_BUYORDERS_DIR",
    r"C:\\Users\\Administrador\\AppData\\Local\\NWMPScanner2\\current\\buy-orders",
    SCRIPT_DIR / "example_snapshot" / "buy-orders",
)

DEFAULT_LOCAL_AUCTIONS_DIR = _resolve_default_local_dir(
    "NWMP_LOCAL_AUCTIONS_DIR",
    r"C:\\Users\\Administrador\\AppData\\Local\\NWMPScanner2\\current\\auctions",
    SCRIPT_DIR / "example_snapshot" / "auctions",
)

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
    # Histórico agora guarda apenas os preços de mercado (sem fills/durações)
    base_cols = ["timestamp", "item", "buy_market", "sell_market"]
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


def merge_patch_by_item(base: pd.DataFrame, patch: pd.DataFrame) -> pd.DataFrame:
    """
    Atualiza 'base' com os valores NÃO nulos de 'patch' (chave = item).
    Apenas colunas presentes em 'patch' são consideradas.
    """
    if patch.empty:
        return base
    base = base.copy()
    patch = patch.copy()
    patch_cols = [c for c in patch.columns if c != "item"]
    if not patch_cols:
        return base

    # Índices por 'item'
    b = base.set_index("item", drop=False)
    p = patch.set_index("item", drop=False)

    # Garante colunas ausentes
    for c in patch_cols:
        if c not in b.columns:
            b[c] = pd.NA

    # Aplica patch (apenas valores não nulos)
    for it, row in p.iterrows():
        if it in b.index:
            for c in patch_cols:
                val = row.get(c, pd.NA)
                if pd.notna(val) and val != "":
                    b.at[it, c] = val
    return b.reset_index(drop=True)


def _to_tier_int(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s:
        return pd.NA
    if s.lower().startswith("t") and s[1:].isdigit():
        return int(s[1:])
    try:
        return int(float(s))
    except Exception:
        return pd.NA

def load_items():
    df = load_json_records(ITEMS_PATH, ["item","categoria","peso","stack_max","tags","tier"])
    if "peso" in df.columns:
        df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    if "stack_max" in df.columns:
        df["stack_max"] = pd.to_numeric(df["stack_max"], errors="coerce").astype("Int64")
    if "tags" not in df.columns:
        df["tags"] = [[] for _ in range(len(df))]
    else:
        df["tags"] = df["tags"].apply(ensure_list_tags)
    if "tier" in df.columns:
        df["tier"] = df["tier"].apply(_to_tier_int).astype("Int64")
    else:
        df["tier"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df

def save_items(df: pd.DataFrame):
    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["item","categoria","peso","stack_max","tags","tier"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df["tags"] = df["tags"].apply(ensure_list_tags)
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    if "stack_max" in df.columns:
        df["stack_max"] = pd.to_numeric(df["stack_max"], errors="coerce").astype("Int64")
    if "tier" in df.columns:
        df["tier"] = df["tier"].apply(_to_tier_int).astype("Int64")

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
tab_best, tab_cad, tab_calc, tab_coletar = st.tabs([
    "Oportunidades",
    "Cadastro",
    "Calculadora",
    "Coletar snapshot",
])

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

    if not items_df.empty:
        if "peso" in items_df.columns:
            items_df["peso"] = pd.to_numeric(items_df["peso"], errors="coerce")
        if "stack_max" in items_df.columns:
            items_df["stack_max"] = pd.to_numeric(items_df["stack_max"], errors="coerce").astype("Int64")

    if hist_df.empty:
        st.info("Ainda não há histórico suficiente. Importe alguns itens primeiro.")
    else:
        # últimos preços por item (de mercado)
        tmp = hist_df.copy()
        tmp["ts"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
        tmp = tmp.sort_values("ts").groupby("item", as_index=False).tail(1)

        # controles para duração assumida dos flips
        c_d1, c_d2 = st.columns(2)
        assumed_buy = c_d1.selectbox("Duração (compra) p/ flip", [1, 3, 7, 14], index=1)
        assumed_sell = c_d2.selectbox("Duração (venda) p/ flip", [1, 3, 7, 14], index=1)

        rows = []
        for _, r in tmp.iterrows():
            buy_market_val = r.get("buy_market")
            if (buy_market_val is None or pd.isna(buy_market_val)) and "buy_price" in tmp.columns:
                raw = r.get("buy_price")
                if pd.notna(raw):
                    buy_market_val = float(raw) - 0.01
            sell_market_val = r.get("sell_market")
            if (sell_market_val is None or pd.isna(sell_market_val)) and "sell_price" in tmp.columns:
                raw = r.get("sell_price")
                if pd.notna(raw):
                    sell_market_val = float(raw) + 0.01

            flip_buy_val = None
            if buy_market_val is not None and pd.notna(buy_market_val):
                flip_buy_val = max(float(buy_market_val) + 0.01, 0.0)
            flip_sell_val = None
            if sell_market_val is not None and pd.notna(sell_market_val):
                flip_sell_val = max(float(sell_market_val) - 0.01, 0.0)

            if flip_buy_val is not None and flip_sell_val is not None:
                pp, roi, Fb, Fs = compute_metrics(flip_buy_val, flip_sell_val, assumed_buy, assumed_sell, st.session_state.tax_pct)
                rows.append({
                    "item": r["item"],
                    "timestamp": r["timestamp"],
                    "flip_buy": round(flip_buy_val, 2),
                    "flip_sell": round(flip_sell_val, 2),
                    "profit_per_unit": pp,
                    "roi": roi,
                    "roi_pct": (roi * 100.0) if pd.notna(roi) else None,
                })
        best = pd.DataFrame(rows)
        if best.empty:
            best = pd.DataFrame(
                columns=[
                    "item",
                    "timestamp",
                    "flip_buy",
                    "flip_sell",
                    "profit_per_unit",
                    "roi",
                    "roi_pct",
                ]
            )
        else:
            best = best.sort_values("roi", ascending=False)

        # Merge com cadastro
        items_df = items_df.drop_duplicates(subset=["item"])
        enriched = best.merge(items_df, on="item", how="left")

        # Normaliza campos usados para filtragem
        if "tags" not in enriched.columns:
            enriched["tags"] = [[] for _ in range(len(enriched))]
        else:
            enriched["tags"] = enriched["tags"].apply(ensure_list_tags)
        if "tier" not in enriched.columns:
            enriched["tier"] = None
        if "categoria" not in enriched.columns:
            enriched["categoria"] = ""
        else:
            enriched["categoria"] = enriched["categoria"].fillna("")
        if "peso" in enriched.columns:
            enriched["peso"] = pd.to_numeric(enriched["peso"], errors="coerce")
        else:
            enriched["peso"] = (
                pd.Series([pd.NA] * len(enriched), index=enriched.index, dtype="Float64")
                if len(enriched)
                else pd.Series(dtype="Float64")
            )

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

        # Liquidez (dos seus fills) — se disponível no histórico legado
        enriched["liquidez"] = "—"
        full_hist, _ = load_history()
        liq_cols = {"buy_placed_ts", "sell_filled_ts"}
        if liq_cols.issubset(set(full_hist.columns)):
            for col in liq_cols.union({"buy_filled_ts"}):
                if col in full_hist.columns:
                    full_hist[col] = pd.to_datetime(full_hist[col], utc=True, errors="coerce")
            liq = []
            for it, g in full_hist.groupby("item"):
                g = g.copy()
                if {"buy_placed_ts", "sell_filled_ts"}.issubset(g.columns):
                    mask_rt = g["buy_placed_ts"].notna() & g["sell_filled_ts"].notna()
                    rt = (g.loc[mask_rt, "sell_filled_ts"] - g.loc[mask_rt, "buy_placed_ts"]).dropna()
                    med_h = median_timedelta_hours(rt) if not rt.empty else None
                else:
                    med_h = None
                liq.append({"item": it, "median_hours": med_h, "liquidez": liquidity_label(med_h)})
            liq_df = pd.DataFrame(liq)
            enriched = enriched.drop(columns=["liquidez"], errors="ignore").merge(liq_df, on="item", how="left")
            if "liquidez" not in enriched.columns:
                enriched["liquidez"] = "—"
            else:
                enriched["liquidez"] = enriched["liquidez"].fillna("—")

        # Controles de filtragem
        st.markdown("### Filtros")
        col_itens, col_cats, col_tags = st.columns(3)
        with col_itens:
            item_options = sorted(enriched["item"].dropna().unique())
            selected_items = st.multiselect("Itens", item_options, placeholder="Selecionar itens")
        with col_cats:
            categoria_options = sorted({c for c in enriched["categoria"].dropna().unique() if str(c).strip()})
            selected_cats = st.multiselect("Categorias", categoria_options, placeholder="Selecionar categorias")
        with col_tags:
            tag_options = sorted({t for tags in enriched["tags"] for t in tags})
            selected_tags = st.multiselect("Tags", tag_options, placeholder="Selecionar tags")

        tier_range = None
        peso_range = None
        col_tier, col_peso = st.columns(2)
        tier_values = [int(t) for t in enriched["tier"].dropna().unique() if str(t).strip()]
        if tier_values:
            tier_min, tier_max = int(min(tier_values)), int(max(tier_values))
            with col_tier:
                if tier_min == tier_max:
                    tier_range = (tier_min, tier_max)
                    st.caption(f"Tier disponível: T{tier_min}")
                else:
                    tier_range = st.slider(
                        "Intervalo de tier",
                        min_value=tier_min,
                        max_value=tier_max,
                        value=(tier_min, tier_max),
                        step=1,
                    )
        peso_valid = enriched["peso"].dropna()
        if not peso_valid.empty:
            peso_min, peso_max = float(peso_valid.min()), float(peso_valid.max())
            with col_peso:
                if peso_min == peso_max:
                    peso_range = (peso_min, peso_max)
                    st.caption(f"Peso disponível: {peso_min:.3f}")
                else:
                    step_val = max(0.001, round((peso_max - peso_min) / 20, 3))
                    peso_range = st.slider(
                        "Intervalo de peso",
                        min_value=float(peso_min),
                        max_value=float(peso_max),
                        value=(float(peso_min), float(peso_max)),
                        step=float(step_val),
                    )

        # Filtros e view
        filt = enriched["roi"].notna() & (enriched["roi"] >= min_roi)
        if selected_items:
            filt &= enriched["item"].isin(selected_items)
        if selected_cats:
            filt &= enriched["categoria"].isin(selected_cats)
        if selected_tags:
            filt &= enriched["tags"].apply(lambda lst: any(tag in lst for tag in selected_tags))
        if tier_range is not None:
            tier_mask = enriched["tier"].isna()
            tier_mask |= enriched["tier"].between(tier_range[0], tier_range[1])
            filt &= tier_mask
        if peso_range is not None:
            peso_series = enriched["peso"].astype(float, errors="ignore")
            peso_mask = peso_series.isna()
            peso_mask |= (peso_series >= peso_range[0]) & (peso_series <= peso_range[1])
            filt &= peso_mask

        view = enriched.loc[filt].copy().sort_values(["lucro_por_slot","roi"], ascending=[False, False])

        # Exibição
        if "timestamp" in view.columns:
            view["timestamp"] = pd.to_datetime(view["timestamp"], utc=True, errors="coerce")

        if "tier" in view.columns:
            def _tier_to_str(val):
                if pd.isna(val):
                    return ""
                if isinstance(val, str):
                    return val
                try:
                    f = float(val)
                    if f.is_integer():
                        return str(int(f))
                    return str(f)
                except Exception:
                    return str(val)

            view["tier"] = view["tier"].apply(_tier_to_str)

        if LIST_COL_AVAILABLE:
            colcfg = {
                "item": st.column_config.TextColumn("item"),
                "categoria": st.column_config.TextColumn("categoria"),
                "tier": st.column_config.TextColumn("tier"),
                "tags": st.column_config.ListColumn("tags"),
                "peso": st.column_config.NumberColumn("peso", format="%.3f"),
                "timestamp": st.column_config.DatetimeColumn("timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "flip_buy": st.column_config.NumberColumn("flip buy (+0.01)", format="%.2f"),
                "flip_sell": st.column_config.NumberColumn("flip sell (−0.01)", format="%.2f"),
                "profit_per_unit": st.column_config.NumberColumn("lucro/u", format="%.4f"),
                "roi_pct": st.column_config.ProgressColumn("ROI", format="%.2f%%", min_value=0, max_value=100),
                "lucro_por_peso": st.column_config.NumberColumn("lucro/peso", format="%.4f"),
                "lucro_100_peso": st.column_config.NumberColumn("lucro/100 peso", format="%.2f"),
                "qty_por_ordem": st.column_config.NumberColumn("qty/ordem"),
                "lucro_por_slot": st.column_config.NumberColumn("lucro/slot", format="%.2f"),
                "liquidez": st.column_config.TextColumn("⚡ liquidez"),
            }
        else:
            view["tags"] = view["tags"].apply(stringify_tags)
            colcfg = {
                "item": st.column_config.TextColumn("item"),
                "categoria": st.column_config.TextColumn("categoria"),
                "tier": st.column_config.TextColumn("tier"),
                "tags": st.column_config.TextColumn("tags"),
                "peso": st.column_config.NumberColumn("peso", format="%.3f"),
                "timestamp": st.column_config.DatetimeColumn("timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "flip_buy": st.column_config.NumberColumn("flip buy (+0.01)", format="%.2f"),
                "flip_sell": st.column_config.NumberColumn("flip sell (−0.01)", format="%.2f"),
                "profit_per_unit": st.column_config.NumberColumn("lucro/u", format="%.4f"),
                "roi_pct": st.column_config.ProgressColumn("ROI", format="%.2f%%", min_value=0, max_value=100),
                "lucro_por_peso": st.column_config.NumberColumn("lucro/peso", format="%.4f"),
                "lucro_100_peso": st.column_config.NumberColumn("lucro/100 peso", format="%.2f"),
                "qty_por_ordem": st.column_config.NumberColumn("qty/ordem"),
                "lucro_por_slot": st.column_config.NumberColumn("lucro/slot", format="%.2f"),
                "liquidez": st.column_config.TextColumn("⚡ liquidez"),
            }

        st.data_editor(
            view[
                [
                    "item",
                    "categoria",
                    "tier",
                    "tags",
                    "peso",
                    "timestamp",
                    "flip_buy",
                    "flip_sell",
                    "profit_per_unit",
                    "roi_pct",
                    "lucro_por_peso",
                    "lucro_100_peso",
                    "qty_por_ordem",
                    "lucro_por_slot",
                    "liquidez",
                ]
            ],
            column_config=colcfg,
            hide_index=True,
            use_container_width=True,
            disabled=True,
            height=min(560, 90 + 38 * max(1, len(view))),
        )

# --------------------------------------------------------------------------------------
# Cadastro (com Tags)
# --------------------------------------------------------------------------------------
with tab_cad:
    st.markdown("## Cadastro")
    st.caption(f"Arquivo: `{ITEMS_PATH.resolve()}`")

    IA_PROMPT = r"""
Você é uma IA que recebe **imagens** ou **textos** contendo **nomes de vários itens** do jogo *New World*.
Para cada item, você deve **consultar o NWDB** (https://nwdb.info) e produzir um **JSON** de cadastro com os campos abaixo.
As entradas podem ser descrições textuais transcritas; use exatamente os dados fornecidos quando disponíveis.

### Saída (um único array JSON):
[
  {"item":"Dark Hide","categoria":"Raw Hide","peso":0.100,"stack_max":1000,"tier":5},
  {"item":"Iron Ore","categoria":"Ore","peso":0.100,"stack_max":1000}
]

### Regras de extração
1) **Nome do item (`item`)**: use o nome **exato** encontrado no NWDB.
2) **Peso (`peso`)**: no NWDB é exibido como **Weight** (ou equivalente). Grave como número decimal com **ponto** e **3 casas** (ex.: 0.100).
3) **Stack máximo (`stack_max`)**: no NWDB é exibido como **Max Stack** (ou equivalente). Grave como inteiro (ex.: 1000). Se não houver, omita o campo.
4) **Tier (`tier`)**: procure o **tier** do item no NWDB (ex.: T3) e grave como número inteiro **quando disponível**.
5) **Categoria (`categoria`)**:
   - A categoria NÃO está claramente na página do item. Então você deve localizar uma **página de listagem** onde esse item aparece (ex.: `https://nwdb.info/db/items/resources/raw-hide/page/1`).
   - Pegue a **última parte legível do caminho** (no exemplo: `raw-hide` → **"Raw Hide"**), substituindo **hífens por espaços** e usando **Title Case**.
   - Exemplo: se **Dark Hide** aparece em `/db/items/resources/raw-hide/page/1`, a categoria deve ser **"Raw Hide"**.
6) **Um único array** JSON com **todos os itens** detectados nas imagens ou textos. **Sem duplicatas**; se houver conflito, mantenha a versão com dados mais completos.
7) **Formatação**:
   - Use **ponto** como separador decimal em `peso`.
   - `stack_max` apenas se encontrado.
   - `tier` apenas se encontrado.
   - Não inclua campos extras.
8) Se algum item não puder ser validado com confiança no NWDB, **ignore**.

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
    st.caption(
        "Campos: `item` (obrig.), `categoria` (obrig. p/ novos), `peso` (obrig. p/ novos), "
        "`stack_max` (opcional), `tags` (opcional), `tier` (opcional)."
    )

    # ✅ Novo: modo atualização (patch)
    patch_mode = st.toggle(
        "Modo atualização (patch): permitir colar apenas campos a atualizar (ex.: `item` + `categoria`) para itens já existentes.",
        value=True,
        help="Quando ativo, linhas com itens já existentes são atualizadas apenas nos campos presentes. "
             "Para itens novos, ainda é obrigatório informar `categoria` e `peso`."
    )

    pasted_items = st.text_area(
        "Colar JSON/CSV do cadastro",
        height=140,
        placeholder='[\n  {"item":"Green Wood"}\n]'
    )
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
        # Normalização leve dos tipos presentes
        if "peso" in df_items_in.columns:
            df_items_in["peso"] = pd.to_numeric(df_items_in["peso"], errors="coerce")
        if "stack_max" in df_items_in.columns:
            df_items_in["stack_max"] = pd.to_numeric(df_items_in["stack_max"], errors="coerce").astype("Int64")
        if "tier" in df_items_in.columns:
            df_items_in["tier"] = df_items_in["tier"].apply(_to_tier_int).astype("Int64")
        if "tags" in df_items_in.columns:
            df_items_in["tags"] = df_items_in["tags"].apply(ensure_list_tags)
        # Validação mínima: precisa ter 'item'
        if "item" not in df_items_in.columns:
            st.error("Campo obrigatório ausente: item")
        else:
            df_items_in = df_items_in.dropna(subset=["item"]).copy()

            # Divide entre itens já existentes vs novos
            exists_mask = df_items_in["item"].isin(items_df["item"])
            df_patch = df_items_in.loc[exists_mask].copy()        # atualizações (patch)
            df_new   = df_items_in.loc[~exists_mask].copy()       # novos (precisam categoria + peso)

            # ❗ Para novos: exigir categoria e peso
            missing_fields = []
            if not df_new.empty:
                for col in ["categoria", "peso"]:
                    if col not in df_new.columns or df_new[col].isna().any():
                        missing_fields.append(col)
                # ‘peso’ precisa ser > 0
                if "peso" in df_new.columns and (df_new["peso"].fillna(0) <= 0).any():
                    missing_fields.append("peso>0")
            if not df_new.empty and missing_fields and not patch_mode:
                st.error("Para itens **novos** (não existentes no cadastro), é obrigatório informar: categoria e peso (>0).")
            else:
                # Prévia do que será aplicado
                st.subheader("Prévia do cadastro")
                prev = df_items_in.copy()
                # colunas ordenadas amigáveis
                wanted = ["item","categoria","peso","stack_max","tier","tags"]
                show_cols = [c for c in wanted if c in prev.columns] + [c for c in prev.columns if c not in wanted]
                # configurações de exibição
                if LIST_COL_AVAILABLE:
                    colcfg_prev = {
                        "item": st.column_config.TextColumn("item"),
                        **({"categoria": st.column_config.TextColumn("categoria")} if "categoria" in prev.columns else {}),
                        **({"peso": st.column_config.NumberColumn("peso", format="%.3f")} if "peso" in prev.columns else {}),
                        **({"stack_max": st.column_config.NumberColumn("stack_max", min_value=1, step=1)} if "stack_max" in prev.columns else {}),
                        **({"tier": st.column_config.NumberColumn("tier", min_value=1, step=1)} if "tier" in prev.columns else {}),
                        **({"tags": st.column_config.ListColumn("tags")} if "tags" in prev.columns else {}),
                    }
                else:
                    if "tags" in prev.columns:
                        prev["tags"] = prev["tags"].apply(stringify_tags)
                    colcfg_prev = {c: st.column_config.TextColumn(c) for c in show_cols}

                st.data_editor(
                    prev[show_cols],
                    column_config=colcfg_prev,
                    hide_index=True, use_container_width=True, disabled=True
                )

                # Resumo
                s1, s2 = st.columns(2)
                s1.info(f"Itens para PATCH (existentes): **{len(df_patch)}**")
                s2.info(f"Itens novos (inserir): **{len(df_new)}**")

                # Botão de aplicar
                if st.button("Adicionar/atualizar cadastro"):
                    base = load_items()

                    # 1) Aplica PATCH nos existentes (apenas campos presentes e não nulos)
                    if not df_patch.empty:
                        base = merge_patch_by_item(base, df_patch)

                    # 2) Insere novos (se houver)
                    inserted = 0
                    if not df_new.empty:
                        # Se modo patch estiver ativo, vamos permitir salvar novos **parciais**?
                        # Decisão: não. Vamos exigir categoria & peso. Mostramos um editor para completar.
                        need_cols = []
                        if "categoria" not in df_new.columns:
                            need_cols.append("categoria")
                        if "peso" not in df_new.columns:
                            need_cols.append("peso")
                        df_ins = df_new.copy()
                        # Caso falte algo, abre editor para completar antes de salvar
                        if need_cols or (("peso" in df_ins.columns) and (df_ins["peso"].fillna(0) <= 0).any()):
                            st.warning("Há itens **novos** sem `categoria` e/ou `peso`. Complete abaixo e clique em **Salvar novos**.")
                            # Prepara editor
                            for c in ["categoria","peso","stack_max","tags","tier"]:
                                if c not in df_ins.columns:
                                    if c == "tags":
                                        df_ins[c] = [[] for _ in range(len(df_ins))]
                                    elif c == "tier":
                                        df_ins[c] = pd.Series([pd.NA]*len(df_ins), dtype="Int64")
                                    elif c == "stack_max":
                                        df_ins[c] = pd.Series([pd.NA]*len(df_ins), dtype="Int64")
                                    elif c == "peso":
                                        df_ins[c] = 0.0
                                    else:
                                        df_ins[c] = ""
                            if LIST_COL_AVAILABLE:
                                colcfg_new = {
                                    "item": st.column_config.TextColumn("item", required=True),
                                    "categoria": st.column_config.TextColumn("categoria", required=True),
                                    "peso": st.column_config.NumberColumn("peso", format="%.3f", required=True),
                                    "stack_max": st.column_config.NumberColumn("stack_max", min_value=1, step=1),
                                    "tags": st.column_config.ListColumn("tags"),
                                    "tier": st.column_config.NumberColumn("tier", min_value=1, step=1),
                                }
                            else:
                                df_ins["tags"] = df_ins["tags"].apply(stringify_tags)
                                colcfg_new = None

                            df_ins_edit = st.data_editor(
                                df_ins[["item","categoria","peso","stack_max","tags","tier"]],
                                column_config=colcfg_new, hide_index=True, use_container_width=True, key="new_items_editor"
                            )
                            if st.button("Salvar novos"):
                                df_ins_edit["peso"] = pd.to_numeric(df_ins_edit["peso"], errors="coerce")
                                if (df_ins_edit["peso"].fillna(0) <= 0).any():
                                    st.error("Há linhas com `peso` ≤ 0.")
                                else:
                                    if "stack_max" in df_ins_edit.columns:
                                        df_ins_edit["stack_max"] = pd.to_numeric(df_ins_edit["stack_max"], errors="coerce").astype("Int64")
                                    if "tier" in df_ins_edit.columns:
                                        df_ins_edit["tier"] = df_ins_edit["tier"].apply(_to_tier_int).astype("Int64")
                                    df_ins_edit["tags"] = df_ins_edit["tags"].apply(ensure_list_tags)
                                    mask = ~base["item"].isin(df_ins_edit["item"])
                                    base = pd.concat([base[mask], df_ins_edit], ignore_index=True)
                                    inserted = len(df_ins_edit)
                                    save_items(base)
                                    st.success(f"Cadastro atualizado. Inseridos: {inserted}. Patched: {len(df_patch)}.")
                                    st.rerun()
                            st.stop()
                        else:
                            # Tudo ok para inserir direto
                            mask = ~base["item"].isin(df_ins["item"])
                            base = pd.concat([base[mask], df_ins], ignore_index=True)
                            inserted = len(df_ins)

                    # 3) Salva resultado
                    save_items(base)
                    st.success(f"Cadastro atualizado. Inseridos: {inserted}. Patched: {len(df_patch)}.")
                    st.rerun()

    st.markdown("### Editar cadastro existente")
    # --- normalização de tipos do cadastro ---
    if "peso" in items_df.columns:
        items_df["peso"] = pd.to_numeric(items_df["peso"], errors="coerce")

    if "stack_max" in items_df.columns:
        items_df["stack_max"] = pd.to_numeric(items_df["stack_max"], errors="coerce").astype("Int64")

    if "tags" not in items_df.columns:
        items_df["tags"] = [[] for _ in range(len(items_df))]
    else:
        items_df["tags"] = items_df["tags"].apply(ensure_list_tags)

    if "tier" in items_df.columns:
        items_df["tier"] = items_df["tier"].apply(_to_tier_int).astype("Int64")
    else:
        items_df["tier"] = pd.Series([pd.NA] * len(items_df), dtype="Int64")

    # --- fim normalização ---

    items_edit_df = items_df.copy()

    if LIST_COL_AVAILABLE:
        colcfg_edit = {
            "item": st.column_config.TextColumn("item", help="Nome do item", required=True),
            "categoria": st.column_config.TextColumn("categoria", help="Ex.: Wood, Ore, Hide, Gem, Consumable..."),
            "peso": st.column_config.NumberColumn("peso", format="%.3f", help="Peso por unidade", required=True),
            "stack_max": st.column_config.NumberColumn("stack_max", help="Opcional", min_value=1, step=1),
            "tags": st.column_config.ListColumn("tags", help="Tags livres", default=[]),
            "tier": st.column_config.NumberColumn("tier", help="Opcional (ex.: 1–5)", min_value=1, step=1),
        }
    else:
        items_edit_df["tags"] = items_edit_df["tags"].apply(stringify_tags)
        colcfg_edit = {
            "item": st.column_config.TextColumn("item", required=True),
            "categoria": st.column_config.TextColumn("categoria"),
            "peso": st.column_config.NumberColumn("peso", format="%.3f", help="Peso por unidade"),
            "stack_max": st.column_config.NumberColumn("stack_max", help="Opcional", min_value=1, step=1),
            "tags": st.column_config.TextColumn("tags", help="Separadas por vírgula"),
            "tier": st.column_config.NumberColumn("tier", help="Opcional (ex.: 1–5)", min_value=1, step=1),
        }

    edited = st.data_editor(
        items_edit_df
        if not items_edit_df.empty
        else pd.DataFrame(columns=["item","categoria","peso","stack_max","tags","tier"]),
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
                    if "tier" in edited.columns:
                        edited["tier"] = edited["tier"].apply(_to_tier_int).astype("Int64")
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

            if dfu.empty or "item" not in dfu.columns:
                st.error("Arquivo inválido. É preciso pelo menos a coluna `item`.")
            else:
                dfu = dfu.dropna(subset=["item"]).copy()

                if "peso" in dfu.columns:
                    dfu["peso"] = pd.to_numeric(dfu["peso"], errors="coerce")
                if "stack_max" in dfu.columns:
                    dfu["stack_max"] = pd.to_numeric(dfu["stack_max"], errors="coerce").astype("Int64")
                if "tier" in dfu.columns:
                    dfu["tier"] = dfu["tier"].apply(_to_tier_int).astype("Int64")
                if "tags" in dfu.columns:
                    dfu["tags"] = dfu["tags"].apply(ensure_list_tags)

                base = load_items()
                exists_mask = dfu["item"].isin(base["item"])
                df_patch = dfu.loc[exists_mask].copy()
                df_new = dfu.loc[~exists_mask].copy()

                if not df_new.empty:
                    valid_new_mask = pd.Series(True, index=df_new.index)
                    if "categoria" in df_new.columns:
                        valid_new_mask &= df_new["categoria"].notna() & (
                            df_new["categoria"].astype(str).str.strip() != ""
                        )
                    else:
                        valid_new_mask &= False

                    if "peso" in df_new.columns:
                        valid_new_mask &= df_new["peso"].notna() & (df_new["peso"] > 0)
                    else:
                        valid_new_mask &= False

                    invalid_count = (~valid_new_mask).sum()
                    if invalid_count:
                        st.warning(
                            "Itens novos precisam informar `categoria` e `peso` (>0). "
                            "Linhas ignoradas: "
                            f"{int(invalid_count)}"
                        )
                        df_new = df_new.loc[valid_new_mask].copy()

                if not df_patch.empty:
                    base = merge_patch_by_item(base, df_patch)

                inserted = 0
                if not df_new.empty:
                    if "tags" not in df_new.columns:
                        df_new["tags"] = [[] for _ in range(len(df_new))]
                    if "tier" not in df_new.columns:
                        df_new["tier"] = pd.Series([pd.NA] * len(df_new), dtype="Int64")
                    if "stack_max" not in df_new.columns:
                        df_new["stack_max"] = pd.Series([pd.NA] * len(df_new), dtype="Int64")
                    inserted = len(df_new)
                    mask = ~base["item"].isin(df_new["item"])
                    base = pd.concat([base[mask], df_new], ignore_index=True)

                if df_patch.empty and df_new.empty:
                    st.info("Nenhuma atualização aplicada.")
                else:
                    save_items(base)
                    st.success(
                        "Cadastro importado: "
                        f"{inserted} novo(s), {len(df_patch)} atualizado(s)."
                    )
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

# --------------------------------------------------------------------------------------
# Coletar snapshot (scraper Devaloka)
# --------------------------------------------------------------------------------------
with tab_coletar:
    st.markdown("## Coletar snapshot (Devaloka)")

    # Parâmetros fixos (sem necessidade de entrada manual)
    settings_remote = {
        "Buy orders": DEFAULT_NWMP_BUY_SRC,
        "Sell orders": DEFAULT_NWMP_SELL_SRC,
        "Pasta RAW": DEFAULT_NWMP_RAW_ROOT,
        "CSV Buy (NWMP)": DEFAULT_NWMP_BUY_CSV,
        "CSV Sell (NWMP)": DEFAULT_NWMP_SELL_CSV,
        "Servidor": DEFAULT_NWMP_SERVER,
        "history_local.json": DEFAULT_HISTORY_JSON,
    }

    settings_local = {
        "Snapshot local (buy-orders)": DEFAULT_LOCAL_BUYORDERS_DIR,
        "Snapshot local (auctions)": DEFAULT_LOCAL_AUCTIONS_DIR,
    }

    st.caption("Parâmetros usados automaticamente para coleta/processamento")
    settings_rows = []
    for name, value in settings_remote.items():
        settings_rows.append({"Origem": "NWMP online", "Configuração": name, "Valor": str(value)})
    for name, value in settings_local.items():
        settings_rows.append({"Origem": "Snapshot local", "Configuração": name, "Valor": str(value)})
    settings_df = pd.DataFrame(settings_rows)
    st.table(settings_df)

    missing_remote = [name for name, value in settings_remote.items() if not str(value).strip()]
    missing_local = [name for name, value in settings_local.items() if not str(value).strip()]

    st.info(
        "Esta página baixa snapshots de buy e sell orders, atualiza ambos os CSVs "
        "e consolida o history_local.json."
    )

    # Importa o novo módulo do scraper
    try:
        import sys
        from pathlib import Path  # ← usar Path diretamente
        ROOT = Path.cwd()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import nwmp_sync
    except Exception as e:
        st.error(f"Falha ao importar nwmp_sync: {e}")
    else:
        source_labels = {
            "remote": "Snapshot online (NWMP)",
            "local": "Snapshot local (pastas current)",
        }

        def _load_last_sync_meta() -> Dict[str, Any]:
            meta_path = Path(DEFAULT_NWMP_RAW_ROOT) / "last_sync_meta.json"
            if not meta_path.exists():
                return {}
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        last_meta = _load_last_sync_meta()
        processed_remote = last_meta.get("remote", {}) if isinstance(last_meta, dict) else {}
        processed_local = last_meta.get("local", {}) if isinstance(last_meta, dict) else {}

        processed_remote_ts = processed_remote.get("snapshot_ts") or processed_remote.get("timestamp")
        processed_local_ts = processed_local.get("timestamp") or processed_local.get("snapshot_ts")

        if missing_remote:
            remote_probe: Dict[str, Any] = {
                "source": "remote",
                "snapshot_ts": None,
                "error": "Configurações obrigatórias ausentes",
            }
        else:
            try:
                remote_probe = nwmp_sync.probe_remote_snapshot(
                    DEFAULT_NWMP_BUY_SRC,
                    DEFAULT_NWMP_SELL_SRC,
                    timeout=15,
                )
            except Exception as exc:
                remote_probe = {"source": "remote", "snapshot_ts": None, "error": str(exc)}

        if missing_local:
            local_probe: Dict[str, Any] = {
                "source": "local",
                "snapshot_ts": None,
                "error": "Configurações obrigatórias ausentes",
            }
        else:
            try:
                local_probe = nwmp_sync.probe_local_snapshot(
                    settings_local["Snapshot local (buy-orders)"],
                    settings_local["Snapshot local (auctions)"],
                )
            except Exception as exc:
                local_probe = {"source": "local", "snapshot_ts": None, "error": str(exc)}

        def _format_counts(info: Dict[str, Any]) -> str:
            if not isinstance(info, dict):
                return ""
            parts: List[str] = []
            buy_entries = info.get("buy_entries")
            sell_entries = info.get("sell_entries")
            records = info.get("records")
            if isinstance(buy_entries, int):
                parts.append(f"{buy_entries:,} ordens de compra")
            if isinstance(sell_entries, int):
                parts.append(f"{sell_entries:,} ordens de venda")
            if isinstance(records, int):
                parts.append(f"{records:,} registros")
            return " • ".join(parts)

        def _render_snapshot_status(
            col, title: str, probe: Dict[str, Any], processed: Dict[str, Any], processed_ts: Optional[str]
        ) -> None:
            col.markdown(f"### {title}")
            probe_ts = probe.get("snapshot_ts") if isinstance(probe, dict) else None
            probe_counts = _format_counts(probe)
            if probe_ts:
                col.metric(
                    "Snapshot disponível",
                    probe_ts,
                    help=probe_counts if probe_counts else None,
                )
            elif probe.get("error"):
                col.warning(f"Não foi possível verificar: {probe['error']}")
            else:
                col.info("Não foi possível obter informações em tempo real.")

            processed_ts = processed_ts or processed.get("snapshot_ts") or processed.get("timestamp")
            processed_counts = _format_counts(processed)
            caption = f"Último processado: {processed_ts or '—'}"
            if processed_counts:
                caption += f"\n{processed_counts}"
            col.caption(caption)

        live_remote_ts = remote_probe.get("snapshot_ts") if isinstance(remote_probe, dict) else None
        live_local_ts = local_probe.get("snapshot_ts") if isinstance(local_probe, dict) else None

        status_remote = live_remote_ts or processed_remote_ts or "—"
        status_local = live_local_ts or processed_local_ts or "—"

        remote_col, local_col = st.columns(2)
        _render_snapshot_status(
            remote_col,
            "Disponível no site",
            remote_probe,
            processed_remote,
            processed_remote_ts,
        )
        _render_snapshot_status(
            local_col,
            "Disponível localmente",
            local_probe,
            processed_local,
            processed_local_ts,
        )

        def _coerce_dt(value: Optional[str]):
            if not value:
                return None
            ts = parse_iso(value)
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                return ts.to_pydatetime()
            return None

        latest_source: Optional[str] = None
        latest_dt = None
        for src, value in (("remote", live_remote_ts), ("local", live_local_ts)):
            dt = _coerce_dt(value)
            if dt and (latest_dt is None or dt > latest_dt):
                latest_dt = dt
                latest_source = src
        if latest_source is None:
            for src, value in (("remote", processed_remote_ts), ("local", processed_local_ts)):
                dt = _coerce_dt(value)
                if dt and (latest_dt is None or dt > latest_dt):
                    latest_dt = dt
                    latest_source = src

        def _format_result_message(source_name: str, payload: Optional[Dict[str, Any]]) -> str:
            if not isinstance(payload, dict):
                return f"{source_name} sincronizado ✅"
            counts = _format_counts(payload)
            if counts:
                return f"{source_name} sincronizado ✅ ({counts})"
            return f"{source_name} sincronizado ✅"

        def _run_remote_sync():
            if missing_remote:
                st.error(
                    "Configurações obrigatórias ausentes: "
                    + ", ".join(missing_remote)
                    + ". Defina-as via variáveis de ambiente."
                )
                return None
            try:
                with st.spinner("Baixando, salvando RAW e atualizando snapshot..."):
                    result = nwmp_sync.run_sync(
                        DEFAULT_NWMP_BUY_SRC,
                        DEFAULT_NWMP_SELL_SRC,
                        raw_root=DEFAULT_NWMP_RAW_ROOT,
                        buy_csv_path=DEFAULT_NWMP_BUY_CSV,
                        sell_csv_path=DEFAULT_NWMP_SELL_CSV,
                        history_json_path=DEFAULT_HISTORY_JSON,
                        server=DEFAULT_NWMP_SERVER,
                    )
                st.success(_format_result_message(source_labels["remote"], result))
                return result
            except Exception as exc:
                st.error(f"Falhou: {exc}")
                return None

        def _run_local_sync():
            if missing_local:
                st.error(
                    "Configurações obrigatórias ausentes: "
                    + ", ".join(missing_local)
                    + ". Defina-as via variáveis de ambiente."
                )
                return None
            try:
                buy_dir = settings_local["Snapshot local (buy-orders)"]
                buy_path = Path(buy_dir)
                if not buy_path.exists():
                    raise FileNotFoundError(str(buy_path))
                auctions_dir = settings_local["Snapshot local (auctions)"]
                auctions_path = Path(auctions_dir)
                if not auctions_path.exists():
                    raise FileNotFoundError(str(auctions_path))

                with st.spinner("Processando snapshot local e atualizando snapshot consolidado..."):
                    result = nwmp_sync.run_sync_local_snapshot(
                        buy_orders_dir=str(buy_path),
                        auctions_dir=str(auctions_path),
                        raw_root=DEFAULT_NWMP_RAW_ROOT,
                        buy_csv_path=DEFAULT_NWMP_BUY_CSV,
                        sell_csv_path=DEFAULT_NWMP_SELL_CSV,
                        history_json_path=DEFAULT_HISTORY_JSON,
                        server=DEFAULT_NWMP_SERVER,
                    )
                st.success(_format_result_message(source_labels["local"], result))
                return result
            except FileNotFoundError as exc:
                st.error(f"Pastas do snapshot local não encontradas: {exc}")
            except Exception as exc:
                st.error(f"Falhou: {exc}")
            return None

        actions_remote, actions_rebuild, actions_local, actions_auto = st.columns(4)
        if actions_remote.button("🔄 Sincronizar agora", use_container_width=True):
            _run_remote_sync()

        if actions_rebuild.button("🧱 Reprocessar tudo do RAW → CSV", use_container_width=True):
            if missing_remote:
                st.error(
                    "Configurações obrigatórias ausentes: "
                    + ", ".join(missing_remote)
                    + ". Defina-as via variáveis de ambiente."
                )
            else:
                try:
                    with st.spinner("Reconstruindo CSV de buy orders a partir de raw/buy.json..."):
                        nwmp_sync.run_rebuild(
                            raw_root=DEFAULT_NWMP_RAW_ROOT,
                            buy_csv_path=DEFAULT_NWMP_BUY_CSV,
                            sell_csv_path=DEFAULT_NWMP_SELL_CSV,
                            history_json_path=DEFAULT_HISTORY_JSON,
                            server=DEFAULT_NWMP_SERVER,
                        )
                    st.success("Rebuild concluído ✅")
                except Exception as exc:
                    st.error(f"Falhou: {exc}")

        if actions_local.button("💾 Sincronizar snapshot local", use_container_width=True):
            _run_local_sync()

        if actions_auto.button("🧭 Sincronizar fonte mais recente", use_container_width=True):
            if latest_source == "local":
                ts_msg = status_local if status_local != "—" else "desconhecido"
                st.info(
                    f"Executando {source_labels['local']} (snapshot {ts_msg})"
                )
                _run_local_sync()
            elif latest_source == "remote":
                ts_msg = status_remote if status_remote != "—" else "desconhecido"
                st.info(
                    f"Executando {source_labels['remote']} (snapshot {ts_msg})"
                )
                _run_remote_sync()
            else:
                st.info(
                    "Nenhum snapshot anterior registrado; executando sincronização online padrão."
                )
                _run_remote_sync()

        # Mostrar prévia do CSV NWMP e do history_local.json (se existirem)
        import pandas as pd
        prev1, prev2 = st.columns(2)
        try:
            csv_path = DEFAULT_NWMP_BUY_CSV
            if Path(csv_path).exists():
                df_csv = pd.read_csv(csv_path)
                prev1.caption(f"Prévia CSV Buy NWMP: {csv_path}")
                prev1.dataframe(df_csv.tail(50), use_container_width=True)
        except Exception:
            pass
        try:
            if Path(DEFAULT_HISTORY_JSON).exists():
                df_hist = pd.read_json(DEFAULT_HISTORY_JSON, orient="records")
                prev2.caption(f"Prévia history_local.json (app): {DEFAULT_HISTORY_JSON}")
                prev2.dataframe(df_hist.tail(50), use_container_width=True)
        except Exception:
            pass

