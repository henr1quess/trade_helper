# opportunities.py — Presets com snapshot completo + filtros ROI/Preço (corrigido)
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Paths e constantes
# --------------------------------------------------------------------------------------
PROJECT_DIR = Path.cwd()
RAW_DIR = PROJECT_DIR / "raw"
LATEST_SNAPSHOT_PATH = RAW_DIR / "latest_snapshot.json"
PRESETS_PATH = RAW_DIR / "opportunity_presets.json"

# Tudo que for estado desta aba deve usar este prefixo para entrar no snapshot
OP_PREFIX = "op_"

# --------------------------------------------------------------------------------------
# IO utils
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def _load_snapshot_df(snapshot_path: Path) -> pd.DataFrame:
    snap = _read_json(snapshot_path, {}) or {}
    recs = (snap.get("records") or []) if isinstance(snap, dict) else []
    price_scale = str((snap.get("price_scale") or "") if isinstance(snap, dict) else "").strip().lower()

    df = pd.DataFrame(recs)
    need = ["item_id","item_name","trading_category","trading_family","trading_group","top_buy","low_sell"]
    for c in need:
        if c not in df.columns:
            df[c] = None

    df["top_buy"] = pd.to_numeric(df["top_buy"], errors="coerce")
    df["low_sell"] = pd.to_numeric(df["low_sell"], errors="coerce")

    # snapshots antigos vinham em centavos; escala atual "coin"/"coins" já é moeda
    if price_scale not in {"coin", "coins"}:
        df["top_buy"] = df["top_buy"] / 100.0
        df["low_sell"] = df["low_sell"] / 100.0

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

    # normalizações úteis
    for c in ["trading_category","trading_family","trading_group","item_id","item_name"]:
        df[c] = df[c].astype(str)
    return df

# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Presets (CRUD) — snapshots completos de estado
# --------------------------------------------------------------------------------------
def _load_presets() -> list[dict]:
    data = _read_json(PRESETS_PATH, {"presets": []}) or {"presets": []}
    presets = data.get("presets") or []
    for p in presets:
        p.setdefault("name", "")
        p.setdefault("state", {})  # estado completo desta aba
    return presets

def _save_presets(presets: list[dict]):
    _write_json(PRESETS_PATH, {"presets": presets})

def _find_preset(presets: list[dict], name: str) -> int:
    for i, p in enumerate(presets):
        if p.get("name","").strip().lower() == name.strip().lower():
            return i
    return -1

def _capture_state_from_session() -> dict:
    snap = {}
    for k, v in st.session_state.items():
        if isinstance(k, str) and k.startswith(OP_PREFIX):
            snap[k] = v
    return snap

def _apply_state_to_session(state: dict):
    for k, v in (state or {}).items():
        st.session_state[k] = v

# --------------------------------------------------------------------------------------
# Filter logic
# --------------------------------------------------------------------------------------
def _apply_filters(
    df: pd.DataFrame,
    inc_cats: list[str],
    inc_fams: list[str],
    inc_groups: list[str],
    include_items: list[str],
    exclude_items: list[str],
    roi_min: float | None,
    roi_max: float | None,
    price_min: float | None,
    price_max: float | None,
    price_field: str = "low_sell",  # "low_sell" ou "top_buy"
) -> pd.DataFrame:

    base = df.copy()

    # filtros por hierarquia
    if inc_cats:
        base = base[base["trading_category"].isin(inc_cats)]
    if inc_fams:
        base = base[base["trading_family"].isin(inc_fams)]
    if inc_groups:
        base = base[base["trading_group"].isin(inc_groups)]

    # incluir itens específicos mesmo se ficaram fora dos grupos
    if include_items:
        extra = df[df["item_id"].isin(include_items)]
        base = pd.concat([base, extra], ignore_index=True).drop_duplicates(subset=["item_id"])

    # excluir itens específicos
    if exclude_items:
        base = base[~base["item_id"].isin(exclude_items)]

    # ROI range
    if roi_min is not None:
        base = base[base["roi_pct"] >= roi_min]
    if roi_max is not None:
        base = base[base["roi_pct"] <= roi_max]

    # Preço unitário range (aplicado ao campo escolhido)
    fld = "low_sell" if price_field not in {"low_sell","top_buy"} else price_field
    if price_min is not None:
        base = base[base[fld] >= price_min]  # <-- corrigido
    if price_max is not None:
        base = base[base[fld] <= price_max]  # <-- corrigido

    return base

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
# Retrocompat: aceita (snapshot_path, *_) para compatibilidade com chamadas antigas
def render(snapshot_path: Path | None = None, *_, **__):
    st.title("📈 Opportunities")
    snapshot_path = snapshot_path or LATEST_SNAPSHOT_PATH
    df = _load_snapshot_df(snapshot_path)

    if df.empty:
        st.info("Nenhum snapshot processado encontrado em raw/latest_snapshot.json.")
        return

    presets = _load_presets()
    preset_names = [p["name"] for p in presets]

    # ---------------- Chips de presets (aplicar rápido) ----------------
    st.caption("Presets salvos")
    if not presets:
        st.write("— (nenhum preset salvo ainda) —")
    else:
        cols = st.columns(max(3, min(6, len(presets))))
        for i, p in enumerate(presets):
            if cols[i % len(cols)].button(p["name"], key=f"{OP_PREFIX}preset_btn_{p['name']}"):
                _apply_state_to_session(p["state"])
                st.success(f'Preset “{p["name"]}” aplicado.')
                st.rerun()
    st.divider()

    # ---------------- Estado inicial (sempre prefixado) ----------------
    def _init(key, val):
        k = f"{OP_PREFIX}{key}"
        if k not in st.session_state:
            st.session_state[k] = val

    _init("flt_cats", [])
    _init("flt_fams", [])
    _init("flt_grps", [])
    _init("flt_include_items", [])
    _init("flt_exclude_items", [])
    _init("roi_min", None)           # float|None
    _init("roi_max", None)
    _init("price_min", None)
    _init("price_max", None)
    _init("price_field", "low_sell")  # "low_sell" ou "top_buy"

    # ---------------- Seletor hierárquico ----------------
    st.markdown("### Filtros por Category / Family / Group")

    all_cats = sorted(df["trading_category"].unique().tolist())
    all_fams = sorted(df["trading_family"].unique().tolist())
    all_groups = sorted(df["trading_group"].unique().tolist())

    sel_cats = st.multiselect(
        "Category (incluir)", all_cats,
        default=st.session_state[f"{OP_PREFIX}flt_cats"],
        placeholder="Selecione 1+ categorias"
    )

    fam_pool = df[df["trading_category"].isin(sel_cats)] if sel_cats else df
    fam_opts = sorted(fam_pool["trading_family"].unique().tolist())
    sel_fams = st.multiselect(
        "Family (incluir)", fam_opts,
        default=st.session_state[f"{OP_PREFIX}flt_fams"],
        placeholder="Selecione 1+ families"
    )

    grp_pool = fam_pool[fam_pool["trading_family"].isin(sel_fams)] if sel_fams else fam_pool
    grp_opts = sorted(grp_pool["trading_group"].unique().tolist())
    sel_grps = st.multiselect(
        "Group (incluir)", grp_opts,
        default=st.session_state[f"{OP_PREFIX}flt_grps"],
        placeholder="Selecione 1+ groups"
    )

    st.markdown("### Refinar por itens (opcional)")
    df_scope = df.copy()
    if sel_cats:
        df_scope = df_scope[df_scope["trading_category"].isin(sel_cats)]
    if sel_fams:
        df_scope = df_scope[df_scope["trading_family"].isin(sel_fams)]
    if sel_grps:
        df_scope = df_scope[df_scope["trading_group"].isin(sel_grps)]

    item_choices = (
        df_scope.loc[:, ["item_id","item_name"]]
        .drop_duplicates()
        .sort_values("item_name")
    )
    id_to_label = {r.item_id: f'{r.item_name} · ({r.item_id})' for r in item_choices.itertuples()}
    all_ids = list(id_to_label.keys())

    include_items = st.multiselect(
        "Sempre incluir estes itens", all_ids,
        default=st.session_state[f"{OP_PREFIX}flt_include_items"],
        format_func=lambda x: id_to_label.get(x, x)
    )
    exclude_items = st.multiselect(
        "Sempre excluir estes itens", all_ids,
        default=st.session_state[f"{OP_PREFIX}flt_exclude_items"],
        format_func=lambda x: id_to_label.get(x, x)
    )

    # ---------------- Intervalos ROI e Preço (inputs com +/-) ----------------
    st.markdown("### Intervalos")
    col_r1, col_r2, col_r3 = st.columns([1,1,1])
    with col_r1:
        roi_min = st.number_input(
            "ROI mínimo (%)",
            value=st.session_state[f"{OP_PREFIX}roi_min"] or 0.0,
            step=0.5, format="%.2f", key=f"{OP_PREFIX}roi_min_input"
        )
    with col_r2:
        roi_max_val = st.session_state[f"{OP_PREFIX}roi_max"]
        roi_max_default = roi_max_val if roi_max_val is not None else 1000.0
        roi_max = st.number_input(
            "ROI máximo (%) (opcional)",
            value=roi_max_default,
            step=0.5, format="%.2f", key=f"{OP_PREFIX}roi_max_input"
        )
    with col_r3:
        no_roi_max = st.checkbox("Sem teto de ROI", value=(st.session_state[f"{OP_PREFIX}roi_max"] is None), key=f"{OP_PREFIX}no_roi_max")
        if no_roi_max:
            roi_max = None

    col_p0, col_p1, col_p2 = st.columns([1,1,1])
    with col_p0:
        price_field = st.radio(
            "Preço base", options=["low_sell","top_buy"],
            format_func=lambda x: "Low Sell" if x=="low_sell" else "Top Buy",
            horizontal=True,
            index=0 if st.session_state[f"{OP_PREFIX}price_field"]=="low_sell" else 1,
            key=f"{OP_PREFIX}price_field_radio"
        )
    with col_p1:
        price_min = st.number_input(
            "Preço mínimo (coins)",
            value=st.session_state[f"{OP_PREFIX}price_min"] or 0.0,
            step=0.01, format="%.2f", key=f"{OP_PREFIX}price_min_input"
        )
    with col_p2:
        price_max_val = st.session_state[f"{OP_PREFIX}price_max"]
        price_max_default = price_max_val if price_max_val is not None else 1_000_000.0
        price_max = st.number_input(
            "Preço máximo (coins) (opcional)",
            value=price_max_default,
            step=0.01, format="%.2f", key=f"{OP_PREFIX}price_max_input"
        )
    no_price_max = st.checkbox("Sem teto de preço", value=(st.session_state[f"{OP_PREFIX}price_max"] is None), key=f"{OP_PREFIX}no_price_max")
    if no_price_max:
        price_max = None

    # ---------------- Persistir estado corrente ----------------
    st.session_state[f"{OP_PREFIX}flt_cats"] = sel_cats
    st.session_state[f"{OP_PREFIX}flt_fams"] = sel_fams
    st.session_state[f"{OP_PREFIX}flt_grps"] = sel_grps
    st.session_state[f"{OP_PREFIX}flt_include_items"] = include_items
    st.session_state[f"{OP_PREFIX}flt_exclude_items"] = exclude_items
    st.session_state[f"{OP_PREFIX}roi_min"] = float(roi_min) if roi_min is not None else None
    st.session_state[f"{OP_PREFIX}roi_max"] = float(roi_max) if roi_max is not None else None
    st.session_state[f"{OP_PREFIX}price_min"] = float(price_min) if price_min is not None else None
    st.session_state[f"{OP_PREFIX}price_max"] = float(price_max) if price_max is not None else None
    st.session_state[f"{OP_PREFIX}price_field"] = price_field

    # ---------------- Salvar preset (snapshot completo) ----------------
    st.markdown("### Preset: salvar estado atual")
    col_sv1, col_sv2, col_sv3 = st.columns([2,1,1])
    with col_sv1:
        new_name = st.text_input("Nome do preset", placeholder="ex.: farm com baú", key=f"{OP_PREFIX}preset_name")
    with col_sv2:
        overwrite = st.checkbox("Sobrescrever se existir", value=False, key=f"{OP_PREFIX}overwrite")
    with col_sv3:
        if st.button("Salvar estado atual", type="primary", width="stretch", key=f"{OP_PREFIX}save_state"):
            nm = (new_name or "").strip()
            if not nm:
                st.error("Defina um nome para o preset.")
            else:
                snap = _capture_state_from_session()
                idx = _find_preset(presets, nm)
                payload = {"name": nm, "state": snap}
                if idx >= 0 and not overwrite:
                    st.error("Já existe um preset com esse nome. Marque 'Sobrescrever' para atualizar.")
                else:
                    if idx >= 0:
                        presets[idx] = payload
                    else:
                        presets.append(payload)
                    _save_presets(presets)
                    st.success(f'Preset “{nm}” salvo.')
                    st.rerun()

    # Gerenciar (apagar/renomear)
    with st.expander("Gerenciar presets"):
        sel = st.selectbox("Selecionar", ["(nenhum)"] + preset_names, index=0, key=f"{OP_PREFIX}mgr_sel")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if st.button("Apagar", disabled=(sel=="(nenhum)"), key=f"{OP_PREFIX}mgr_del"):
                i = _find_preset(presets, sel)
                if i >= 0:
                    presets.pop(i)
                    _save_presets(presets)
                    st.success(f'Preset “{sel}” apagado.')
                    st.rerun()
        with col_g2:
            newlabel = st.text_input("Renomear para", key=f"{OP_PREFIX}mgr_newlabel")
            if st.button("Renomear", disabled=(sel=="(nenhum)" or not (newlabel or "").strip()), key=f"{OP_PREFIX}mgr_rename"):
                i = _find_preset(presets, sel)
                if i >= 0:
                    presets[i]["name"] = newlabel.strip()
                    _save_presets(presets)
                    st.success("Renomeado.")
                    st.rerun()

    # ---------------- Resultado filtrado ----------------
    df_out = _apply_filters(
        df,
        sel_cats, sel_fams, sel_grps,
        include_items, exclude_items,
        roi_min=st.session_state[f"{OP_PREFIX}roi_min"],
        roi_max=st.session_state[f"{OP_PREFIX}roi_max"],
        price_min=st.session_state[f"{OP_PREFIX}price_min"],
        price_max=st.session_state[f"{OP_PREFIX}price_max"],
        price_field=st.session_state[f"{OP_PREFIX}price_field"],
    )

    st.markdown("### Resultados")
    if df_out.empty:
        st.warning("Nenhum item após aplicar os filtros.")
        return

    show_cols = [
        "item_name","item_id","trading_category","trading_family","trading_group",
        "top_buy","low_sell","roi_pct"
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    df_view = df_out.loc[:, show_cols].copy()
    df_view["top_buy_fmt"] = df_view["top_buy"].map(_fmt_money)
    df_view["low_sell_fmt"] = df_view["low_sell"].map(_fmt_money)
    df_view["roi"] = df_view["roi_pct"].map(_roi_badge)

    df_view = df_view.rename(columns={
        "item_name": "Item",
        "item_id": "Item ID",
        "trading_category": "Category",
        "trading_family": "Family",
        "trading_group": "Group",
        "top_buy_fmt": "Top Buy",
        "low_sell_fmt": "Low Sell",
        "roi": "ROI"
    })[["Item","Item ID","Category","Family","Group","Top Buy","Low Sell","ROI"]]

    st.dataframe(
        df_view,
        width="stretch",
        hide_index=True,
        column_config={
            "Top Buy": st.column_config.TextColumn(width="small"),
            "Low Sell": st.column_config.TextColumn(width="small"),
            "ROI": st.column_config.TextColumn(width="small"),
            "Item ID": st.column_config.TextColumn(width="medium"),
        }
    )
