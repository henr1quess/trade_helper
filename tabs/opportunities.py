# opportunities.py — Presets snapshot + include/exclude hierárquico + ROI/Preço ao lado + fixes
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

# Prefixo para chaves de estado desta aba
OP_PREFIX = "op_"

# Keys de botões (não entram no snapshot!)
BUTTON_KEYS_DENYLIST_PREFIXES = (
    f"{OP_PREFIX}preset_btn_",
)
BUTTON_KEYS_DENYLIST_EXACT = {
    f"{OP_PREFIX}save_state",
    f"{OP_PREFIX}mgr_del",
    f"{OP_PREFIX}mgr_rename",
}

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

    # snapshots antigos em centavos
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
        p.setdefault("state", {})
    return presets

def _save_presets(presets: list[dict]):
    _write_json(PRESETS_PATH, {"presets": presets})

def _find_preset(presets: list[dict], name: str) -> int:
    for i, p in enumerate(presets):
        if p.get("name","").strip().lower() == name.strip().lower():
            return i
    return -1

def _is_button_key(k: str) -> bool:
    if k in BUTTON_KEYS_DENYLIST_EXACT:
        return True
    return any(k.startswith(pref) for pref in BUTTON_KEYS_DENYLIST_PREFIXES)

def _capture_state_from_session() -> dict:
    snap = {}
    for k, v in st.session_state.items():
        if isinstance(k, str) and k.startswith(OP_PREFIX) and not _is_button_key(k):
            snap[k] = v
    return snap

def _apply_state_to_session(state: dict):
    for k, v in (state or {}).items():
        if not _is_button_key(k):
            st.session_state[k] = v

# --------------------------------------------------------------------------------------
# Filter logic
# --------------------------------------------------------------------------------------
def _apply_filters(
    df: pd.DataFrame,
    inc_cats: list[str],
    inc_fams: list[str],
    inc_groups: list[str],
    exc_cats: list[str],
    exc_fams: list[str],
    exc_groups: list[str],
    include_items: list[str],
    exclude_items: list[str],
    roi_min: float | None,
    roi_max: float | None,
    price_min: float | None,
    price_max: float | None,
    price_field: str = "low_sell",
) -> pd.DataFrame:

    base = df.copy()

    # filtros por hierarquia (include)
    if inc_cats:
        base = base[base["trading_category"].isin(inc_cats)]
    if inc_fams:
        base = base[base["trading_family"].isin(inc_fams)]
    if inc_groups:
        base = base[base["trading_group"].isin(inc_groups)]

    # exclusões por hierarquia
    if exc_cats:
        base = base[~base["trading_category"].isin(exc_cats)]
    if exc_fams:
        base = base[~base["trading_family"].isin(exc_fams)]
    if exc_groups:
        base = base[~base["trading_group"].isin(exc_groups)]

    # incluir itens específicos mesmo se ficaram fora
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

    # Preço unitário range
    fld = "low_sell" if price_field not in {"low_sell","top_buy"} else price_field
    if price_min is not None:
        base = base[base[fld] >= price_min]
    if price_max is not None:
        base = base[base[fld] <= price_max]

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

    # Acesso rápido
    st.markdown("[Ir direto para resultados](#results_anchor)")

    # ---------------- Chips de presets ----------------
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

    # ---------------- Estado inicial (só 1x) ----------------
    def _init(key, val):
        k = f"{OP_PREFIX}{key}"
        if k not in st.session_state:
            st.session_state[k] = val

    # Hierarquia
    _init("flt_cats_inc", [])
    _init("flt_fams_inc", [])
    _init("flt_grps_inc", [])
    _init("flt_cats_exc", [])
    _init("flt_fams_exc", [])
    _init("flt_grps_exc", [])
    # Itens finos
    _init("flt_include_items", [])
    _init("flt_exclude_items", [])
    # Intervalos
    _init("roi_min", 0.0)
    _init("roi_max", None)
    _init("price_min", 0.0)
    _init("price_max", None)
    _init("price_field", "low_sell")
    # Busca textual por item (nome/ID)
    _init("search_item", "")
    # Controle de exibição das configurações
    _init("search_cfg_collapsed", False)

    # ---------------- Layout em 2 colunas ----------------
    st.checkbox(
        "Minimizar configurações de busca",
        key=f"{OP_PREFIX}search_cfg_collapsed",
        help="Oculta filtros avançados e mantém visível apenas o campo de busca por item.",
    )

    if st.session_state[f"{OP_PREFIX}search_cfg_collapsed"]:
        st.info("Configurações minimizadas. Desmarque a opção acima para ajustar os filtros.")
    else:
        col_left, col_right = st.columns([1.6, 1], vertical_alignment="top")

        # ===== LEFT: filtros hierárquicos e itens =====
        with col_left:
            st.markdown("### Filtros por Category / Family / Group")

            all_cats = sorted(df["trading_category"].unique().tolist())
            all_fams = sorted(df["trading_family"].unique().tolist())
            all_groups = sorted(df["trading_group"].unique().tolist())

            # Category include/exclude
            sel_cats_inc = st.multiselect(
                "Category (incluir)", all_cats, key=f"{OP_PREFIX}flt_cats_inc",
                placeholder="Selecione 1+ categorias"
            )
            sel_cats_exc = st.multiselect(
                "Category (excluir)", [c for c in all_cats if c not in sel_cats_inc],
                key=f"{OP_PREFIX}flt_cats_exc", placeholder="Opcional"
            )

            # Family options condicionadas
            fam_pool = df[df["trading_category"].isin(sel_cats_inc)] if sel_cats_inc else df
            fam_opts = sorted(fam_pool["trading_family"].unique().tolist())
            sel_fams_inc = st.multiselect(
                "Family (incluir)", fam_opts, key=f"{OP_PREFIX}flt_fams_inc",
                placeholder="Selecione 1+ families"
            )
            fam_exc_opts = [f for f in fam_opts if f not in sel_fams_inc]
            sel_fams_exc = st.multiselect(
                "Family (excluir)", fam_exc_opts, key=f"{OP_PREFIX}flt_fams_exc", placeholder="Opcional"
            )

            # Group options condicionadas
            grp_pool = fam_pool[fam_pool["trading_family"].isin(sel_fams_inc)] if sel_fams_inc else fam_pool
            grp_opts = sorted(grp_pool["trading_group"].unique().tolist())
            sel_grps_inc = st.multiselect(
                "Group (incluir)", grp_opts, key=f"{OP_PREFIX}flt_grps_inc",
                placeholder="Selecione 1+ groups"
            )
            grp_exc_opts = [g for g in grp_opts if g not in sel_grps_inc]
            sel_grps_exc = st.multiselect(
                "Group (excluir)", grp_exc_opts, key=f"{OP_PREFIX}flt_grps_exc", placeholder="Opcional"
            )

            st.markdown("### Refinar por itens (opcional)")
            # universo de itens baseado só no include (faz sentido para procurar rapidamente)
            df_scope = df.copy()
            if sel_cats_inc:
                df_scope = df_scope[df_scope["trading_category"].isin(sel_cats_inc)]
            if sel_fams_inc:
                df_scope = df_scope[df_scope["trading_family"].isin(sel_fams_inc)]
            if sel_grps_inc:
                df_scope = df_scope[df_scope["trading_group"].isin(sel_grps_inc)]

            item_choices = (
                df_scope.loc[:, ["item_id","item_name"]]
                .drop_duplicates()
                .sort_values("item_name")
            )
            id_to_label = {r.item_id: f'{r.item_name} · ({r.item_id})' for r in item_choices.itertuples()}
            all_ids = list(id_to_label.keys())

            st.multiselect(
                "Sempre incluir estes itens", all_ids,
                key=f"{OP_PREFIX}flt_include_items",
                format_func=lambda x: id_to_label.get(x, x)
            )
            st.multiselect(
                "Sempre excluir estes itens", all_ids,
                key=f"{OP_PREFIX}flt_exclude_items",
                format_func=lambda x: id_to_label.get(x, x)
            )

        # ===== RIGHT: intervalos =====
        with col_right:
            st.markdown("### Intervalos")
            roi_min = st.number_input(
                "ROI mínimo (%)",
                value=st.session_state[f"{OP_PREFIX}roi_min"],
                step=0.5, format="%.2f", key=f"{OP_PREFIX}roi_min_input"
            )
            # atualiza o estado (sem defaults)
            st.session_state[f"{OP_PREFIX}roi_min"] = float(roi_min)

            roi_max_curr = st.session_state[f"{OP_PREFIX}roi_max"]
            roi_max = st.number_input(
                "ROI máximo (%) (opcional)",
                value=roi_max_curr if roi_max_curr is not None else 1000.0,
                step=0.5, format="%.2f", key=f"{OP_PREFIX}roi_max_input"
            )
            if st.checkbox("Sem teto de ROI", value=(roi_max_curr is None), key=f"{OP_PREFIX}no_roi_max"):
                st.session_state[f"{OP_PREFIX}roi_max"] = None
            else:
                st.session_state[f"{OP_PREFIX}roi_max"] = float(roi_max)

            st.radio(
                "Preço base", options=["low_sell","top_buy"],
                format_func=lambda x: "Low Sell" if x=="low_sell" else "Top Buy",
                horizontal=True,
                key=f"{OP_PREFIX}price_field"
            )

            price_min = st.number_input(
                "Preço mínimo (coins)",
                value=st.session_state[f"{OP_PREFIX}price_min"],
                step=0.01, format="%.2f", key=f"{OP_PREFIX}price_min_input"
            )
            st.session_state[f"{OP_PREFIX}price_min"] = float(price_min)

            price_max_curr = st.session_state[f"{OP_PREFIX}price_max"]
            price_max = st.number_input(
                "Preço máximo (coins) (opcional)",
                value=price_max_curr if price_max_curr is not None else 1_000_000.0,
                step=0.01, format="%.2f", key=f"{OP_PREFIX}price_max_input"
            )
            if st.checkbox("Sem teto de preço", value=(price_max_curr is None), key=f"{OP_PREFIX}no_price_max"):
                st.session_state[f"{OP_PREFIX}price_max"] = None
            else:
                st.session_state[f"{OP_PREFIX}price_max"] = float(price_max)

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

    # ---------------- Busca + Resultado filtrado ----------------
    # Barra de busca por item (aplica sobre o resultado)
    st.text_input(
        "Buscar item",
        key=f"{OP_PREFIX}search_item",
        placeholder="Digite parte do nome ou ID",
    )

    df_out = _apply_filters(
        df,
        inc_cats=st.session_state[f"{OP_PREFIX}flt_cats_inc"],
        inc_fams=st.session_state[f"{OP_PREFIX}flt_fams_inc"],
        inc_groups=st.session_state[f"{OP_PREFIX}flt_grps_inc"],
        exc_cats=st.session_state[f"{OP_PREFIX}flt_cats_exc"],
        exc_fams=st.session_state[f"{OP_PREFIX}flt_fams_exc"],
        exc_groups=st.session_state[f"{OP_PREFIX}flt_grps_exc"],
        include_items=st.session_state[f"{OP_PREFIX}flt_include_items"],
        exclude_items=st.session_state[f"{OP_PREFIX}flt_exclude_items"],
        roi_min=st.session_state[f"{OP_PREFIX}roi_min"],
        roi_max=st.session_state[f"{OP_PREFIX}roi_max"],
        price_min=st.session_state[f"{OP_PREFIX}price_min"],
        price_max=st.session_state[f"{OP_PREFIX}price_max"],
        price_field=st.session_state[f"{OP_PREFIX}price_field"],
    )

    # Aplica filtro de busca textual (case-insensitive) em nome/ID
    q = str(st.session_state.get(f"{OP_PREFIX}search_item", "") or "").strip().lower()
    if q:
        df_out = df_out[
            df_out["item_name"].str.lower().str.contains(q, na=False)
            | df_out["item_id"].str.lower().str.contains(q, na=False)
        ]

    st.markdown("<div id='results_anchor'></div>", unsafe_allow_html=True)
    st.markdown("### Resultados")
    if df_out.empty:
        st.warning("Nenhum item após aplicar os filtros.")
        return

    # Ordena por ROI numérico desc por padrão
    df_out = df_out.sort_values("roi_pct", ascending=False, na_position="last")

    # VIEW: mantenha colunas numéricas como números (sem formatar para string)
    df_view = df_out.loc[:, [
        "item_name", "trading_category", "trading_family", "trading_group",
        "top_buy", "low_sell", "roi_pct"
    ]].copy()

    # Renomeia colunas finais — importantes: Top Buy / Low Sell / ROI ficam NUMÉRICAS
    df_view = df_view.rename(columns={
        "item_name": "Item",
        "trading_category": "Category",
        "trading_family": "Family",
        "trading_group": "Group",
        "top_buy": "Top Buy",      # numérico
        "low_sell": "Low Sell",    # numérico
        "roi_pct": "ROI",          # numérico
    })[["Item", "Category", "Family", "Group", "Top Buy", "Low Sell", "ROI"]]

    # Garante dtype numérico (se vieram NaNs/objetos de snapshots antigos)
    for col in ["Top Buy", "Low Sell", "ROI"]:
        df_view[col] = pd.to_numeric(df_view[col], errors="coerce")

    df_styled = df_view.style.format({
        "ROI": _roi_badge,
    })

    st.dataframe(
        df_styled,
        use_container_width=True,
        height=680,
        hide_index=True,
        column_config={
            # Numéricas: agora o sort do Streamlit funciona corretamente
            "Top Buy": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Low Sell": st.column_config.NumberColumn(format="%.2f", width="small"),
            # Texto/visuais:
            "Category": st.column_config.TextColumn(width="medium"),
            "Family": st.column_config.TextColumn(width="medium"),
            "Group": st.column_config.TextColumn(width="medium"),
            "Item": st.column_config.TextColumn(width="large"),
            "ROI": st.column_config.Column(width="small"),
        }
    )
