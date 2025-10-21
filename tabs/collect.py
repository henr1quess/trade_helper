# tabs/collect.py
from __future__ import annotations
from pathlib import Path
import json
import datetime as dt
import streamlit as st
import nwmp_sync as sync
from zoneinfo import ZoneInfo

BRT = ZoneInfo("America/Sao_Paulo")

# ----- IO -----
def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ----- TS helpers -----
def _to_iso_utc(ts_val):
    if ts_val is None:
        return None
    if isinstance(ts_val, str) and "T" in ts_val:
        return ts_val
    try:
        if isinstance(ts_val, str):
            ts_val = float(ts_val)
        if isinstance(ts_val, (int, float)):
            return dt.datetime.fromtimestamp(ts_val, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return None

def _parse_iso(ts: str):
    if not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.datetime.fromisoformat(ts)
    except Exception:
        return None

def _fmt_ts(ts_iso: str | None, brt: bool) -> str | None:
    if not ts_iso:
        return None
    d = _parse_iso(ts_iso)
    if not d:
        return ts_iso
    if brt:
        return d.astimezone(BRT).strftime("%Y-%m-%d %H:%M:%S %Z")
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _extract_any_ts(entry: dict):
    if not isinstance(entry, dict):
        return None
    for key in ("timestamp", "created_at", "expires_at"):
        if key in entry:
            iso = _to_iso_utc(entry[key])
            if iso:
                return iso
    return None

def _max_ts_from_array(arr):
    if not isinstance(arr, list) or not arr:
        return None, None
    best_raw, best_dt = None, None
    for x in arr:
        ts_iso = _extract_any_ts(x)
        d = _parse_iso(ts_iso) if ts_iso else None
        if d and (best_dt is None or d > best_dt):
            best_dt, best_raw = d, ts_iso
    return best_raw, best_dt

def _arr_info(path: Path):
    out = {"exists": False, "entries": 0, "ts_raw": None, "ts_dt": None, "ts_display": None}
    if not path.exists():
        return out
    out["exists"] = True
    arr = _read_json(path)
    if isinstance(arr, list):
        out["entries"] = len(arr)
        ts_raw, ts_dt = _max_ts_from_array(arr)
        out["ts_raw"], out["ts_dt"] = ts_raw, ts_dt
    if out["ts_dt"] is None:
        m = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        out["ts_display"] = m.isoformat().replace("+00:00", "Z")
    else:
        out["ts_display"] = out["ts_raw"]
    return out

# ----- UI blocks -----
def _four_status_cards(show_brt: bool):
    st.markdown("### Status dos arquivos (CLOUD × LOCAL)")
    col1, col2 = st.columns(2)

    ci_buy = _arr_info(sync.CLOUD_BUY_PATH)
    ci_sell = _arr_info(sync.CLOUD_SELL_PATH)
    li_buy = _arr_info(sync.LOCAL_BUY_PATH)
    li_sell = _arr_info(sync.LOCAL_SELL_PATH)

    with col1:
        st.markdown("**CLOUD / BUY**")
        st.code(f"{sync.CLOUD_BUY_PATH.name}", language="text")
        st.success(f"ts: {_fmt_ts(ci_buy['ts_display'], show_brt) or '—'}   |   registros: {ci_buy['entries']}")
    with col2:
        st.markdown("**CLOUD / SELL**")
        st.code(f"{sync.CLOUD_SELL_PATH.name}", language="text")
        st.success(f"ts: {_fmt_ts(ci_sell['ts_display'], show_brt) or '—'}   |   registros: {ci_sell['entries']}")

    with col1:
        st.markdown("**LOCAL / BUY**")
        st.code(f"{sync.LOCAL_BUY_PATH.name}", language="text")
        st.info(f"ts: {_fmt_ts(li_buy['ts_display'], show_brt) or '—'}     |   registros: {li_buy['entries']}")
    with col2:
        st.markdown("**LOCAL / SELL**")
        st.code(f"{sync.LOCAL_SELL_PATH.name}", language="text")
        st.info(f"ts: {_fmt_ts(li_sell['ts_display'], show_brt) or '—'}     |   registros: {li_sell['entries']}")

    return ci_buy, ci_sell, li_buy, li_sell

def _config_panel():
    st.markdown("### Fontes locais padrão (somente leitura)")
    cfg = sync.get_local_source_config()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Arquivo de config detectado:**")
        st.code(cfg.get("config_path") or "—", language="text")
    with c2:
        st.write("**Caminhos:**")
        st.caption(f"BUY  → {cfg.get('buy_file') or '—'}")
        st.caption(f"SELL → {cfg.get('sell_dir') or '—'}")

# ----- Render -----
def render():
    st.subheader("☁️ Coletar snapshot")

    show_brt = st.checkbox("Mostrar horários em Brasília (BRT)", value=True)

    a, b, c, d = st.columns([1, 1, 1, 1])

    with a:
        if st.button("☁️ Baixar da nuvem", use_container_width=True):
            try:
                meta = sync.download_cloud_snapshots()
                st.success(
                    f"Cloud baixado: BUY={meta['saved']['cloud_buy_entries']} | "
                    f"SELL={meta['saved']['cloud_sell_entries']}"
                )
            except Exception as e:
                st.error(f"Erro ao baixar: {e}")

    with b:
        if st.button("📥 Copiar snapshots locais (padrões)", use_container_width=True):
            try:
                sync.copy_local_from_defaults()
                st.success("Snapshots locais copiados para `raw/collected/`.")
            except Exception as e:
                st.error(f"Erro ao copiar dos caminhos padrão: {e}")

    with c:
        if st.button("🔄 Atualizar status", use_container_width=True):
            st.rerun()

    # Escolha manual de fonte
    st.divider()
    st.markdown("### Escolha da fonte para processamento")
    st.caption("Defina manualmente de onde o processamento vai ler cada lado (ou deixe em Auto).")

    colx, coly = st.columns(2)
    with colx:
        prefer_buy = st.radio(
            "Fonte de **BUY**",
            ["Auto (mais recente)", "Forçar CLOUD", "Forçar LOCAL"],
            index=0,
            horizontal=True,
        )
    with coly:
        prefer_sell = st.radio(
            "Fonte de **SELL**",
            ["Auto (mais recente)", "Forçar CLOUD", "Forçar LOCAL"],
            index=0,
            horizontal=True,
        )

    st.divider()
    _config_panel()

    st.divider()
    _four_status_cards(show_brt)

    st.divider()
    d_col, = st.columns(1)
    with d_col:
        if st.button("⚙️ Processar snapshot", use_container_width=True):
            try:
                map_opt = {
                    "Auto (mais recente)": "auto",
                    "Forçar CLOUD": "cloud",
                    "Forçar LOCAL": "local",
                }
                meta = sync.process_latest_snapshot(
                    prefer_buy=map_opt[prefer_buy],
                    prefer_sell=map_opt[prefer_sell],
                )
                st.success(f"OK. Records: {meta['processed']['records']}")
                ts_ref = meta['processed']['ref_snapshot_ts']
                st.caption(f"Ref ts: {_fmt_ts(ts_ref, show_brt) if ts_ref else '—'}")
                st.caption(f"BUY usado ({meta['buy']['mode']}): {meta['buy']['chosen_path']}")
                st.caption(f"SELL usado ({meta['sell']['mode']}): {meta['sell']['chosen_path']}")
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

    st.divider()
    st.markdown("**Arquivos presentes em `raw/collected/`**")
    files = sorted(sync.COLLECTED_DIR.glob("*.json"))
    if files:
        st.code("\n".join(f"- {f.name}" for f in files), language="text")
    else:
        st.caption("Nenhum arquivo encontrado.")
