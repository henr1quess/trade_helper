# nwmp_sync.py
# --------------------------------------------------------------------------------------
# 1) Download cloud -> raw/collected/
# 2) Copy local (BUY arquivo, SELL pasta) -> raw/collected/ + ENRICH com item_name via raw/item_name_map.json
# 3) Process -> raw/latest_snapshot.json (top_buy / low_sell)
#    prefer_buy/prefer_sell: 'auto' | 'cloud' | 'local'
# --------------------------------------------------------------------------------------

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import requests
import pandas as pd

# --- paths ---
PROJECT_DIR = Path.cwd()
RAW_DIR = PROJECT_DIR / "raw"
COLLECTED_DIR = RAW_DIR / "collected"
RAW_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

LAST_META_PATH = RAW_DIR / "last_sync_meta.json"
PROCESSED_SNAPSHOT_PATH = RAW_DIR / "latest_snapshot.json"

# mapa de nomes
ITEM_NAME_MAP_PATH = RAW_DIR / "item_name_map.json"

# arquivos coletados
CLOUD_BUY_PATH = COLLECTED_DIR / "cloud_buy_devaloka.json"
CLOUD_SELL_PATH = COLLECTED_DIR / "cloud_sell_devaloka.json"
LOCAL_BUY_PATH = COLLECTED_DIR / "local_buy_devaloka.json"
LOCAL_SELL_PATH = COLLECTED_DIR / "local_sell_devaloka.json"

# endpoints cloud
GT_BASE = "https://nwmpdata.gaming.tools"
BUY_URL = f"{GT_BASE}/buy-orders2/devaloka.json"
SELL_URL = f"{GT_BASE}/auctions2/devaloka.json"

# fontes locais padrão
LOCAL_SOURCES_CANDIDATES = [
    PROJECT_DIR / "local_sources.json",
    Path.home() / ".nw_local_sources.json",
]

# --- utils ---
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _to_iso_utc(ts_val) -> Optional[str]:
    # aceita ISO ou epoch (int/float/str)
    if ts_val is None:
        return None
    if isinstance(ts_val, str) and "T" in ts_val:
        return ts_val
    try:
        if isinstance(ts_val, str):
            ts_val = float(ts_val)
        if isinstance(ts_val, (int, float)):
            return datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return None

def _extract_any_ts(entry: dict) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    for k in ("timestamp", "created_at", "expires_at"):
        if k in entry:
            iso = _to_iso_utc(entry[k])
            if iso:
                return iso
    return None

def _max_snapshot_ts(arr: List[Dict[str, Any]]) -> Optional[str]:
    best_iso: Optional[str] = None
    best_dt: Optional[datetime] = None
    for el in arr or []:
        ts_iso = _extract_any_ts(el)
        dt = _parse_ts(ts_iso) if ts_iso else None
        if dt and (best_dt is None or dt > best_dt):
            best_dt, best_iso = dt, ts_iso
    return best_iso

def _choose_by_newest(candidates: List[Tuple[str, Path]]) -> Tuple[Optional[Path], Optional[str], int]:
    winner_path: Optional[Path] = None
    winner_ts_raw: Optional[str] = None
    winner_dt: Optional[datetime] = None
    winner_len: int = 0
    for _, p in candidates:
        if not p.exists():
            continue
        try:
            arr = _read_json(p)
            if not isinstance(arr, list):
                continue
            ts_raw = _max_snapshot_ts(arr)
            ts_dt = _parse_ts(ts_raw) if ts_raw else None
            n = len(arr)
            if not ts_dt:
                ts_dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if (winner_dt is None) or (ts_dt and ts_dt > winner_dt):
                winner_dt = ts_dt
                winner_ts_raw = ts_raw
                winner_path = p
                winner_len = n
        except Exception:
            continue
    return winner_path, winner_ts_raw, winner_len

def _load_json_array(p: Path) -> List[Dict[str, Any]]:
    try:
        data = _read_json(p)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _load_from_file_or_folder(src: Optional[str]) -> List[Dict[str, Any]]:
    if not src:
        return []
    p = Path(src)
    if not p.exists():
        return []
    if p.is_dir():
        rows: List[Dict[str, Any]] = []
        for f in sorted(p.glob("*.json")):
            rows.extend(_load_json_array(f))
        return rows
    else:
        return _load_json_array(p)

# --- item_name mapping ---
def _load_item_name_map(path: Path = ITEM_NAME_MAP_PATH) -> Dict[str, str]:
    """
    Lê raw/item_name_map.json e normaliza as CHAVES para minúsculas.
    Aceita:
      - dict: { "perkcharm_healing": "Anointed Lifesteal II Charm", ... }
      - lista de objetos com (item_id/item_name) ou (id/name)
    """
    try:
        if not path.exists():
            return {}
        data = _read_json(path)
        if isinstance(data, dict):
            return {str(k).strip().lower(): str(v) for k, v in data.items() if k and v}
        if isinstance(data, list):
            out: Dict[str, str] = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                iid = row.get("item_id") or row.get("id")
                nm  = row.get("item_name") or row.get("name")
                if iid and nm:
                    out[str(iid).strip().lower()] = str(nm)
            return out
    except Exception:
        pass
    return {}

def _enrich_with_names(rows: List[Dict[str, Any]], name_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Adiciona item_name usando o mapa quando estiver ausente."""
    if not rows or not name_map:
        return rows
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            if not r.get("item_name"):
                iid = r.get("item_id")
                if iid:
                    key = str(iid).strip().lower()
                    nm = name_map.get(key)
                    if nm:
                        r = {**r, "item_name": nm}
        out.append(r)
    return out

# --- config local_sources ---
def _load_local_sources():
    for p in LOCAL_SOURCES_CANDIDATES:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                buy_file = cfg.get("buy_file") or cfg.get("buy") or None
                sell_dir = cfg.get("sell_dir") or cfg.get("sell") or None
                if isinstance(buy_file, str):
                    buy_file = str(Path(buy_file).expanduser())
                if isinstance(sell_dir, str):
                    sell_dir = str(Path(sell_dir).expanduser())
                return p, buy_file, sell_dir
            except Exception:
                continue
    return None, None, None

def get_local_source_config() -> Dict[str, Any]:
    cfg_path, buy_src, sell_src = _load_local_sources()
    return {
        "config_path": str(cfg_path) if cfg_path else None,
        "buy_file": buy_src,
        "sell_dir": sell_src,
    }

# --- 1) cloud download ---
def download_cloud_snapshots(timeout: int = 60) -> Dict[str, Any]:
    r_buy = requests.get(BUY_URL, timeout=timeout)
    r_buy.raise_for_status()
    buy_arr = r_buy.json()
    _write_json(CLOUD_BUY_PATH, buy_arr)

    r_sell = requests.get(SELL_URL, timeout=timeout)
    r_sell.raise_for_status()
    sell_arr = r_sell.json()
    _write_json(CLOUD_SELL_PATH, sell_arr)

    meta = {
        "action": "download_cloud_snapshots",
        "when": _now_utc_iso(),
        "saved": {
            "cloud_buy_path": str(CLOUD_BUY_PATH),
            "cloud_sell_path": str(CLOUD_SELL_PATH),
            "cloud_buy_entries": _safe_len(buy_arr),
            "cloud_sell_entries": _safe_len(sell_arr),
            "cloud_buy_snapshot_ts": _max_snapshot_ts(buy_arr),
            "cloud_sell_snapshot_ts": _max_snapshot_ts(sell_arr),
        },
    }
    _merge_and_write_meta(meta)
    return meta

# --- 2) copy local (and ENRICH with names) ---
def copy_local_snapshots(src_buy: Optional[str] = None, src_sell: Optional[str] = None) -> Dict[str, Any]:
    saved: Dict[str, Any] = {}
    name_map = _load_item_name_map(ITEM_NAME_MAP_PATH)

    if src_buy:
        buy_rows = _load_from_file_or_folder(src_buy)
        buy_rows = _enrich_with_names(buy_rows, name_map)
        _write_json(LOCAL_BUY_PATH, buy_rows)
        saved.update({
            "local_buy_path": str(LOCAL_BUY_PATH),
            "local_buy_entries": len(buy_rows),
            "local_buy_snapshot_ts": _max_snapshot_ts(buy_rows),
            "local_buy_source": src_buy,
        })

    if src_sell:
        sell_rows = _load_from_file_or_folder(src_sell)
        sell_rows = _enrich_with_names(sell_rows, name_map)
        _write_json(LOCAL_SELL_PATH, sell_rows)
        saved.update({
            "local_sell_path": str(LOCAL_SELL_PATH),
            "local_sell_entries": len(sell_rows),
            "local_sell_snapshot_ts": _max_snapshot_ts(sell_rows),
            "local_sell_source": src_sell,
        })

    meta = {"action": "copy_local_snapshots", "when": _now_utc_iso(), "saved": saved}
    _merge_and_write_meta(meta)
    return meta

def copy_local_from_defaults() -> Dict[str, Any]:
    _, buy_file, sell_dir = _load_local_sources()
    if not buy_file and not sell_dir:
        raise RuntimeError(
            "Caminhos padrão não configurados. Crie local_sources.json com "
            "{\"buy_file\": \"C:/.../buy-orders/devaloka.json\", \"sell_dir\": \"C:/.../auctions\"}"
        )
    return copy_local_snapshots(buy_file, sell_dir)

# --- 3) process ---
def _names_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if {"item_id", "item_name"}.issubset(cols):
        out = df.dropna(subset=["item_id"]).copy()
        out = out.loc[:, ["item_id", "item_name"]]
        return out.drop_duplicates("item_id")
    return pd.DataFrame(columns=["item_id", "item_name"])

def _group_buy(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"item_id", "price", "quantity"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=["item_id","top_buy","buy_qty"])
    x = df.copy()
    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["quantity"] = pd.to_numeric(x["quantity"], errors="coerce").fillna(0)
    return x.groupby("item_id", as_index=False).agg(top_buy=("price", "max"),
                                                    buy_qty=("quantity", "sum"))

def _group_sell(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"item_id", "price", "quantity"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=["item_id","low_sell","sell_qty"])
    x = df.copy()
    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["quantity"] = pd.to_numeric(x["quantity"], errors="coerce").fillna(0)
    return x.groupby("item_id", as_index=False).agg(low_sell=("price", "min"),
                                                    sell_qty=("quantity", "sum"))

def process_latest_snapshot(prefer_buy: str = "auto", prefer_sell: str = "auto") -> Dict[str, Any]:
    # BUY select
    if prefer_buy == "cloud":
        buy_path = CLOUD_BUY_PATH if CLOUD_BUY_PATH.exists() else None
        buy_arr = _read_json(buy_path) if buy_path else []
        buy_ts = _max_snapshot_ts(buy_arr); buy_count = _safe_len(buy_arr)
    elif prefer_buy == "local":
        buy_path = LOCAL_BUY_PATH if LOCAL_BUY_PATH.exists() else None
        buy_arr = _read_json(buy_path) if buy_path else []
        buy_ts = _max_snapshot_ts(buy_arr); buy_count = _safe_len(buy_arr)
    else:
        buy_path, buy_ts, buy_count = _choose_by_newest([("cloud_buy", CLOUD_BUY_PATH), ("local_buy", LOCAL_BUY_PATH)])
        buy_arr = _read_json(buy_path) if buy_path else []

    # SELL select
    if prefer_sell == "cloud":
        sell_path = CLOUD_SELL_PATH if CLOUD_SELL_PATH.exists() else None
        sell_arr = _read_json(sell_path) if sell_path else []
        sell_ts = _max_snapshot_ts(sell_arr); sell_count = _safe_len(sell_arr)
    elif prefer_sell == "local":
        sell_path = LOCAL_SELL_PATH if LOCAL_SELL_PATH.exists() else None
        sell_arr = _read_json(sell_path) if sell_path else []
        sell_ts = _max_snapshot_ts(sell_arr); sell_count = _safe_len(sell_arr)
    else:
        sell_path, sell_ts, sell_count = _choose_by_newest([("cloud_sell", CLOUD_SELL_PATH), ("local_sell", LOCAL_SELL_PATH)])
        sell_arr = _read_json(sell_path) if sell_path else []

    buy_df  = pd.DataFrame(buy_arr)  if buy_arr  else pd.DataFrame()
    sell_df = pd.DataFrame(sell_arr) if sell_arr else pd.DataFrame()

    top_buy  = _group_buy(buy_df)
    low_sell = _group_sell(sell_df)
    name_buy  = _names_df(buy_df)
    name_sell = _names_df(sell_df)

    merged = top_buy.merge(low_sell, on="item_id", how="outer")
    if not name_buy.empty or not name_sell.empty:
        names = pd.concat([name_sell, name_buy], ignore_index=True).drop_duplicates("item_id")
    else:
        names = pd.DataFrame(columns=["item_id","item_name"])
    merged = merged.merge(names, on="item_id", how="left")

    # fallback de nomes via mapa também no processamento
    item_name_map = _load_item_name_map(ITEM_NAME_MAP_PATH)
    if not merged.empty:
        if "item_name" not in merged.columns:
            merged["item_name"] = None
        merged["iid_norm"] = merged["item_id"].astype(str).str.strip().str.lower()
        merged["item_name"] = merged["item_name"].where(
            merged["item_name"].notna(),
            merged["iid_norm"].map(item_name_map)
        )
        merged = merged.drop(columns=["iid_norm"])

    # timestamp de referência
    ref_ts = buy_ts or sell_ts
    bdt = _parse_ts(buy_ts) if buy_ts else None
    sdt = _parse_ts(sell_ts) if sell_ts else None
    if bdt and sdt:
        ref_ts = buy_ts if bdt >= sdt else sell_ts

    records: List[Dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        records.append({
            "timestamp": ref_ts,
            "item_id": row.item_id,
            "item_name": getattr(row, "item_name", None),
            "top_buy":  float(row.top_buy)  if hasattr(row, "top_buy")  and pd.notna(row.top_buy)  else None,
            "low_sell": float(row.low_sell) if hasattr(row, "low_sell") and pd.notna(row.low_sell) else None,
            "buy_qty":  int(row.buy_qty)    if hasattr(row, "buy_qty")  and pd.notna(row.buy_qty)  else 0,
            "sell_qty": int(row.sell_qty)   if hasattr(row, "sell_qty") and pd.notna(row.sell_qty) else 0,
        })

    snapshot = {
        "server": "devaloka",
        "processed_at": _now_utc_iso(),
        "snapshot_ts": ref_ts,
        "buy_source_path": str(buy_path) if buy_path else None,
        "sell_source_path": str(sell_path) if sell_path else None,
        "records": records,
    }
    _write_json(PROCESSED_SNAPSHOT_PATH, snapshot)

    meta = {
        "action": "process_latest_snapshot",
        "when": _now_utc_iso(),
        "buy":  {"chosen_path": str(buy_path) if buy_path else None,  "entries": buy_count,  "snapshot_ts": buy_ts,  "mode": prefer_buy},
        "sell": {"chosen_path": str(sell_path) if sell_path else None, "entries": sell_count, "snapshot_ts": sell_ts, "mode": prefer_sell},
        "processed": {"records": len(records), "output_path": str(PROCESSED_SNAPSHOT_PATH), "ref_snapshot_ts": ref_ts},
    }
    _merge_and_write_meta(meta)
    return meta

# --- meta ---
def _merge_and_write_meta(update: Dict[str, Any]) -> None:
    try:
        current = _read_json(LAST_META_PATH)
        if not isinstance(current, dict):
            current = {}
    except Exception:
        current = {}

    history = current.get("history", [])
    history.append(update)

    current.update({
        "updated_at": _now_utc_iso(),
        "last_action": update.get("action"),
        "history": history[-50:],
    })
    if "saved" in update:
        current["last_saved"] = update["saved"]
    if "processed" in update:
        current["last_processed"] = update["processed"]
    if "buy" in update or "sell" in update:
        current["last_sources"] = {"buy": update.get("buy"), "sell": update.get("sell")}

    _write_json(LAST_META_PATH, current)

# --- CLI opcional ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="nwmp_sync", description="Coleta e processa snapshots (cloud/local).")
    ap.add_argument("--download-cloud", action="store_true")
    ap.add_argument("--copy-local-buy", type=str, default=None)
    ap.add_argument("--copy-local-sell", type=str, default=None, help="Arquivo ou diretório (SELL)")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--prefer-buy", choices=["auto", "cloud", "local"], default="auto")
    ap.add_argument("--prefer-sell", choices=["auto", "cloud", "local"], default="auto")
    args = ap.parse_args()

    if args.download_cloud:
        print(json.dumps(download_cloud_snapshots(), ensure_ascii=False, indent=2))
    if args.copy_local_buy or args.copy_local_sell:
        print(json.dumps(copy_local_snapshots(args.copy_local_buy, args.copy_local_sell), ensure_ascii=False, indent=2))
    if args.process:
        print(json.dumps(process_latest_snapshot(args.prefer_buy, args.prefer_sell), ensure_ascii=False, indent=2))
