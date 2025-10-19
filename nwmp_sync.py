# nwmp_sync.py
from __future__ import annotations
import json, gzip, os
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd

# ------------------ Utils (I/O & time) ------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_json_bytes(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))

def _read_bytes_local_or_url(path_or_url: str, timeout: int = 60) -> bytes:
    p = (path_or_url or "").strip()
    if not p:
        raise ValueError("path_or_url vazio")
    if p.startswith("http://") or p.startswith("https://"):
        try:
            import requests  # opcional
        except Exception as exc:
            raise RuntimeError("requests não disponível para baixar URLs") from exc
        resp = requests.get(p, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Falha ao baixar: {p} ({resp.status_code})")
        return resp.content
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    return path.read_bytes()

def _maybe_gunzip(raw: bytes, hint: Optional[str] = None) -> bytes:
    name = (hint or "").lower()
    gz_ext = name.endswith(".gz")
    gz_magic = raw[:2] == b"\x1f\x8b"
    if gz_ext or gz_magic:
        try:
            return gzip.decompress(raw)
        except OSError:
            return raw
    return raw

def _normalise_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # heurística: >=1e12 assume ms
        div = 1000.0 if float(value) >= 1_000_000_000_000 else 1.0
        try:
            return datetime.fromtimestamp(float(value)/div, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts): return None
            return ts.to_pydatetime()
        except Exception:
            return None
    return None

def _ts_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def _detect_snapshot_ts(entries: List[Dict[str, Any]]) -> str:
    tss: List[datetime] = []
    for e in entries:
        ts = _normalise_timestamp(e.get("timestamp"))
        if ts is not None:
            tss.append(ts)
    base = min(tss).astimezone(timezone.utc) if tss else datetime.now(timezone.utc)
    return _ts_iso_z(base)

def _ts_for_filename(ts_iso_z: str) -> str:
    # 2025-10-19T15:21:10Z -> 20251019T152110Z
    return ts_iso_z.replace("-", "").replace(":", "")

# ------------------ Load snapshot arrays ------------------

def load_snapshot_array(path_or_url: str, timeout: int = 60) -> List[Dict[str, Any]]:
    raw = _read_bytes_local_or_url(path_or_url, timeout=timeout)
    raw = _maybe_gunzip(raw, hint=path_or_url)
    data = _read_json_bytes(raw)
    if not isinstance(data, list):
        raise TypeError("Snapshot deve ser array JSON")
    return [x for x in data if isinstance(x, dict)]

# ------------------ RAW save ------------------

def _dump_json_gz(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(raw)

def _append_ndjson(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

def save_raw_snapshot_array(
    path_or_url: str, out_root_raw: str, label: str, timeout: int = 60
) -> Tuple[List[Dict[str, Any]], str, Path]:
    entries = load_snapshot_array(path_or_url, timeout=timeout)
    snapshot_ts = _detect_snapshot_ts(entries)
    dt = _normalise_timestamp(snapshot_ts)
    assert dt is not None
    y, m, d = dt.year, dt.month, dt.day
    out_dir = Path(out_root_raw) / "snapshots" / f"{y:04d}" / f"{m:02d}" / f"{d:02d}"
    fname = f"{_ts_for_filename(snapshot_ts)}_{label}.json.gz"
    out_path = out_dir / fname
    _dump_json_gz(out_path, entries)

    ndjson_rows = []
    for e in entries:
        r = dict(e)
        r["side"] = "buy" if label == "buy" else "sell"
        r["snapshot_ts"] = snapshot_ts
        ndjson_rows.append(r)
    _append_ndjson(Path(out_root_raw) / "combined.ndjson", ndjson_rows)

    return entries, snapshot_ts, out_path

# ------------------ Transform to historical rows ------------------

def _price_cents_to_float(v: Any) -> Optional[float]:
    try:
        iv = int(v)
    except Exception:
        return None
    return round(iv/100.0, 2)

def _group_key(entry: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    ts = _normalise_timestamp(entry.get("timestamp"))
    if ts is None:
        return None
    tsz = _ts_iso_z(ts)
    server_id = (entry.get("server_id") or entry.get("server") or "devaloka").lower()
    item_id = (entry.get("item_id") or entry.get("slug") or entry.get("id") or "").strip()
    if not item_id:
        return None
    return (tsz, server_id, item_id)

def extract_records_from_snapshots(
    buy_orders: List[Dict[str, Any]],
    auctions: List[Dict[str, Any]],
    server: str = "devaloka",
) -> List[Dict[str, Any]]:
    server_norm = (server or "devaloka").lower()
    buy_map: Dict[Tuple[str,str,str], Dict[str, Any]] = {}
    for e in buy_orders:
        k = _group_key(e)
        if not k: continue
        ts, srv, item = k
        if server_norm and srv != server_norm: continue
        price = _price_cents_to_float(e.get("price"))
        qty = e.get("quantity")
        name = e.get("item_name")
        rec = buy_map.setdefault(k, {"max_price": None, "qty_sum": 0, "item_name": name})
        if price is not None:
            rec["max_price"] = price if rec["max_price"] is None else max(rec["max_price"], price)
        if isinstance(qty, int):
            rec["qty_sum"] += qty
        if not rec.get("item_name") and name:
            rec["item_name"] = name

    sell_map: Dict[Tuple[str,str,str], Dict[str, Any]] = {}
    for e in auctions:
        k = _group_key(e)
        if not k: continue
        ts, srv, item = k
        if server_norm and srv != server_norm: continue
        price = _price_cents_to_float(e.get("price"))
        qty = e.get("quantity")
        name = e.get("item_name")
        rec = sell_map.setdefault(k, {"min_price": None, "qty_sum": 0, "item_name": name})
        if price is not None:
            rec["min_price"] = price if rec["min_price"] is None else min(rec["min_price"], price)
        if isinstance(qty, int):
            rec["qty_sum"] += qty
        if not rec.get("item_name") and name:
            rec["item_name"] = name

    keys = set(buy_map.keys()) | set(sell_map.keys())
    rows: List[Dict[str, Any]] = []
    for ts, srv, item in keys:
        buy = buy_map.get((ts, srv, item), {})
        sell = sell_map.get((ts, srv, item), {})
        item_name = sell.get("item_name") or buy.get("item_name") or item
        rows.append({
            "timestamp": ts,
            "item": item_name,
            "slug": item,
            "server": srv,
            "top_buy": buy.get("max_price"),
            "low_sell": sell.get("min_price"),
            "avg_price": None,
            "volume": sell.get("qty_sum") if isinstance(sell.get("qty_sum"), int) else None,
            "source": "nwmp_snapshot",
        })
    return rows

# ------------------ CSV merge (historical) ------------------

def update_csv(records: List[Dict[str, Any]], csv_path: str) -> None:
    if not records:
        return
    df_new = pd.DataFrame(records)
    if df_new.empty: return
    exp_cols = ["timestamp","item","slug","server","top_buy","low_sell","avg_price","volume","source"]
    for c in exp_cols:
        if c not in df_new.columns:
            df_new[c] = None
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True, errors="coerce")
    df_new = df_new.dropna(subset=["timestamp"])
    for c in ["top_buy","low_sell","avg_price"]:
        df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
    if "volume" in df_new.columns:
        df_new["volume"] = pd.to_numeric(df_new["volume"], errors="coerce").astype("Int64")
    df_new["timestamp"] = df_new["timestamp"].dt.tz_convert(timezone.utc)

    dest = Path(csv_path); _ensure_dir(dest.parent)
    if dest.exists():
        try:
            df_old = pd.read_csv(dest)
        except Exception:
            df_old = pd.DataFrame(columns=exp_cols)
    else:
        df_old = pd.DataFrame(columns=exp_cols)

    if not df_old.empty:
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], utc=True, errors="coerce")

    combined = pd.concat([df_old, df_new], ignore_index=True)
    subset = [c for c in ["timestamp","item","server","slug"] if c in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp")
        combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    combined.to_csv(dest, index=False)

# ------------------ History.json projection (for the current app) ------------------

def append_history_json_from_records(records: List[Dict[str, Any]], history_json_path: str) -> None:
    """
    Projeta para o formato atual do app:
    {"timestamp","item","buy_market","sell_market"}
    """
    if not records: return
    proj = []
    for r in records:
        if r.get("item") and r.get("top_buy") is not None and r.get("low_sell") is not None:
            proj.append({
                "timestamp": r["timestamp"],
                "item": r["item"],
                "buy_market": r["top_buy"],
                "sell_market": r["low_sell"],
            })
    if not proj: return
    # append estilo JSON lines (leremos no app via pandas depois)
    p = Path(history_json_path); _ensure_dir(p.parent)
    try:
        # carrega existente (lista) ou cria
        if p.exists():
            try:
                df_old = pd.read_json(p, orient="records")
            except Exception:
                df_old = pd.DataFrame(columns=["timestamp","item","buy_market","sell_market"])
        else:
            df_old = pd.DataFrame(columns=["timestamp","item","buy_market","sell_market"])
        df_new = pd.DataFrame(proj)
        # dedupe por (timestamp,item)
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True, errors="coerce")
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], utc=True, errors="coerce")
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.dropna(subset=["timestamp","item"])
        df = df.drop_duplicates(subset=["timestamp","item"], keep="last")
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.to_json(p, orient="records", indent=2)
    except Exception:
        # fallback: não quebrar pipeline caso history.json esteja corrompido
        pass

# ------------------ Orchestration ------------------

def sync_sources_save_raw_and_update_csv(
    buy_orders_path_or_url: str,
    auctions_path_or_url: str,
    raw_root: str,
    csv_path: str,
    history_json_path: str,
    server: str = "devaloka",
    timeout: int = 60,
) -> None:
    buy_entries, _, _ = save_raw_snapshot_array(buy_orders_path_or_url, raw_root, label="buy", timeout=timeout)
    sell_entries, _, _ = save_raw_snapshot_array(auctions_path_or_url, raw_root, label="sell", timeout=timeout)
    records = extract_records_from_snapshots(buy_entries, sell_entries, server=server)
    update_csv(records, csv_path)
    append_history_json_from_records(records, history_json_path)

def rebuild_csv_from_raw(raw_root: str, csv_path: str, server: str = "devaloka") -> None:
    root = Path(raw_root) / "snapshots"
    if not root.exists():
        raise FileNotFoundError(f"Nada em {root}")
    buy_files = sorted(root.rglob("*_buy.json.gz"))
    sell_files = sorted(root.rglob("*_sell.json.gz"))

    def _key(p: Path) -> Optional[str]:
        n = p.name
        if n.endswith("_buy.json.gz"): return n.replace("_buy.json.gz","")
        if n.endswith("_sell.json.gz"): return n.replace("_sell.json.gz","")
        return None

    buy_map = { _key(p): p for p in buy_files }
    sell_map = { _key(p): p for p in sell_files }
    keys = sorted(set(k for k in buy_map.keys() if k) | set(k for k in sell_map.keys() if k))
    all_rows: List[Dict[str, Any]] = []
    for k in keys:
        be = []
        se = []
        if k in buy_map:
            with gzip.open(buy_map[k], "rb") as f:
                be = _read_json_bytes(f.read())
        if k in sell_map:
            with gzip.open(sell_map[k], "rb") as f:
                se = _read_json_bytes(f.read())
        rows = extract_records_from_snapshots(be or [], se or [], server=server)
        if rows:
            all_rows.extend(rows)
    update_csv(all_rows, csv_path)

# Conveniências simples para o app
def run_sync(buy_url_or_path: str, sell_url_or_path: str, raw_root: str="raw",
             csv_path: str="data/history_devaloka.csv", history_json_path: str="history.json",
             server: str="devaloka", timeout: int=60) -> None:
    sync_sources_save_raw_and_update_csv(
        buy_url_or_path, sell_url_or_path, raw_root, csv_path, history_json_path, server, timeout
    )

def run_rebuild(raw_root: str="raw", csv_path: str="data/history_devaloka.csv", server: str="devaloka") -> None:
    rebuild_csv_from_raw(raw_root, csv_path, server=server)
