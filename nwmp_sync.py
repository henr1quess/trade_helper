# nwmp_sync.py
from __future__ import annotations
# Requisitos: pandas (obrigatório), numpy (para estatísticas), requests (apenas para URLs)
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BUY_HISTORY_COLUMNS = [
    "snapshot_ts",
    "item_id",
    "item_name",
    "min_price",
    "median_price",
    "max_price",
    "total_quantity",
    "order_count",
]

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


def _ensure_iso_timestamp(value: Any, fallback: Optional[datetime] = None) -> str:
    dt = _normalise_timestamp(value)
    if dt is None:
        dt = fallback or datetime.now(timezone.utc)
    return _ts_iso_z(dt)


def _maybe_iso_datetime_field(entry: Dict[str, Any], key: str, fallback: Optional[datetime] = None) -> None:
    if key not in entry:
        return
    entry[key] = _ensure_iso_timestamp(entry.get(key), fallback=fallback)

# ------------------ Load snapshot arrays ------------------

def load_snapshot_array(path_or_url: str, timeout: int = 60) -> List[Dict[str, Any]]:
    raw = _read_bytes_local_or_url(path_or_url, timeout=timeout)
    raw = _maybe_gunzip(raw, hint=path_or_url)
    data = _read_json_bytes(raw)
    if not isinstance(data, list):
        raise TypeError("Snapshot deve ser array JSON")
    return [x for x in data if isinstance(x, dict)]

# ------------------ RAW save ------------------

def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []

def _write_json_list(path: Path, entries: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_object(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _update_last_sync_metadata(
    meta_path: Path,
    source: str,
    payload: Dict[str, Any],
    snapshot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not source:
        return {}

    meta = _load_json_object(meta_path)
    existing = meta.get(source)
    if not isinstance(existing, dict):
        existing = {}

    payload_copy = dict(existing)
    payload_copy.update(payload)

    if snapshot_path is not None:
        payload_copy["snapshot_path"] = str(snapshot_path)
    else:
        snapshot_value = payload_copy.get("snapshot_path")
        if isinstance(snapshot_value, Path):
            payload_copy["snapshot_path"] = str(snapshot_value)

    meta[source] = payload_copy
    meta["_last_updated"] = _ts_iso_z(datetime.now(timezone.utc))
    _write_json_object(meta_path, meta)
    return payload_copy

def _merge_history_entries(
    existing: List[Dict[str, Any]], new_entries: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []

    def _key(entry: Dict[str, Any]) -> str:
        return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    for e in existing:
        k = _key(e)
        if k in seen:
            continue
        seen.add(k)
        merged.append(e)

    for e in new_entries:
        k = _key(e)
        if k in seen:
            continue
        seen.add(k)
        merged.append(e)

    return merged


def _load_local_snapshot_entries(directory: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not directory.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")

    for json_path in sorted(directory.rglob("*.json")):
        if not json_path.is_file():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            for raw in data:
                if isinstance(raw, dict):
                    entries.append(raw.copy())
    return entries


def _normalise_local_entry(
    entry: Dict[str, Any],
    snapshot_dt: datetime,
    server: str,
) -> Dict[str, Any]:
    norm: Dict[str, Any] = dict(entry)

    norm["timestamp"] = _ensure_iso_timestamp(norm.get("timestamp"), fallback=snapshot_dt)
    _maybe_iso_datetime_field(norm, "created_at", fallback=snapshot_dt)
    _maybe_iso_datetime_field(norm, "expires_at", fallback=snapshot_dt)
    norm["snapshot_ts"] = _ts_iso_z(snapshot_dt)

    server_norm = (server or "devaloka").lower()
    norm["server_id"] = server_norm
    if not norm.get("server"):
        norm["server"] = server_norm

    slug = str(norm.get("item_id") or norm.get("slug") or norm.get("id") or "").strip()
    if slug:
        norm.setdefault("slug", slug)
    if not norm.get("item_name"):
        norm["item_name"] = slug or norm.get("item") or ""

    for key in ["price", "quantity"]:
        if key in norm:
            try:
                norm[key] = int(norm[key])
            except Exception:
                try:
                    norm[key] = int(float(norm[key]))
                except Exception:
                    pass

    return norm


def _append_raw_entries(raw_path: Path, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    existing = _load_json_list(raw_path)
    merged = _merge_history_entries(existing, entries)
    _write_json_list(raw_path, merged)


def _prepare_buy_entries_dataframe(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(columns=["item_id", "item_name", "price", "quantity"])

    df = pd.DataFrame(entries)
    if "item_id" not in df.columns:
        df["item_id"] = ""
    if "item_name" not in df.columns:
        df["item_name"] = ""

    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce")
    df["quantity"] = df["quantity"].fillna(0)

    df = df.dropna(subset=["price"])
    return df


def _build_buy_snapshot_summary(
    entries: List[Dict[str, Any]], snapshot_iso: str
) -> pd.DataFrame:
    df = _prepare_buy_entries_dataframe(entries)
    if df.empty:
        return pd.DataFrame(columns=BUY_HISTORY_COLUMNS)

    grouped = df.groupby("item_id", dropna=False)
    summary = grouped.agg(
        item_name=("item_name", "first"),
        min_price=("price", "min"),
        median_price=("price", "median"),
        max_price=("price", "max"),
        total_quantity=("quantity", "sum"),
        order_count=("price", "count"),
    ).reset_index()

    summary.insert(0, "snapshot_ts", snapshot_iso)

    summary["total_quantity"] = (
        pd.to_numeric(summary["total_quantity"], errors="coerce").fillna(0).astype(int)
    )
    summary["order_count"] = (
        pd.to_numeric(summary["order_count"], errors="coerce").fillna(0).astype(int)
    )

    numeric_cols = ["min_price", "median_price", "max_price"]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary[BUY_HISTORY_COLUMNS]


def _load_buy_history_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BUY_HISTORY_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=BUY_HISTORY_COLUMNS)

    missing = [col for col in BUY_HISTORY_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return df[BUY_HISTORY_COLUMNS]


def _write_buy_history_csv(path: Path, df: pd.DataFrame) -> None:
    _ensure_dir(path.parent)
    df = df.copy()
    df.sort_values(by=["snapshot_ts", "item_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)


def _update_buy_history_csv(
    path: Path, summary: pd.DataFrame, *, replace: bool = False
) -> pd.DataFrame:
    if summary.empty and not replace:
        existing = _load_buy_history_csv(path)
        if not path.exists():
            _write_buy_history_csv(path, existing)
        return existing

    if replace:
        combined = summary.copy()
    else:
        existing = _load_buy_history_csv(path)
        combined = pd.concat([existing, summary], ignore_index=True)

    combined = combined.drop_duplicates(subset=["snapshot_ts", "item_id"], keep="last")
    _write_buy_history_csv(path, combined)
    return combined


# ------------------ Snapshot probes ------------------

def _summarise_snapshot_entries(
    entries: List[Dict[str, Any]], *, prefer: str = "min"
) -> Optional[datetime]:
    if not entries:
        return None
    timestamps: List[datetime] = []
    for entry in entries:
        ts = _normalise_timestamp(entry.get("timestamp"))
        if ts is not None:
            timestamps.append(ts)
    if not timestamps:
        return None
    if prefer == "max":
        return max(timestamps)
    return min(timestamps)


def probe_remote_snapshot(
    buy_orders_url: str,
    sell_orders_url: Optional[str] = None,
    *,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Collect lightweight metadata about the remote snapshots without persisting."""

    result: Dict[str, Any] = {
        "source": "remote",
        "snapshot_ts": None,
        "buy_entries": None,
        "buy_snapshot_ts": None,
        "sell_entries": None,
        "sell_snapshot_ts": None,
        "error": None,
    }

    errors: List[str] = []
    buy_dt: Optional[datetime] = None
    sell_dt: Optional[datetime] = None

    if not (buy_orders_url or "").strip():
        errors.append("buy: URL não configurada")
    else:
        try:
            buy_entries = load_snapshot_array(buy_orders_url, timeout=timeout)
            result["buy_entries"] = len(buy_entries)
            buy_dt = _summarise_snapshot_entries(buy_entries, prefer="min")
            if buy_dt is not None:
                result["buy_snapshot_ts"] = _ts_iso_z(buy_dt)
        except Exception as exc:  # pragma: no cover - falhas de rede/IO
            errors.append(f"buy: {exc}")

    if sell_orders_url and sell_orders_url.strip():
        try:
            sell_entries = load_snapshot_array(sell_orders_url, timeout=timeout)
            result["sell_entries"] = len(sell_entries)
            sell_dt = _summarise_snapshot_entries(sell_entries, prefer="min")
            if sell_dt is not None:
                result["sell_snapshot_ts"] = _ts_iso_z(sell_dt)
        except Exception as exc:  # pragma: no cover - falhas de rede/IO
            errors.append(f"sell: {exc}")

    ts_candidates = [dt for dt in [buy_dt, sell_dt] if dt is not None]
    if ts_candidates:
        latest = max(ts_candidates)
        result["snapshot_ts"] = _ts_iso_z(latest)

    if errors:
        result["error"] = "; ".join(errors)

    return result


def probe_local_snapshot(
    buy_orders_dir: str,
    auctions_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect metadata from local snapshot folders without mutating RAW."""

    result: Dict[str, Any] = {
        "source": "local",
        "snapshot_ts": None,
        "buy_entries": None,
        "buy_snapshot_ts": None,
        "sell_entries": None,
        "sell_snapshot_ts": None,
        "error": None,
    }

    errors: List[str] = []
    buy_dt: Optional[datetime] = None
    sell_dt: Optional[datetime] = None

    try:
        buy_entries = _load_local_snapshot_entries(Path(buy_orders_dir))
        result["buy_entries"] = len(buy_entries)
        buy_dt = _summarise_snapshot_entries(buy_entries, prefer="max")
        if buy_dt is not None:
            result["buy_snapshot_ts"] = _ts_iso_z(buy_dt)
    except FileNotFoundError as exc:
        errors.append(f"buy: {exc}")
    except Exception as exc:  # pragma: no cover - leitura local
        errors.append(f"buy: {exc}")

    if auctions_dir and str(auctions_dir).strip():
        try:
            sell_entries = _load_local_snapshot_entries(Path(auctions_dir))
            result["sell_entries"] = len(sell_entries)
            sell_dt = _summarise_snapshot_entries(sell_entries, prefer="max")
            if sell_dt is not None:
                result["sell_snapshot_ts"] = _ts_iso_z(sell_dt)
        except FileNotFoundError as exc:
            errors.append(f"sell: {exc}")
        except Exception as exc:  # pragma: no cover - leitura local
            errors.append(f"sell: {exc}")

    ts_candidates = [dt for dt in [buy_dt, sell_dt] if dt is not None]
    if ts_candidates:
        latest = max(ts_candidates)
        result["snapshot_ts"] = _ts_iso_z(latest)

    if errors:
        result["error"] = "; ".join(errors)

    return result

def save_raw_snapshot_array(
    path_or_url: str, out_root_raw: str, label: str, timeout: int = 60
) -> Tuple[List[Dict[str, Any]], str, Path]:
    entries = load_snapshot_array(path_or_url, timeout=timeout)
    snapshot_ts = _detect_snapshot_ts(entries)
    dt = _normalise_timestamp(snapshot_ts)
    assert dt is not None
    history_path = Path(out_root_raw) / f"{label}.json"
    existing_entries = _load_json_list(history_path)
    merged_entries = _merge_history_entries(existing_entries, entries)
    _write_json_list(history_path, merged_entries)

    return entries, snapshot_ts, history_path

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


# ------------------ Aggregated metrics per side ------------------

def _entries_to_price_df(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert snapshot entries into a tidy price DataFrame."""

    rows: List[Dict[str, Any]] = []
    for raw in entries:
        price = _price_cents_to_float(raw.get("price"))
        qty = raw.get("quantity")
        ts = _normalise_timestamp(raw.get("timestamp"))
        slug = (raw.get("item_id") or raw.get("slug") or raw.get("id") or "").strip()
        server = (raw.get("server_id") or raw.get("server") or "devaloka").lower()
        name = str(raw.get("item_name") or slug or "").strip()

        if price is None or ts is None or not slug:
            continue
        if not isinstance(qty, (int, float)):
            continue
        if float(qty) <= 0:
            continue

        rows.append(
            {
                "price": float(price),
                "quantity": float(qty),
                "timestamp": ts,
                "date": ts.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
                "server": server,
                "slug": slug,
                "item": name or slug,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["price", "quantity", "timestamp", "date", "server", "slug", "item"])

    df = pd.DataFrame(rows)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    df = df[df["quantity"] > 0]
    return df


def _weighted_quantiles(values: Sequence[float], weights: Sequence[float], quantiles: Sequence[float]) -> List[float]:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    qs = np.asarray(list(quantiles), dtype=float)
    if vals.size == 0 or w.size == 0 or qs.size == 0:
        return [float("nan")] * len(quantiles)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return [float("nan")] * len(quantiles)
    sorter = np.argsort(vals)
    vals = vals[sorter]
    w = w[sorter]
    cum_weights = np.cumsum(w)
    cum_weights /= cum_weights[-1]
    qs = np.clip(qs, 0.0, 1.0)
    return np.interp(qs, cum_weights, vals).tolist()


def _aggregate_side(entries: List[Dict[str, Any]], side: str) -> pd.DataFrame:
    df = _entries_to_price_df(entries)
    if df.empty:
        return pd.DataFrame()

    percentiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    groups: List[Dict[str, Any]] = []

    for (date, server, slug), grp in df.groupby(["date", "server", "slug"], sort=False):
        qty_total = float(grp["quantity"].sum())
        if qty_total <= 0 or not np.isfinite(qty_total):
            continue

        prices = grp["price"].to_numpy(dtype=float)
        weights = grp["quantity"].to_numpy(dtype=float)
        pct_values = _weighted_quantiles(prices, weights, percentiles)

        price_min = float(np.nanmin(prices)) if prices.size else float("nan")
        price_max = float(np.nanmax(prices)) if prices.size else float("nan")
        weighted_mean = float(np.average(prices, weights=weights)) if weights.sum() else float("nan")

        item_names = grp["item"].dropna().astype(str)
        item_name = next((name.strip() for name in item_names if name.strip()), slug)

        groups.append(
            {
                "date": date,
                "server": server,
                "slug": slug,
                "item": item_name,
                "side": side,
                "price_min": price_min,
                "price_max": price_max,
                "price_weighted_mean": weighted_mean,
                "price_weighted_median": pct_values[2] if len(pct_values) >= 3 else float("nan"),
                "quantity_total": qty_total,
                "pct_10": pct_values[0] if pct_values else float("nan"),
                "pct_30": pct_values[1] if len(pct_values) >= 2 else float("nan"),
                "pct_50": pct_values[2] if len(pct_values) >= 3 else float("nan"),
                "pct_70": pct_values[3] if len(pct_values) >= 4 else float("nan"),
                "pct_90": pct_values[4] if len(pct_values) >= 5 else float("nan"),
            }
        )

    if not groups:
        return pd.DataFrame()

    agg = pd.DataFrame(groups)
    agg["date"] = pd.to_datetime(agg["date"], utc=True, errors="coerce")
    agg = agg.dropna(subset=["date", "slug", "server"])
    return agg


def update_side_csv(entries: List[Dict[str, Any]], csv_path: str, side: str) -> None:
    agg = _aggregate_side(entries, side)
    if agg.empty:
        return

    dest = Path(csv_path)
    _ensure_dir(dest.parent)

    if dest.exists():
        try:
            df_old = pd.read_csv(dest)
        except Exception:
            df_old = pd.DataFrame(columns=agg.columns)
    else:
        df_old = pd.DataFrame(columns=agg.columns)

    for col in agg.columns:
        if col not in df_old.columns:
            df_old[col] = pd.NA

    agg_out = agg.copy()
    agg_out["date"] = agg_out["date"].dt.strftime("%Y-%m-%d")
    agg_out["quantity_total"] = (
        pd.to_numeric(agg_out["quantity_total"], errors="coerce").round().astype("Int64")
    )

    combined = pd.concat([df_old, agg_out], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["date", "slug", "server"])
    combined = combined.drop_duplicates(subset=["date", "slug", "server"], keep="last")
    combined = combined.sort_values(["slug", "date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(dest, index=False)

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
        if r.get("item") and r.get("top_buy") is not None:
            proj.append({
                "timestamp": r["timestamp"],
                "item": r["item"],
                "buy_market": r["top_buy"],
                "sell_market": r.get("low_sell"),
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
        # fallback: não quebrar pipeline caso history_local.json esteja corrompido
        pass

# ------------------ Orchestration ------------------

def sync_sources_save_raw_and_update_csv(
    buy_orders_path_or_url: str,
    sell_orders_path_or_url: Optional[str],
    raw_root: str,
    buy_csv_path: str,
    sell_csv_path: str,
    history_json_path: str,
    server: str = "devaloka",
    timeout: int = 60,
) -> Dict[str, Any]:
    """Baixa snapshots de buy/sell orders, atualiza CSVs e history_local.json."""

    buy_entries: List[Dict[str, Any]] = []
    sell_entries: List[Dict[str, Any]] = []
    buy_snapshot_ts: Optional[str] = None
    sell_snapshot_ts: Optional[str] = None

    if buy_orders_path_or_url:
        buy_entries, buy_snapshot_ts, _ = save_raw_snapshot_array(
            buy_orders_path_or_url, raw_root, label="buy", timeout=timeout
        )
        if buy_csv_path:
            update_side_csv(buy_entries, buy_csv_path, side="buy")

    if sell_orders_path_or_url:
        sell_entries, sell_snapshot_ts, _ = save_raw_snapshot_array(
            sell_orders_path_or_url, raw_root, label="sell", timeout=timeout
        )
        if sell_csv_path:
            update_side_csv(sell_entries, sell_csv_path, side="sell")

    records = extract_records_from_snapshots(buy_entries, sell_entries, server=server)
    append_history_json_from_records(records, history_json_path)

    ts_candidates = [
        _normalise_timestamp(buy_snapshot_ts),
        _normalise_timestamp(sell_snapshot_ts),
    ]
    ts_candidates = [ts for ts in ts_candidates if ts is not None]
    now_dt = datetime.now(timezone.utc)
    latest_dt = max(ts_candidates) if ts_candidates else now_dt
    snapshot_iso = _ts_iso_z(latest_dt)

    latest_snapshot_path = Path(raw_root) / "latest_snapshot.json"
    latest_snapshot_payload = {
        "source": "remote",
        "snapshot_ts": snapshot_iso,
        "records": records,
        "record_count": len(records),
        "generated_at": _ts_iso_z(now_dt),
    }
    _ensure_dir(latest_snapshot_path.parent)
    with open(latest_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(latest_snapshot_payload, f, ensure_ascii=False, indent=2)

    now_iso = _ts_iso_z(now_dt)

    meta_payload = {
        "source": "remote",
        "snapshot_ts": snapshot_iso,
        "buy_snapshot_ts": buy_snapshot_ts,
        "sell_snapshot_ts": sell_snapshot_ts,
        "buy_entries": len(buy_entries),
        "sell_entries": len(sell_entries),
        "records": len(records),
        "updated_at": now_iso,
        "snapshot_path": str(latest_snapshot_path),
    }

    meta_path = Path(raw_root) / "last_sync_meta.json"
    _update_last_sync_metadata(meta_path, "remote", meta_payload, latest_snapshot_path)

    return meta_payload


def _write_snapshot_bundle(
    bundle_path: Path,
    source: str,
    snapshot_iso: str,
    buy_entries: List[Dict[str, Any]],
    sell_entries: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "source": source,
        "snapshot_ts": snapshot_iso,
        "buy": buy_entries,
        "sell": sell_entries,
    }
    if extra:
        payload.update(extra)
    _ensure_dir(bundle_path.parent)
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def collect_remote_snapshot(
    buy_url_or_path: str,
    sell_url_or_path: Optional[str] = None,
    raw_root: str = "raw",
    server: str = "devaloka",
    timeout: int = 60,
) -> Dict[str, Any]:
    buy_entries: List[Dict[str, Any]] = []
    sell_entries: List[Dict[str, Any]] = []
    buy_snapshot_ts: Optional[str] = None
    sell_snapshot_ts: Optional[str] = None

    if buy_url_or_path:
        buy_entries, buy_snapshot_ts, _ = save_raw_snapshot_array(
            buy_url_or_path, raw_root, label="buy", timeout=timeout
        )

    if sell_url_or_path:
        sell_entries, sell_snapshot_ts, _ = save_raw_snapshot_array(
            sell_url_or_path, raw_root, label="sell", timeout=timeout
        )

    ts_candidates = [
        _normalise_timestamp(buy_snapshot_ts),
        _normalise_timestamp(sell_snapshot_ts),
    ]
    ts_candidates = [ts for ts in ts_candidates if ts is not None]
    snapshot_dt = max(ts_candidates) if ts_candidates else datetime.now(timezone.utc)
    snapshot_iso = _ts_iso_z(snapshot_dt)

    bundle_path = Path(raw_root) / "collected" / "remote_latest.json"
    _write_snapshot_bundle(
        bundle_path,
        source="remote",
        snapshot_iso=snapshot_iso,
        buy_entries=buy_entries,
        sell_entries=sell_entries,
        extra={
            "buy_snapshot_ts": buy_snapshot_ts,
            "sell_snapshot_ts": sell_snapshot_ts,
        },
    )

    now_iso = _ts_iso_z(datetime.now(timezone.utc))
    meta_payload: Dict[str, Any] = {
        "source": "remote",
        "collected_snapshot_ts": snapshot_iso,
        "collected_at": now_iso,
        "buy_snapshot_ts": buy_snapshot_ts,
        "sell_snapshot_ts": sell_snapshot_ts,
        "buy_entries": len(buy_entries),
        "sell_entries": len(sell_entries),
        "collected_payload_path": str(bundle_path),
        "server": server,
    }

    meta_path = Path(raw_root) / "last_sync_meta.json"
    updated_meta = _update_last_sync_metadata(meta_path, "remote", meta_payload)

    return {
        "meta": updated_meta,
        "buy_entries": buy_entries,
        "sell_entries": sell_entries,
        "snapshot_iso": snapshot_iso,
        "bundle_path": str(bundle_path),
    }


def collect_local_snapshot(
    buy_orders_dir: str,
    auctions_dir: Optional[str] = None,
    raw_root: str = "raw",
    server: str = "devaloka",
    snapshot_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    buy_entries_raw = _load_local_snapshot_entries(Path(buy_orders_dir))

    sell_entries_raw: List[Dict[str, Any]] = []
    if auctions_dir and str(auctions_dir).strip():
        sell_entries_raw = _load_local_snapshot_entries(Path(auctions_dir))

    timestamp_candidates: List[datetime] = []
    buy_dt = _summarise_snapshot_entries(buy_entries_raw, prefer="max")
    if buy_dt is not None:
        timestamp_candidates.append(buy_dt)
    sell_dt = None
    if sell_entries_raw:
        sell_dt = _summarise_snapshot_entries(sell_entries_raw, prefer="max")
        if sell_dt is not None:
            timestamp_candidates.append(sell_dt)

    if timestamp_candidates:
        snapshot_dt = max(timestamp_candidates).astimezone(timezone.utc)
    else:
        snapshot_dt = _normalise_timestamp(snapshot_time) if snapshot_time else None
        if snapshot_dt is None:
            snapshot_dt = datetime.now(timezone.utc)
    snapshot_iso = _ts_iso_z(snapshot_dt)

    buy_entries = [_normalise_local_entry(e, snapshot_dt, server) for e in buy_entries_raw]

    raw_root_path = Path(raw_root)
    _append_raw_entries(raw_root_path / "buy.json", buy_entries)

    sell_entries: List[Dict[str, Any]] = []
    if sell_entries_raw:
        sell_entries = [
            _normalise_local_entry(e, snapshot_dt, server) for e in sell_entries_raw
        ]
        _append_raw_entries(raw_root_path / "sell.json", sell_entries)

    bundle_path = raw_root_path / "collected" / "local_latest.json"
    _write_snapshot_bundle(
        bundle_path,
        source="local",
        snapshot_iso=snapshot_iso,
        buy_entries=buy_entries,
        sell_entries=sell_entries,
    )

    now_iso = _ts_iso_z(datetime.now(timezone.utc))
    meta_payload: Dict[str, Any] = {
        "source": "local",
        "collected_snapshot_ts": snapshot_iso,
        "collected_at": now_iso,
        "buy_entries": len(buy_entries),
        "sell_entries": len(sell_entries),
        "collected_payload_path": str(bundle_path),
        "server": server,
    }

    meta_path = raw_root_path / "last_sync_meta.json"
    updated_meta = _update_last_sync_metadata(meta_path, "local", meta_payload)

    return {
        "meta": updated_meta,
        "buy_entries": buy_entries,
        "sell_entries": sell_entries,
        "snapshot_iso": snapshot_iso,
        "bundle_path": str(bundle_path),
    }


def _process_snapshot_entries(
    source: str,
    snapshot_iso: str,
    buy_entries: List[Dict[str, Any]],
    sell_entries: List[Dict[str, Any]],
    raw_root_path: Path,
    buy_csv_path: Optional[str],
    sell_csv_path: Optional[str],
    history_json_path: str,
    server: str,
    buy_history_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    if buy_csv_path:
        update_side_csv(buy_entries, buy_csv_path, side="buy")
    if sell_csv_path:
        update_side_csv(sell_entries, sell_csv_path, side="sell")

    if source == "local" and buy_history_csv_path:
        summary_df = _build_buy_snapshot_summary(buy_entries, snapshot_iso)
        _update_buy_history_csv(Path(buy_history_csv_path), summary_df)

    records = extract_records_from_snapshots(buy_entries, sell_entries, server=server)
    append_history_json_from_records(records, history_json_path)

    now_dt = datetime.now(timezone.utc)
    latest_snapshot_path = raw_root_path / "latest_snapshot.json"
    latest_snapshot_payload = {
        "source": source,
        "snapshot_ts": snapshot_iso,
        "records": records,
        "record_count": len(records),
        "generated_at": _ts_iso_z(now_dt),
    }
    _ensure_dir(latest_snapshot_path.parent)
    with open(latest_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(latest_snapshot_payload, f, ensure_ascii=False, indent=2)

    meta_payload: Dict[str, Any] = {
        "source": source,
        "snapshot_ts": snapshot_iso,
        "processed_snapshot_ts": snapshot_iso,
        "processed_at": _ts_iso_z(now_dt),
        "records": len(records),
        "updated_at": _ts_iso_z(now_dt),
        "buy_entries": len(buy_entries),
        "sell_entries": len(sell_entries),
    }

    if buy_history_csv_path and source == "local":
        meta_payload["buy_history_path"] = str(Path(buy_history_csv_path))

    meta_path = raw_root_path / "last_sync_meta.json"
    updated_meta = _update_last_sync_metadata(
        meta_path, source, meta_payload, latest_snapshot_path
    )

    return updated_meta


def _load_snapshot_bundle(path_value: Optional[str], raw_root: Path) -> Dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = raw_root / path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def process_latest_snapshot(
    raw_root: str = "raw",
    buy_csv_path: str = "data/history_devaloka_buy.csv",
    sell_csv_path: str = "data/history_devaloka_sell.csv",
    history_json_path: str = "history_local.json",
    buy_history_csv_path: Optional[str] = "data/history_buy_local.csv",
    server: str = "devaloka",
    prefer_source: Optional[str] = None,
) -> Dict[str, Any]:
    raw_root_path = Path(raw_root)
    meta_path = raw_root_path / "last_sync_meta.json"
    meta = _load_json_object(meta_path)

    def _candidate_from_meta(src: str) -> Optional[Tuple[datetime, str, Dict[str, Any]]]:
        entry = meta.get(src)
        if not isinstance(entry, dict):
            return None
        ts_value = entry.get("collected_snapshot_ts") or entry.get("snapshot_ts")
        ts_dt = _normalise_timestamp(ts_value)
        if ts_dt is None:
            return None
        return ts_dt, src, entry

    candidates: List[Tuple[datetime, str, Dict[str, Any]]] = []
    if prefer_source:
        preferred = _candidate_from_meta(prefer_source)
        if preferred:
            candidates.append(preferred)
    else:
        for src in ("remote", "local"):
            candidate = _candidate_from_meta(src)
            if candidate:
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("Nenhum snapshot coletado para processar")

    chosen_dt, chosen_source, chosen_entry = max(candidates, key=lambda tup: tup[0])

    bundle_data = _load_snapshot_bundle(
        chosen_entry.get("collected_payload_path"), raw_root_path
    )

    buy_entries = bundle_data.get("buy") if isinstance(bundle_data, dict) else None
    sell_entries = bundle_data.get("sell") if isinstance(bundle_data, dict) else None

    if buy_entries is None:
        buy_entries = []
    if sell_entries is None:
        sell_entries = []

    snapshot_iso = chosen_entry.get("collected_snapshot_ts") or chosen_entry.get(
        "snapshot_ts"
    )
    if not snapshot_iso:
        snapshot_iso = _ts_iso_z(chosen_dt)

    processed_meta = _process_snapshot_entries(
        source=chosen_source,
        snapshot_iso=snapshot_iso,
        buy_entries=buy_entries,
        sell_entries=sell_entries,
        raw_root_path=raw_root_path,
        buy_csv_path=buy_csv_path,
        sell_csv_path=sell_csv_path,
        history_json_path=history_json_path,
        server=chosen_entry.get("server") or server,
        buy_history_csv_path=buy_history_csv_path if chosen_source == "local" else None,
    )

    refreshed_meta = _load_json_object(meta_path)
    if isinstance(refreshed_meta, dict):
        refreshed_meta["_last_processed_source"] = chosen_source
        _write_json_object(meta_path, refreshed_meta)

    return {
        "source": chosen_source,
        "snapshot_ts": processed_meta.get("snapshot_ts"),
        "records": processed_meta.get("records"),
        "processed_meta": processed_meta,
    }

def rebuild_csv_from_raw(
    raw_root: str,
    buy_csv_path: str,
    sell_csv_path: Optional[str],
    history_json_path: str,
    server: str = "devaloka",
) -> None:
    buy_path = Path(raw_root) / "buy.json"
    sell_path = Path(raw_root) / "sell.json"

    if not buy_path.exists() and not sell_path.exists():
        raise FileNotFoundError(f"Nada em {buy_path.parent}")

    buy_entries = _load_json_list(buy_path)
    sell_entries = _load_json_list(sell_path)

    if buy_csv_path:
        update_side_csv(buy_entries, buy_csv_path, side="buy")
    if sell_csv_path:
        update_side_csv(sell_entries, sell_csv_path, side="sell")

    records = extract_records_from_snapshots(buy_entries, sell_entries, server=server)
    append_history_json_from_records(records, history_json_path)

# Conveniências simples para o app
def run_sync(
    buy_url_or_path: str,
    sell_url_or_path: Optional[str] = None,
    raw_root: str = "raw",
    buy_csv_path: str = "data/history_devaloka_buy.csv",
    sell_csv_path: str = "data/history_devaloka_sell.csv",
    history_json_path: str = "history_local.json",
    server: str = "devaloka",
    timeout: int = 60,
    process: bool = True,
) -> Dict[str, Any]:
    collected = collect_remote_snapshot(
        buy_url_or_path,
        sell_url_or_path,
        raw_root=raw_root,
        server=server,
        timeout=timeout,
    )

    if not process:
        return collected["meta"]

    processed_meta = _process_snapshot_entries(
        source="remote",
        snapshot_iso=collected.get("snapshot_iso") or _ts_iso_z(datetime.now(timezone.utc)),
        buy_entries=collected.get("buy_entries", []),
        sell_entries=collected.get("sell_entries", []),
        raw_root_path=Path(raw_root),
        buy_csv_path=buy_csv_path,
        sell_csv_path=sell_csv_path,
        history_json_path=history_json_path,
        server=server,
    )
    return processed_meta


def run_rebuild(
    raw_root: str = "raw",
    buy_csv_path: str = "data/history_devaloka_buy.csv",
    sell_csv_path: str = "data/history_devaloka_sell.csv",
    history_json_path: str = "history_local.json",
    server: str = "devaloka",
) -> None:
    rebuild_csv_from_raw(
        raw_root,
        buy_csv_path,
        sell_csv_path,
        history_json_path,
        server=server,
    )


def rebuild_buy_history_from_raw(
    raw_root: str = "raw",
    buy_history_csv_path: str = "data/history_buy_local.csv",
) -> Dict[str, Any]:
    buy_path = Path(raw_root) / "buy.json"
    entries = _load_json_list(buy_path)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        snapshot_dt = _normalise_timestamp(entry.get("snapshot_ts"))
        if snapshot_dt is None:
            snapshot_dt = _normalise_timestamp(entry.get("timestamp"))
        if snapshot_dt is None:
            continue
        snapshot_iso = _ts_iso_z(snapshot_dt)
        grouped.setdefault(snapshot_iso, []).append(entry)

    frames: List[pd.DataFrame] = []
    for snapshot_iso in sorted(grouped.keys()):
        summary_df = _build_buy_snapshot_summary(grouped[snapshot_iso], snapshot_iso)
        if summary_df.empty:
            continue
        frames.append(summary_df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=BUY_HISTORY_COLUMNS)

    updated = _update_buy_history_csv(Path(buy_history_csv_path), combined, replace=True)
    return {
        "path": str(Path(buy_history_csv_path)),
        "rows": int(len(updated)),
        "snapshots": len(grouped),
    }


def run_sync_local_snapshot(
    buy_orders_dir: str,
    auctions_dir: Optional[str] = None,
    raw_root: str = "raw",
    buy_csv_path: str = "data/history_devaloka_buy.csv",
    sell_csv_path: str = "data/history_devaloka_sell.csv",
    history_json_path: str = "history_local.json",
    buy_history_csv_path: str = "data/history_buy_local.csv",
    server: str = "devaloka",
    snapshot_time: Optional[datetime] = None,
    process: bool = True,
) -> Dict[str, Any]:
    collected = collect_local_snapshot(
        buy_orders_dir,
        auctions_dir=auctions_dir,
        raw_root=raw_root,
        server=server,
        snapshot_time=snapshot_time,
    )

    if not process:
        return collected["meta"]

    snapshot_iso = collected.get("snapshot_iso")
    if not snapshot_iso:
        snapshot_iso = _ts_iso_z(
            _normalise_timestamp(snapshot_time) or datetime.now(timezone.utc)
        )

    processed_meta = _process_snapshot_entries(
        source="local",
        snapshot_iso=snapshot_iso,
        buy_entries=collected.get("buy_entries", []),
        sell_entries=collected.get("sell_entries", []),
        raw_root_path=Path(raw_root),
        buy_csv_path=buy_csv_path,
        sell_csv_path=sell_csv_path,
        history_json_path=history_json_path,
        server=server,
        buy_history_csv_path=buy_history_csv_path,
    )

    if buy_history_csv_path:
        processed_meta.setdefault("buy_history_path", str(Path(buy_history_csv_path)))

    return processed_meta
