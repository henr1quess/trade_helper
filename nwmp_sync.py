# nwmp_sync.py
from __future__ import annotations
# Requisitos: pandas (obrigatório), numpy (para estatísticas), requests (apenas para URLs)
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

# ------------------ History.json projection (for the current app) ------------------

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
        buy_entries = load_snapshot_array(buy_url_or_path, timeout=timeout)
        if buy_entries:
            buy_snapshot_ts = _detect_snapshot_ts(buy_entries)

    if sell_url_or_path:
        sell_entries = load_snapshot_array(sell_url_or_path, timeout=timeout)
        if sell_entries:
            sell_snapshot_ts = _detect_snapshot_ts(sell_entries)

    ts_candidates = [
        _normalise_timestamp(buy_snapshot_ts),
        _normalise_timestamp(sell_snapshot_ts),
    ]
    ts_candidates = [dt for dt in ts_candidates if dt is not None]
    if ts_candidates:
        snapshot_dt = max(ts_candidates)
    else:
        snapshot_dt = datetime.now(timezone.utc)
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

    buy_dt = _summarise_snapshot_entries(buy_entries_raw, prefer="max")

    snapshot_dt = buy_dt.astimezone(timezone.utc) if buy_dt is not None else None
    if snapshot_dt is None:
        snapshot_dt = _normalise_timestamp(snapshot_time) if snapshot_time else None
    if snapshot_dt is None:
        snapshot_dt = datetime.now(timezone.utc)
    snapshot_iso = _ts_iso_z(snapshot_dt)

    buy_entries = [_normalise_local_entry(e, snapshot_dt, server) for e in buy_entries_raw]

    raw_root_path = Path(raw_root)
    sell_entries: List[Dict[str, Any]] = []
    if sell_entries_raw:
        sell_entries = [
            _normalise_local_entry(e, snapshot_dt, server) for e in sell_entries_raw
        ]

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
    server: str,
) -> Dict[str, Any]:
    records = extract_records_from_snapshots(buy_entries, sell_entries, server=server)

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
        server=chosen_entry.get("server") or server,
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

# Conveniências simples para o app
def run_sync(
    buy_url_or_path: str,
    sell_url_or_path: Optional[str] = None,
    raw_root: str = "raw",
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
        server=server,
    )
    return processed_meta


def run_sync_local_snapshot(
    buy_orders_dir: str,
    auctions_dir: Optional[str] = None,
    raw_root: str = "raw",
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
        server=server,
    )

    return processed_meta
