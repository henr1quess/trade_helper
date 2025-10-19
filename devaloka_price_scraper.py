"""Utilities to fetch and normalize price history from NWMP (Devaloka).

This module exposes three helpers that are consumed by the Streamlit app:
    * :func:`fetch_item_data` — obtain the raw JSON for a given item slug.
    * :func:`extract_devaloka_records` — normalize the payload into flat rows.
    * :func:`update_csv` — merge the collected rows into a CSV history file.

The implementation favours resiliency because the NWMP endpoints are not
publicly documented. Multiple URL patterns are attempted and the parser accepts
several payload shapes (list of dicts, dicts keyed by timestamp, etc.).
"""
from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

try:  # pragma: no cover - fallback to stdlib if requests is unavailable
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    from urllib import request as urllib_request  # type: ignore
else:
    urllib_request = None  # type: ignore

REMOTE_URL_PATTERNS: Tuple[str, ...] = (
    "https://cdn.nwmarketprices.com/history/{slug}.json",
    "https://cdn.nwmarketprices.com/history/{slug}.json.gz",
    "https://cdn.nwmarketprices.com/market-history/{slug}.json",
    "https://cdn.nwmarketprices.com/market-history/{slug}.json.gz",
    "https://storage.googleapis.com/nwmp-history/{slug}.json",
    "https://storage.googleapis.com/nwmp-history/{slug}.json.gz",
    "https://nwmarketprices.com/api/history/{slug}",
)


def _read_json_bytes(raw: bytes) -> Any:
    text = raw.decode("utf-8")
    return json.loads(text)


def _load_local_payload(slug: str, local_dir: Path) -> Optional[Any]:
    candidates: List[Path] = [
        local_dir / f"{slug}.json",
        local_dir / f"{slug}.json.gz",
    ]
    if not local_dir.exists():
        return None
    if not any(p.exists() for p in candidates):
        # fallback: pick first file that contains the slug name
        candidates.extend(sorted(local_dir.glob(f"*{slug}*.json*")))
    last_error: Optional[Exception] = None
    for path in candidates:
        if not path.exists():
            continue
        data = path.read_bytes()
        if path.suffix == ".gz" or path.suffixes[-1:] == [".gz"]:
            data = gzip.decompress(data)
        try:
            payload = _read_json_bytes(data)
            return payload
        except Exception as exc:  # pragma: no cover - lenient: try next file
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"Não foi possível carregar {slug!r} localmente: {last_error}")
    return None


def _http_get(url: str, timeout: int = 30) -> Optional[bytes]:
    if requests is not None:
        try:
            resp = requests.get(url, timeout=timeout)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        return resp.content
    # fallback using urllib
    assert urllib_request is not None  # pragma: no cover
    try:
        with urllib_request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:  # pragma: no cover
        return None


def fetch_item_data(slug: str, local_dir: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve the raw NWMP payload for ``slug``.

    Parameters
    ----------
    slug:
        Item identifier used by NWMP.
    local_dir:
        Optional directory containing ``.json``/``.json.gz`` snapshots. When
        supplied, the local snapshot is preferred over remote downloads.
    """

    slug = (slug or "").strip()
    if not slug:
        raise ValueError("slug não pode ser vazio")

    payload: Optional[Any] = None
    if local_dir:
        payload = _load_local_payload(slug, Path(local_dir))

    if payload is None:
        last_error: Optional[str] = None
        for pattern in REMOTE_URL_PATTERNS:
            url = pattern.format(slug=slug)
            raw = _http_get(url)
            if raw is None:
                last_error = url
                continue
            if url.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    # some CDNs serve the plain JSON even with .gz extension
                    pass
            try:
                payload = _read_json_bytes(raw)
                break
            except Exception:
                last_error = url
                continue
        if payload is None and last_error:
            raise RuntimeError(f"Falha ao baixar dados do slug {slug!r} (última URL: {last_error})")

    if payload is None:
        raise RuntimeError(f"Nenhum dado encontrado para slug {slug!r}")

    return {"slug": slug, "payload": payload}


def _normalise_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # heurística: valores grandes (>= 1e12) representam milissegundos
        divisor = 1000.0 if float(value) >= 1_000_000_000_000 else 1.0
        try:
            return datetime.fromtimestamp(float(value) / divisor, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            return None
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    return None


def _first_not_null(entry: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in entry and entry[key] is not None:
            try:
                val = float(entry[key])
            except (TypeError, ValueError):
                continue
            if math.isnan(val):
                continue
            return val
    return None


def _first_int(entry: Dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    val = _first_not_null(entry, keys)
    if val is None:
        return None
    return int(val)


def _iter_server_entries(payload: Any) -> Iterator[Tuple[Optional[str], List[Dict[str, Any]]]]:
    if isinstance(payload, dict):
        nested_keys = ("data", "prices", "servers")
        for key in nested_keys:
            node = payload.get(key)
            if isinstance(node, dict):
                for srv, entries in node.items():
                    yield srv, _ensure_entries(entries)
                return
        if "history" in payload and isinstance(payload["history"], dict):
            history = payload["history"]
            for srv, entries in history.items():
                yield srv, _ensure_entries(entries)
            return
        yield payload.get("server"), _ensure_entries(payload)
        return
    if isinstance(payload, list):
        yield None, _ensure_entries(payload)
        return
    yield None, []


def _ensure_entries(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [entry for entry in obj if isinstance(entry, dict)]
    if isinstance(obj, dict):
        out: List[Dict[str, Any]] = []
        for key, value in obj.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("timestamp", key)
                out.append(entry)
        return out
    return []


def extract_devaloka_records(
    response: Dict[str, Any],
    server: str = "devaloka",
) -> List[Dict[str, Any]]:
    """Normalise a NWMP payload into rows ready to append to CSV."""

    payload = response.get("payload") if isinstance(response, dict) else response
    slug_hint = response.get("slug") if isinstance(response, dict) else None
    item_hint = response.get("item") or response.get("item_name") if isinstance(response, dict) else None
    server_norm = (server or "").lower()
    rows: List[Dict[str, Any]] = []

    for srv_key, entries in _iter_server_entries(payload):
        srv_name = srv_key or server_norm or None
        if server_norm and srv_name and srv_name.lower() != server_norm:
            continue
        for entry in entries:
            timestamp = _normalise_timestamp(
                entry.get("timestamp")
                or entry.get("time")
                or entry.get("Date")
                or entry.get("date")
                or entry.get("ts")
            )
            if timestamp is None:
                continue
            buy_price = _first_not_null(
                entry,
                (
                    "top_buy",
                    "buy_price",
                    "highest_buy",
                    "max_buy",
                    "highest",
                    "buy",
                    "buyPrice",
                    "max",  # algumas fontes usam min/max
                ),
            )
            sell_price = _first_not_null(
                entry,
                (
                    "low_sell",
                    "sell_price",
                    "lowest_sell",
                    "min_sell",
                    "sell",
                    "sellPrice",
                    "min",
                    "price",
                    "price_low",
                ),
            )
            avg_price = _first_not_null(entry, ("avg_price", "average", "avg", "averagePrice"))
            volume = _first_int(entry, ("volume", "quantity", "count", "qty"))

            record = {
                "timestamp": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "item": entry.get("item")
                or entry.get("itemName")
                or entry.get("name")
                or item_hint
                or slug_hint,
                "slug": entry.get("slug") or slug_hint,
                "server": (srv_name or server_norm or "devaloka").lower(),
                "top_buy": buy_price,
                "low_sell": sell_price,
                "avg_price": avg_price,
                "volume": volume,
                "source": entry.get("source") or "nwmp",
            }
            rows.append(record)
    return rows


def update_csv(records: List[Dict[str, Any]], csv_path: str) -> None:
    """Append ``records`` to ``csv_path`` dropping duplicates."""

    if not records:
        return
    df_new = pd.DataFrame(records)
    if df_new.empty:
        return

    expected_cols = [
        "timestamp",
        "item",
        "slug",
        "server",
        "top_buy",
        "low_sell",
        "avg_price",
        "volume",
        "source",
    ]
    for col in expected_cols:
        if col not in df_new.columns:
            df_new[col] = None

    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True, errors="coerce")
    df_new = df_new.dropna(subset=["timestamp"])  # descarta linhas inválidas

    for col in ["top_buy", "low_sell", "avg_price"]:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
    if "volume" in df_new.columns:
        df_new["volume"] = pd.to_numeric(df_new["volume"], errors="coerce").astype("Int64")

    df_new["timestamp"] = df_new["timestamp"].dt.tz_convert(timezone.utc)

    dest = Path(csv_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        try:
            df_existing = pd.read_csv(dest)
        except Exception:
            df_existing = pd.DataFrame(columns=expected_cols)
    else:
        df_existing = pd.DataFrame(columns=expected_cols)

    if not df_existing.empty:
        df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True, errors="coerce")

    combined = pd.concat([df_existing, df_new], ignore_index=True)
    subset = [c for c in ["timestamp", "item", "server", "slug"] if c in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")

    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp")
        combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    combined.to_csv(dest, index=False)
