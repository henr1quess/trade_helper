"""Script to fetch a NWMP cloud snapshot and generate an item id -> item name map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

GT_BASE = "https://nwmpdata.gaming.tools"
DEFAULT_SERVER = "devaloka"
DEFAULT_OUTPUT = Path("raw/item_name_map.json")


class SnapshotFormatError(RuntimeError):
    """Raised when the snapshot payload does not match the expected format."""


def build_snapshot_url(server: str) -> str:
    """Return the Gaming Tools auctions snapshot URL for a given server."""
    server = server.strip().lower()
    if not server:
        raise ValueError("O servidor não pode ser vazio.")
    return f"{GT_BASE}/auctions2/{server}.json"


def fetch_snapshot(url: str) -> Iterable[Dict[str, Any]]:
    """Fetch a snapshot from the provided URL and return the JSON payload."""
    try:
        with urlopen(url, timeout=60) as response:  # type: ignore[arg-type]
            status = getattr(response, "status", 200)
            if status not in (None, 200):
                raise SnapshotFormatError(
                    f"Falha ao baixar snapshot: status HTTP {status}."
                )
            data = json.load(response)
    except HTTPError as exc:
        raise SnapshotFormatError(
            f"Falha ao baixar snapshot: {exc.code} {exc.reason}."
        ) from exc
    except URLError as exc:
        raise SnapshotFormatError(
            f"Não foi possível acessar o snapshot em {url}: {exc.reason}."
        ) from exc
    if not isinstance(data, list):
        raise SnapshotFormatError(
            f"Snapshot em {url} deveria ser uma lista, mas veio {type(data).__name__}."
        )
    return data


def build_item_name_map(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, set[str]]]:
    """Return the item_id -> item_name mapping and track conflicting names."""
    mapping: Dict[str, str] = {}
    conflicts: Dict[str, set[str]] = {}

    for entry in records:
        if not isinstance(entry, dict):
            continue
        item_id = entry.get("item_id")
        item_name = entry.get("item_name")
        if not item_id or not item_name:
            continue
        existing = mapping.get(item_id)
        if existing and existing != item_name:
            conflicts.setdefault(item_id, {existing}).add(item_name)
        else:
            mapping[item_id] = item_name

    return mapping, conflicts


def write_mapping(
    mapping: Dict[str, str], output_path: Path, *, indent: int | None = 2
) -> None:
    """Persist the mapping to the desired output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp, ensure_ascii=False, indent=indent, sort_keys=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa um snapshot da nuvem e gera um mapa item_id -> item_name."
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Servidor do New World (padrão: {DEFAULT_SERVER}).",
    )
    parser.add_argument(
        "--url",
        help="URL completa para o snapshot. Se omitido, usa o padrão baseado no servidor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Caminho do arquivo de saída (padrão: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentação para o JSON gerado (padrão: 2). Use 0 para JSON compacto.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    url = args.url or build_snapshot_url(args.server)

    records = fetch_snapshot(url)
    mapping, conflicts = build_item_name_map(records)

    write_mapping(mapping, args.output, indent=args.indent if args.indent > 0 else None)

    print(f"Mapa gerado com {len(mapping)} itens e salvo em {args.output}.")
    if conflicts:
        print("\nAviso: Foram encontradas divergências de nomes para alguns item_ids:")
        for item_id, names in conflicts.items():
            lista = ", ".join(sorted(names))
            print(f"  - {item_id}: {lista}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
