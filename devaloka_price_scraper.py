"""
devaloka_price_scraper.py
===========================

Coleta histórico de preços do servidor Devaloka (New World, NWMP) e
atualiza/append em um CSV. Funciona sem argumentos de linha de comando:
basta editar ITEM_IDS, CSV_PATH e LOCAL_JSON_DIR abaixo.

- Extrai data (YYYY-MM-DD) e hora (HH:MM:SS) a partir do timestamp.
- Garante que a coluna 'time' fique como 2ª coluna (logo após 'date').
- Evita duplicatas usando (date, time, item_id).
- Remove linhas antigas com 'time' vazio quando existir registro com hora
  para a mesma (date, item_id).
- Registra logs de progresso.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import gzip
import json
import os
from datetime import datetime

import pandas as pd  # type: ignore
import requests
import logging

# -------- CONFIGURAÇÕES --------
ITEM_IDS: List[str] = [
    "oret1",      # Iron Ore
    "bananat1",   # Banana (exemplo)
]

_ENV_CSV_PATH = os.getenv("DEVALOKA_SCRAPER_CSV_PATH")
_ENV_OUTPUT_DIR = os.getenv("DEVALOKA_SCRAPER_OUTPUT_DIR")
_ENV_CSV_FILENAME = os.getenv("DEVALOKA_SCRAPER_CSV_FILENAME", "devaloka_prices.csv")

if _ENV_CSV_PATH:
    CSV_PATH: str = _ENV_CSV_PATH
elif _ENV_OUTPUT_DIR:
    CSV_PATH = os.path.join(_ENV_OUTPUT_DIR, _ENV_CSV_FILENAME)
else:
    CSV_PATH = _ENV_CSV_FILENAME

# Se o CDN bloquear requests no seu ambiente, baixe {item}.json(.gz) manualmente
# e aponte LOCAL_JSON_DIR para a pasta. Deixe None para baixar do CDN.
LOCAL_JSON_DIR: Optional[str] = os.getenv("DEVALOKA_SCRAPER_LOCAL_DIR")
# --------------------------------

# ---- LOGGING ----
logging.basicConfig(
    level=logging.INFO,  # troque para logging.DEBUG para mais verbosidade
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
# -----------------


def fetch_item_data(item_id: str, *, local_dir: Optional[str] = None) -> Dict:
    """Busca o JSON bruto do item (CDN ou diretório local)."""
    if local_dir:
        for ext in (".json", ".json.gz"):
            path = os.path.join(local_dir, f"{item_id}{ext}")
            if os.path.exists(path):
                logger.info(f"Lendo {item_id} de {path}")
                with open(path, "rb") as f:
                    data = f.read()
                if ext.endswith(".gz"):
                    data = gzip.decompress(data)
                return json.loads(data.decode("utf-8"))
        raise FileNotFoundError(f"{item_id}: arquivo não encontrado em {local_dir}")

    url = f"https://scdn.gaming.tools/nwmp/dev/history/items/{item_id}.json.gz"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; devaloka-scraper)",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    logger.info(f"Baixando {item_id} de {url}")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    raw = r.content

    # Descomprime se for gzip
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        try:
            raw = gzip.decompress(raw)
        except OSError:
            logger.warning(f"{item_id}: falha ao descomprimir; tentando como JSON puro.")
    return json.loads(raw.decode("utf-8"))


def extract_devaloka_records(item_data: Dict, *, server: str = "devaloka") -> List[Dict[str, object]]:
    """Normaliza as entradas de Devaloka, convertendo preços e data/hora."""
    if "servers" not in item_data or server not in item_data["servers"]:
        return []

    records: List[Dict[str, object]] = []
    for entry in item_data["servers"][server]:
        ts = datetime.utcfromtimestamp(entry["timestamp"])
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M:%S")

        min_price = entry["min_price"] / 100.0
        median_price = entry["median_price"] / 100.0
        mean_price = entry["mean_price"] / 100.0
        quantity = int(str(entry["quantity"]).replace(",", ""))

        percentile_map: Dict[int, Tuple[float, int]] = {}
        for pct, price, qty in entry.get("means", []):
            key = int(round(pct * 100))  # 0.1->10 etc.
            percentile_map[key] = (price / 100.0, int(qty))

        records.append({
            "date": date_str,
            "time": time_str,  # <- garantimos a coluna time
            "item_id": item_data.get("item_id"),
            "min_price": min_price,
            "median_price": median_price,
            "mean_price": mean_price,
            "quantity": quantity,
            "p10_price": percentile_map.get(10, (None, None))[0],
            "p10_qty":   percentile_map.get(10, (None, None))[1],
            "p30_price": percentile_map.get(30, (None, None))[0],
            "p30_qty":   percentile_map.get(30, (None, None))[1],
            "p50_price": percentile_map.get(50, (None, None))[0],
            "p50_qty":   percentile_map.get(50, (None, None))[1],
            "p70_price": percentile_map.get(70, (None, None))[0],
            "p70_qty":   percentile_map.get(70, (None, None))[1],
            "p90_price": percentile_map.get(90, (None, None))[0],
            "p90_qty":   percentile_map.get(90, (None, None))[1],
        })
    return records


def _ensure_column_order(df: pd.DataFrame) -> pd.DataFrame:
    """Garante a ordem com 'time' como 2ª coluna."""
    desired = [
        "date", "time", "item_id",
        "min_price", "median_price", "mean_price", "quantity",
        "p10_price", "p10_qty", "p30_price", "p30_qty",
        "p50_price", "p50_qty", "p70_price", "p70_qty",
        "p90_price", "p90_qty",
    ]
    # Adiciona colunas faltantes como vazias
    for col in desired:
        if col not in df.columns:
            df[col] = "" if col in ("time",) else None
    # Mantém colunas extras ao final (se existirem)
    extras = [c for c in df.columns if c not in desired]
    return df[desired + extras]


def update_csv(records: List[Dict[str, object]], *, csv_path: str) -> None:
    """Anexa registros ao CSV usando (date, time, item_id) como chave única
    e limpa linhas antigas sem 'time' quando houver registro com hora.
    """
    if not records:
        logger.info("Nenhum registro para salvar.")
        return

    df_new = pd.DataFrame.from_records(records)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Se CSV antigo não tiver 'time', cria coluna vazia
        if "time" not in df_old.columns:
            df_old["time"] = ""
        # Mesma coisa para colunas que possam estar faltando
        df_old = _ensure_column_order(df_old)
        df_new = _ensure_column_order(df_new)

        # Dedup incremental: só considera novas linhas que não existam pela chave (date,time,item_id)
        new_idx = ~df_new.set_index(["date", "time", "item_id"]).index.isin(
            df_old.set_index(["date", "time", "item_id"]).index
        )
        df_add = df_new[new_idx]
        appended_rows = len(df_add)

        df_concat = pd.concat([df_old, df_add], ignore_index=True)
    else:
        df_concat = _ensure_column_order(df_new)
        appended_rows = len(df_concat)

    # Remover linhas com time vazio quando existir outra linha (mesma data+item) com time preenchido
    if "time" in df_concat.columns:
        # Trata NaN como vazio
        time_series = df_concat["time"].astype(str)
        mask_blank_time = time_series.str.strip().isin(["", "nan", "None"])
        groups = df_concat.groupby(["date", "item_id"])["time"]
        has_non_blank = groups.transform(lambda x: x.astype(str).str.strip().isin(["", "nan", "None"]).eq(False).any())
        to_drop = mask_blank_time & has_non_blank
        if to_drop.any():
            dropped = int(to_drop.sum())
            logger.info(f"Removendo {dropped} linha(s) antiga(s) sem hora, onde já existe registro com hora.")
            df_concat = df_concat.loc[~to_drop].copy()

    # Remover duplicatas exatas por (date,time,item_id)
    before = len(df_concat)
    df_concat.drop_duplicates(subset=["date", "time", "item_id"], keep="last", inplace=True)
    after = len(df_concat)
    if after < before:
        logger.info(f"Removidas {before - after} duplicata(s) por (date,time,item_id).")

    # Ordenação e ordem de colunas final
    df_concat.sort_values(by=["item_id", "date", "time"], inplace=True)
    df_concat = _ensure_column_order(df_concat)

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    df_concat.to_csv(csv_path, index=False)

    logger.info(f"Foram adicionadas {appended_rows} linha(s) novas ao arquivo {csv_path}")


def main() -> None:
    logger.info(f"Iniciando a coleta para {len(ITEM_IDS)} item(ns): {', '.join(ITEM_IDS)}")
    all_records: List[Dict[str, object]] = []

    for item_id in ITEM_IDS:
        logger.info(f"Processando {item_id}")
        try:
            item_data = fetch_item_data(item_id, local_dir=LOCAL_JSON_DIR)
        except Exception as exc:
            logger.error(f"{item_id}: erro ao buscar dados -> {exc}")
            continue

        recs = extract_devaloka_records(item_data, server="devaloka")
        if not recs:
            logger.warning(f"{item_id}: não há dados de Devaloka")
            continue

        logger.info(f"{item_id}: {len(recs)} registro(s) extraído(s)")
        all_records.extend(recs)

    if all_records:
        logger.info(f"Salvando {len(all_records)} registro(s) ao CSV")
    update_csv(all_records, csv_path=CSV_PATH)
    logger.info("Coleta concluída.")


if __name__ == "__main__":
    main()
