from __future__ import annotations
from pathlib import Path
import json
import streamlit as st

# Importes das abas (sem import_manual)
from tabs import opportunities, collect, calculadora, cadastro

st.set_page_config(page_title="New World Helper", page_icon="🪙", layout="wide")

PROJECT_DIR = Path.cwd()
RAW_DIR = PROJECT_DIR / "raw"
LATEST_SNAPSHOT_PATH = RAW_DIR / "latest_snapshot.json"
LAST_META_PATH = RAW_DIR / "last_sync_meta.json"

def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Agora são só 4 abas
tab_opps, tab_collect, tab_calc, tab_cad = st.tabs(
    ["📈 Oportunidades", "☁️ Coletar snapshot", "🧮 Calculadora", "🗂️ Cadastro"]
)

with tab_opps:
    try:
        opportunities.render(LATEST_SNAPSHOT_PATH, LAST_META_PATH)
    except Exception as e:
        st.error("Falha ao renderizar a aba Oportunidades.")
        st.exception(e)

with tab_collect:
    try:
        collect.render()
    except Exception as e:
        st.error("Falha ao renderizar a aba Coletar.")
        st.exception(e)

with tab_calc:
    try:
        calculadora.render()
    except Exception as e:
        st.error("Falha ao renderizar a aba Calculadora.")
        st.exception(e)

with tab_cad:
    try:
        cadastro.render()
    except Exception as e:
        st.error("Falha ao renderizar a aba Cadastro.")
        st.exception(e)
