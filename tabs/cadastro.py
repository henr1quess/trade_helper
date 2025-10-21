from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ITEMS_PATH = Path("items.json")  # pode trocar para onde você preferir

def _load_items():
    try:
        with open(ITEMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def _save_items(items):
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def render():
    st.subheader("🗂️ Cadastro de Itens (peso/stack)")

    items = _load_items()

    with st.expander("Adicionar/editar item", expanded=True):
        c1, c2 = st.columns([2,1])
        with c1:
            item_name = st.text_input("Nome do item (NWDB exato)", key="cad_item_name")
            categoria = st.text_input("Categoria (ex.: Ore, Raw Hide, ...)", key="cad_categoria")
            slug      = st.text_input("Slug (gaming.tools, opcional)", key="cad_slug")
        with c2:
            peso      = st.number_input("Peso (kg)", value=0.100, step=0.001, format="%.3f")
            stack_max = st.number_input("Max stack", value=1000, step=1)

        if st.button("Salvar item", type="primary"):
            if not item_name.strip():
                st.error("Informe o nome do item.")
            else:
                # atualiza se já existe, senão adiciona
                idx = next((i for i,x in enumerate(items) if x.get("item")==item_name), None)
                obj = {"item": item_name, "categoria": categoria, "peso": float(peso), "stack_max": int(stack_max)}
                if slug.strip(): obj["slug"] = slug.strip()
                if idx is None:
                    items.append(obj)
                else:
                    items[idx] = obj
                _save_items(items)
                st.success("Item salvo.")

    st.divider()
    if items:
        df = pd.DataFrame(items)
        st.dataframe(df, use_container_width=True, height=420)
    else:
        st.caption("Nenhum item cadastrado ainda.")
