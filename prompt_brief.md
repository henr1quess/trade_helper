# Brief: Prompts lado a lado na aba "Cadastro"

## Onde mexer
- **Arquivo:** `streamlit_app.py`
- **Aba:** "Cadastro"
- **Seção alvo:** inserir imediatamente acima do bloco "Importar cadastro (JSON/CSV)" (ou dentro do mesmo bloco "Prompts para IA", se já existir).

## Entregáveis

### 1. Layout em duas colunas
- Criar `col1, col2 = st.columns(2)`.
- **col1** exibe `st.text_area` somente leitura com label `Prompt IA (cadastro de itens)`.
- **col2** exibe `st.text_area` somente leitura com label `Prompt IA (gerar slugs NWMP)`.
- Cada coluna deve conter:
  - `st.text_area(..., disabled=True, label_visibility="visible", height=~200)`.
  - Botão "Copiar" logo abaixo que envia o conteúdo da respectiva caixa para o clipboard.

### 2. Auto-copy
- Reutilizar helper existente para copiar (se já houver em outra parte do app).
- Caso não exista, criar helper mínimo com `st.components.v1.html` que renderize:
  - Botão com `id` único por prompt (`copy-cadastro`, `copy-slugs`).
  - `<script>` que receba o texto e use `navigator.clipboard.writeText(text)`.
  - Feedback: toast simples (JS) ou `st.toast` via `st.session_state`.

### 3. Textos fixos
Os prompts são fixos e sempre visíveis. Colocar exatamente estes conteúdos:

#### Prompt IA (cadastro de itens)
```
A partir da imagem/lista com nomes de itens do jogo New World, produza um JSON com os campos:
[
  {
    "item": "<nome exato in-game>",
    "categoria": "<categoria navegando no NWDB, ex.: Raw Hide, Ore, Wood, Cooking Ingredient, Gem, etc.>",
    "peso": <peso por unidade, número>,
    "stack_max": <tamanho máximo de pilha, número ou null>,
    "tags": [],               // opcional: lista de tags livres
    "tier": null              // opcional: T1–T5 (ou null se não aplicável)
  }
]

Regras:
- Use o NWDB para confirmar PESO e STACK MAX; se não achar, deixe null e preserve o item.
- “categoria” deve refletir a pasta onde o item aparece ao navegar no NWDB (ex.: Resources > Raw Hide).
- Não invente: se não encontrou algum dado, retorne null para aquele campo.
- Somente os campos listados; sem extras.
- JSON válido, pronto para colar no app.
```

#### Prompt IA (gerar slugs NWMP)
```
A partir da imagem ou lista de nomes de itens do jogo New World, retorne um JSON no formato:
[
  {"item": "Runewood", "slug_nwmp": "woodt52"},
  {"item": "Powerful Oakflesh Balm", "slug_nwmp": "oakfleshbalmt5"}
]

Regras:
- "slug_nwmp" deve ser o identificador usado no Gaming Tools (nwmp.gaming.tools),
  o mesmo que aparece em https://nwmp.gaming.tools/items/<slug>.
- Pesquise o slug correto (Gaming Tools e/ou NWDB como apoio).
- Se o item não existir, use null em "slug_nwmp" e mantenha o nome original.
- Não inclua campos além de item e slug_nwmp.
- JSON válido, pronto para colar no app.
```

## Requisitos funcionais
1. Duas colunas lado a lado em telas ≥ 1100px; podem empilhar em telas estreitas.
2. Botões "Copiar" funcionam em Chrome/Edge/Firefox modernos (usar `navigator.clipboard.writeText`).
3. Sem expansores/popovers — prompts sempre visíveis.
4. `st.text_area` com `disabled=True`.
5. `key`s exclusivas (ex.: `prompt_cadastro_txt`, `prompt_slugs_txt`) e `id`s exclusivos nos botões.

## Checklist de implementação
1. Localizar a seção dos prompts na aba Cadastro.
2. Inserir o layout com `st.columns(2)` e as duas text areas.
3. Implementar/usar helper de cópia com `st.components.v1.html`.
4. Conectar cada botão "Copiar" ao texto correspondente.
5. Testar copiando e colando os prompts.
6. Validar responsividade e preservar o layout existente.
