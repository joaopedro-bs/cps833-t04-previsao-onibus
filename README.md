# T04 — Previsão de Localização de Ônibus

**Disciplina:** CPS833 — Data Mining · PESC/COPPE-UFRJ  
**Professor:** Geraldo Zimbrão  
**Aluno:** Joao Pedro Barbosa Martins

Previsão de ETA e posição geográfica de ônibus do Rio de Janeiro a partir de histórico GPS, usando DuckDB, scikit-learn e haversine.

---

## Estrutura de Pastas

```
t04_previsao-onibus/
├── data/
│   ├── raw/
│   │   ├── treino/          # ZIPs de GPS histórico (2024-04-25 a 2024-05-10)
│   │   ├── teste/           # ZIPs de GPS dos dias de avaliação (2024-05-11 a 2024-05-15)
│   │   └── teste-final/     # ZIPs do período final de avaliação (2024-05-16 a 2024-05-20)
│   └── processed/
│       └── bus.duckdb       # Banco analítico gerado pelo pipeline de ingestão
├── figures/                 # Mapas Folium e gráficos gerados pelo notebook (não versionados)
├── results/                 # Arquivos de resposta e scores da API
│   ├── resposta-YYYY-MM-DD-v<N>.json   # Previsões submetidas (versionadas)
│   └── api_scores.json                 # Histórico de scores retornados pela API
├── src/
│   ├── ingest.py            # Carga e normalização dos JSONs → DuckDB
│   ├── clean.py             # Filtros de qualidade (geocerca, velocidade, timestamp)
│   ├── routes.py            # Extração de trajetos canônicos por linha
│   ├── direction.py         # Sentido do veículo e detecção de terminais
│   ├── speed_model.py       # Modelo de velocidade por segmento × faixa horária
│   ├── predict.py           # Predição ETA (T1) e posição (T2)
│   └── utils.py             # Conversões, haversine, helpers
├── main.ipynb               # Notebook único end-to-end (executar do início ao fim)
├── main.html                # Export HTML do notebook com todas as saídas
├── requirements.txt         # Dependências fixadas
└── .gitignore
```

---

## Dados

Os arquivos de dados **não estão no repositório** (exceto a estrutura de pastas). Devem ser obtidos via Google Drive e colocados nos diretórios abaixo:

| Diretório | Conteúdo | Formato |
|---|---|---|
| `data/raw/treino/` | GPS histórico de treinamento | `*.zip` contendo JSONs diários |
| `data/raw/teste/` | GPS dos dias de avaliação intermediária | `*.zip` contendo JSONs |
| `data/raw/teste-final/` | GPS do período de avaliação final | `*.zip` contendo JSONs |
| `data/processed/` | Banco DuckDB gerado automaticamente | `bus.duckdb` (gerado pelo notebook) |

> Os ZIPs não devem ser descompactados manualmente — o módulo `src/ingest.py` faz isso durante a ingestão.

---

## Configuração do Ambiente

```bash
# 1. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Instalar Jupyter (não está no requirements.txt pois é ferramenta de execução)
pip install jupyter nbconvert
```

---

## Como Rodar

```bash
# Executar o notebook do início ao fim (restart kernel + run all)
.venv/bin/jupyter nbconvert --to notebook --execute main.ipynb \
    --output main.ipynb \
    --ExecutePreprocessor.timeout=600

# Gerar o HTML de entrega
.venv/bin/jupyter nbconvert --to html main.ipynb --output main.html
```

Ou abrir interativamente:

```bash
.venv/bin/jupyter notebook main.ipynb
```

---

## Submissão para a API

As previsões são submetidas ao endpoint:

```
POST https://barra.cos.ufrj.br:443/datamining/rpc/avalia
```

O arquivo de resposta gerado fica em `results/resposta-YYYY-MM-DD-v<N>.json`. O histórico de scores retornados pela API é salvo em `results/api_scores.json`.

---

## Linhas Modeladas

50 linhas do sistema BRT/ônibus do Rio de Janeiro:

```
483, 864, 639, 3, 309, 774, 629, 371, 397, 100, 838, 315, 624, 388,
918, 665, 328, 497, 878, 355, 138, 606, 457, 550, 803, 917, 638,
2336, 399, 298, 867, 553, 565, 422, 756, 186012003, 292, 554, 634,
232, 415, 2803, 324, 852, 557, 759, 343, 779, 905, 108
```
