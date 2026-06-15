#!/usr/bin/env bash
set -euo pipefail

NOME="joao-pedro-barbosa-t04"
ZIP="${NOME}.zip"
TMP=$(mktemp -d)

echo "==> Copiando entregaveis para $TMP/$NOME ..."
mkdir -p "$TMP/$NOME/results"
mkdir -p "$TMP/$NOME/figures"

cp main.ipynb        "$TMP/$NOME/"
cp main.html         "$TMP/$NOME/"

# Todos os arquivos de results (JSONs de resposta e scores)
cp results/*.json    "$TMP/$NOME/results/" 2>/dev/null || true

# Mapas e figuras
cp figures/*         "$TMP/$NOME/figures/" 2>/dev/null || true

echo "==> Gerando $ZIP ..."
(cd "$TMP" && zip -r - "$NOME") > "$ZIP"

rm -rf "$TMP"
echo "==> Pronto: $ZIP ($(du -sh "$ZIP" | cut -f1))"