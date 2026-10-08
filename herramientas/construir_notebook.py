"""Convierte fuente/nfl_apuestas.py (formato de celdas '# %%') en el notebook de Colab.

Uso:  python herramientas/construir_notebook.py
"""
from pathlib import Path
import re
import nbformat

RAIZ = Path(__file__).resolve().parent.parent
FUENTE = RAIZ / 'fuente' / 'nfl_apuestas.py'
DESTINO = RAIZ / 'NFL_Apuestas_2026.ipynb'


def construir():
    texto = FUENTE.read_text(encoding='utf-8')
    bloques = re.split(r'^# %%(.*)$', texto, flags=re.M)
    celdas = []
    for cabecera, cuerpo in zip(bloques[1::2], bloques[2::2]):
        cuerpo = cuerpo.strip('\n')
        if '[markdown]' in cabecera:
            md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in cuerpo.splitlines())
            celdas.append(nbformat.v4.new_markdown_cell(md))
        else:
            celdas.append(nbformat.v4.new_code_cell(cuerpo))
    for i, celda in enumerate(celdas):
        celda['id'] = f'celda-{i:02d}'          # IDs estables: diffs limpios al regenerar
    nb = nbformat.v4.new_notebook(cells=celdas)
    nb.metadata = {'colab': {'provenance': [], 'toc_visible': True},
                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                   'language_info': {'name': 'python'}}
    nbformat.write(nb, DESTINO)
    print(f'{DESTINO.name}: {len(celdas)} celdas')


if __name__ == '__main__':
    construir()
