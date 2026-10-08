# %% [markdown]
# # 🏈 Asistente de análisis de apuestas NFL — temporada 2026
#
# **Qué hace este cuaderno:** descarga datos reales y actualizados de la NFL, estima probabilidades con varios modelos
# estadísticos validados en el tiempo, las compara con las cuotas de las casas de apuestas y te muestra, en un panel
# sencillo, **solo** las oportunidades con respaldo estadístico. Si no hay ninguna, te lo dice claramente: **NO APOSTAR**.
#
# > ⚠️ **Aviso importante.** Esto es una herramienta de análisis, no una garantía. Ningún modelo acierta siempre.
# > Las apuestas implican riesgo real de perder dinero. Nunca apuestes dinero que necesites para tus gastos.
# > Todas las cantidades que aparecen aquí son **hipotéticas**. Si el juego deja de ser un entretenimiento,
# > busca ayuda (en España: [jugarbien.es](https://www.jugarbien.es); en México: CIJ 55 5212 1212; en EE. UU.: 1-800-GAMBLER).
#
# ## ▶️ Cómo usarlo (3 pasos)
# 1. Menú **Entorno de ejecución → Ejecutar todas** (o `Ctrl+F9`). La primera vez tarda unos 5–10 minutos porque descarga
#    ~10 temporadas de datos jugada a jugada; las siguientes veces usa la copia guardada y es mucho más rápido.
# 2. Baja hasta la **celda 12 (Panel)**: allí verás el resumen semanal, la tabla de recomendaciones y las explicaciones.
# 3. Usa las listas desplegables para cambiar de partido o de semana. **No necesitas escribir código.**
#
# (Opcional) Para tener cuotas en vivo de varias casas, configura una clave gratuita de *The Odds API* en la celda 2.

# %% [markdown]
# ## 🧭 Arquitectura del sistema y fuentes de datos reales
#
# ```
#  ┌───────────────────────── FUENTES (reales y documentadas) ─────────────────────────┐
#  │ nflverse (GitHub releases, gratis, sin clave)                                     │
#  │   • schedules/games.csv  → calendario, resultados, descanso, QB titular, clima,   │
#  │                            estadio y CUOTAS (moneyline, spread, total + precios). │
#  │                            Para partidos jugados son líneas de cierre.            │
#  │   • pbp/play_by_play_AAAA.parquet → jugada a jugada: EPA, success rate, 3ª        │
#  │                            oportunidad, zona roja, pérdidas de balón, yardas.     │
#  │   • injuries/injuries_AAAA.csv → reporte oficial de lesiones (Out/Doubtful/...).  │
#  │ The Odds API (opcional, clave gratuita) → cuotas en vivo de varias casas.         │
#  │ CSV manual (opcional) → tus propias cuotas si no tienes clave.                    │
#  └───────────────────────────────────────────────────────────────────────────────────┘
#              │ descarga + caché + registro de fuente y fecha
#              ▼
#   Validación (semáforo de calidad) → Variables "pre-partido" (sin fuga de información)
#              ▼
#   Modelos: Elo · Regresión logística · Random Forest · HistGradientBoosting · XGBoost (si está) ·
#            Ridge de margen y de total de puntos  →  validación temporal (walk-forward por temporada)
#              ▼
#   Mezcla modelo/mercado + umbrales elegidos por backtesting (calibración 2019-21, prueba 2022+)
#              ▼
#   Valor esperado por mercado → Decisión (Apostar / Esperar / No apostar / Problema de datos)
#              ▼
#   Panel visual · Análisis por partido · Simulador de riesgo · Registro CSV + calibración
# ```
#
# **Lo que NO se usa y por qué:** no se raspan (scraping) webs de casas de apuestas (va contra sus términos); no se
# inventan datos faltantes; las cuotas históricas disponibles gratis son solo de **cierre** (no de apertura), así que el
# backtest es conservador y el *CLV* histórico no puede medirse (sí se mide en vivo en el registro, celda 14).

# %% [markdown]
# ## 1️⃣ Instalación de dependencias
# **Qué hace:** comprueba las librerías e instala las que falten. **Qué esperar:** un mensaje "✅ Dependencias listas".
#
# **Errores comunes:** si ves un error de red, vuelve a ejecutar la celda. Si Colab pide "Reiniciar sesión", acepta y
# ejecuta de nuevo *Ejecutar todas*.

# %%
import sys, subprocess, importlib.util

EN_COLAB = 'google.colab' in sys.modules
_necesarias = {'pandas': 'pandas', 'numpy': 'numpy', 'sklearn': 'scikit-learn', 'plotly': 'plotly',
               'ipywidgets': 'ipywidgets', 'pyarrow': 'pyarrow', 'requests': 'requests',
               'matplotlib': 'matplotlib', 'scipy': 'scipy', 'joblib': 'joblib'}
_faltan = [pip for mod, pip in _necesarias.items() if importlib.util.find_spec(mod) is None]
if _faltan:
    print('Instalando:', ', '.join(_faltan))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_faltan], check=False)

XGBOOST_DISPONIBLE = importlib.util.find_spec('xgboost') is not None

import plotly.io as pio
if EN_COLAB:
    pio.renderers.default = 'colab'
print('✅ Dependencias listas' + ('' if XGBOOST_DISPONIBLE else ' (XGBoost no disponible: se usará HistGradientBoosting)'))

# %% [markdown]
# ## 2️⃣ Configuración: fuentes, claves API y reglas de prudencia
#
# **No necesitas cambiar nada para empezar.** Los valores por defecto son conservadores.
#
# ### 🔑 (Opcional) Cuotas en vivo con *The Odds API*
# 1. Crea una cuenta gratuita en <https://the-odds-api.com> (plan gratuito: 500 consultas/mes; cada actualización de
#    este cuaderno gasta 3).
# 2. En Colab, pulsa el icono de la **llave 🔑 (Secretos)** en la barra izquierda → *Añadir secreto nuevo*.
# 3. Nombre: `ODDS_API_KEY` · Valor: tu clave · Activa "Acceso al cuaderno".
#
# Así la clave **nunca queda escrita en el cuaderno**. Si no la configuras, el sistema usa las líneas publicadas por
# nflverse (de referencia, pueden tener horas de retraso) y lo indica en el panel con menor confianza.
#
# ### 📄 (Opcional) Cuotas propias por CSV
# Pon `USAR_CSV_MANUAL = True`; en la celda 10 aparecerá un botón para subir un archivo con este formato (se genera una
# plantilla descargable automáticamente):
#
# | local | visitante | mercado | lado | linea | cuota | casa | fecha_hora |
# |---|---|---|---|---|---|---|---|
# | DAL | TB | moneyline | local | | 1.25 | MiCasa | 2026-10-08 18:00 |
# | DAL | TB | spread | local | -7.5 | 1.91 | MiCasa | 2026-10-08 18:00 |
# | DAL | TB | total | over | 47.5 | 1.90 | MiCasa | 2026-10-08 18:00 |
#
# `cuota` acepta formato decimal (1.91) o americano (-110). `linea` del spread es el hándicap del lado indicado.

# %%
import os, datetime as dt
from pathlib import Path

CONFIG = {
    # --- Temporadas ---
    'TEMPORADA_ACTUAL': 'auto',          # 'auto' = según la fecha de hoy; o un año, p. ej. 2026
    'PRIMERA_TEMPORADA_PBP': 2016,       # desde qué año se descargan datos jugada a jugada (variables)
    'PRIMERA_TEMPORADA_ELO': 2002,       # el Elo necesita historia larga para estabilizarse
    'PRIMERA_TEMPORADA_TEST': 2019,      # validación temporal: se evalúa de aquí en adelante
    'TEMPORADAS_CALIBRACION': [2019, 2020, 2021],  # se usan para ELEGIR umbrales y mezcla
    # (las temporadas posteriores son de PRUEBA: confirman o descartan lo elegido)

    # --- Fuentes ---
    'USAR_ODDS_API': True,               # solo funciona si existe el secreto ODDS_API_KEY
    'USAR_CSV_MANUAL': False,
    'REGION_ODDS_API': 'us',             # 'us', 'eu', 'uk' o 'au'
    'HORAS_CACHE_ACTUAL': 6,             # cada cuánto se vuelven a descargar los datos de la temporada en curso

    # --- Filtros de decisión (los umbrales de valor esperado se eligen en el backtesting, celda 8) ---
    'CONFIANZA_MINIMA': 60,              # 0-100; por debajo, la oportunidad pasa a "Esperar"
    'VENTAJA_MINIMA': 0.02,              # diferencia mínima modelo-mercado (2 puntos porcentuales)
    'CUOTA_MAXIMA': 4.0,                 # no se recomiendan cuotas mayores (más varianza, más error de modelo)
    'EXIGIR_RESPALDO_BACKTEST': True,    # solo "Apostar" en mercados que fueron rentables en calibración Y prueba
    'MIN_APUESTAS_BACKTEST': 30,

    # --- Gestión del capital (todo es HIPOTÉTICO) ---
    'MOSTRAR_MONTOS_SUGERIDOS': True,    # pon False para ocultar cualquier importe sugerido
    'CAPITAL_HIPOTETICO': 1000.0,
    'FRACCION_KELLY': 0.25,              # Kelly fraccional (1/4): mucho más prudente que Kelly completo
    'TOPE_POR_APUESTA': 0.02,            # nunca más del 2 % del capital en una apuesta
    'TOPE_SEMANAL': 0.08,                # nunca más del 8 % del capital en total en una semana

    # --- Registro ---
    'GUARDAR_EN_DRIVE': False,           # True = guarda caché, modelos y registro en tu Google Drive
    'CARPETA_TRABAJO': '/content/nfl_apuestas' if EN_COLAB else 'nfl_apuestas',
    'MIN_PARTIDOS_NUEVOS_PARA_REENTRENAR': 16,
}

HOY = dt.datetime.now(dt.timezone.utc)
TEMPORADA = (HOY.year if HOY.month >= 3 else HOY.year - 1) if CONFIG['TEMPORADA_ACTUAL'] == 'auto' \
    else int(CONFIG['TEMPORADA_ACTUAL'])

if CONFIG['GUARDAR_EN_DRIVE'] and EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CONFIG['CARPETA_TRABAJO'] = '/content/drive/MyDrive/nfl_apuestas'
CARPETA = Path(CONFIG['CARPETA_TRABAJO'])
for sub in ('cache', 'modelos', 'registro'):
    (CARPETA / sub).mkdir(parents=True, exist_ok=True)


def obtener_secreto(nombre):
    """Lee una clave de los Secretos de Colab o de una variable de entorno. Nunca la imprime."""
    try:
        from google.colab import userdata
        valor = userdata.get(nombre)
        if valor:
            return valor
    except Exception:
        pass
    return os.environ.get(nombre) or None


ODDS_API_KEY = obtener_secreto('ODDS_API_KEY') if CONFIG['USAR_ODDS_API'] else None
print(f'📅 Temporada analizada: {TEMPORADA}')
print(f'📁 Carpeta de trabajo: {CARPETA.resolve()}')
print('🔑 The Odds API:', 'clave encontrada ✅' if ODDS_API_KEY else
      'sin clave → se usarán líneas de referencia de nflverse (ver instrucciones arriba)')

# %% [markdown]
# ## 3️⃣ Descarga de datos
# **Qué hace:** descarga calendario, cuotas, datos jugada a jugada y lesiones desde nflverse. Guarda una copia (caché) para
# no repetir descargas: las temporadas pasadas se descargan una sola vez; la actual se refresca cada pocas horas.
#
# **Qué esperar:** una tabla de fuentes con fecha de publicación, fecha de descarga y número de filas.
#
# **Errores comunes:** `HTTPError 404` en la temporada actual antes de que empiece = normal (aún no hay jugadas).
# Errores de red: vuelve a ejecutar; si había copia guardada se usará y se avisará.

# %%
import io, json, time, math, hashlib, warnings
import numpy as np
import pandas as pd
import requests
from IPython.display import display, HTML

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
CACHE = CARPETA / 'cache'
NFLVERSE = 'https://github.com/nflverse/nflverse-data/releases/download'
FUENTES = []          # registro de cada fuente usada: nombre, url, fechas, filas, estado
AVISOS_DATOS = []     # avisos que se mostrarán en el panel


def descargar(url, nombre_local, max_horas, obligatorio=True):
    """Descarga con caché. Devuelve (ruta, info) o (None, info) si no se pudo y no es obligatorio."""
    destino = CACHE / nombre_local
    meta_ruta = CACHE / (nombre_local + '.meta.json')
    meta = json.loads(meta_ruta.read_text()) if meta_ruta.exists() else {}
    if destino.exists() and (time.time() - destino.stat().st_mtime) / 3600 < max_horas:
        return destino, {**meta, 'estado': 'copia guardada (vigente)'}
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        tmp = destino.with_suffix('.descargando')
        tmp.write_bytes(r.content)
        tmp.replace(destino)
        meta = {'url': url, 'descargado': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                'publicado': r.headers.get('Last-Modified', 'no informado')}
        meta_ruta.write_text(json.dumps(meta))
        return destino, {**meta, 'estado': 'descargado ahora'}
    except Exception as e:
        if destino.exists():
            AVISOS_DATOS.append(f'No se pudo actualizar {nombre_local} ({e.__class__.__name__}); se usa la copia guardada.')
            return destino, {**meta, 'estado': '⚠️ copia guardada ANTIGUA (falló la actualización)'}
        if obligatorio:
            raise RuntimeError(f'No se pudo descargar {url}: {e}. Revisa tu conexión y vuelve a ejecutar.') from e
        return None, {'url': url, 'estado': f'no disponible ({e.__class__.__name__})'}


def registrar(nombre, info, filas, dato_mas_reciente=''):
    FUENTES.append({'Fuente': nombre, 'Estado': info.get('estado', ''), 'Publicado por la fuente': info.get('publicado', ''),
                    'Descargado': info.get('descargado', ''), 'Filas': filas, 'Dato más reciente': dato_mas_reciente,
                    'URL': info.get('url', '')})


# --- Calendario, resultados y cuotas de referencia ---
ruta, info = descargar(f'{NFLVERSE}/schedules/games.csv', 'games.csv', CONFIG['HORAS_CACHE_ACTUAL'])
calendario = pd.read_csv(ruta, low_memory=False)
_jug = calendario[calendario['home_score'].notna()]
registrar('nflverse · calendario, resultados y cuotas', info, len(calendario),
          f"último partido jugado: {_jug['gameday'].max()}")

# --- Jugada a jugada (solo las columnas necesarias) ---
COLUMNAS_PBP = ['game_id', 'season', 'week', 'posteam', 'defteam', 'play_type', 'epa', 'success', 'down',
                'yardline_100', 'third_down_converted', 'interception', 'fumble_lost', 'pass', 'yards_gained',
                'qb_dropback', 'passer_player_id', 'qb_epa', 'fixed_drive', 'fixed_drive_result']
_partes = []
for temporada in range(CONFIG['PRIMERA_TEMPORADA_PBP'], TEMPORADA + 1):
    actual = temporada == TEMPORADA
    ruta, info = descargar(f'{NFLVERSE}/pbp/play_by_play_{temporada}.parquet', f'pbp_{temporada}.parquet',
                           CONFIG['HORAS_CACHE_ACTUAL'] if actual else 24 * 365 * 20, obligatorio=not actual)
    if ruta is None:
        AVISOS_DATOS.append(f'Sin datos jugada a jugada de {temporada} todavía.')
        registrar(f'nflverse · jugada a jugada {temporada}', info, 0)
        continue
    df = pd.read_parquet(ruta, columns=COLUMNAS_PBP)
    _partes.append(df)
    registrar(f'nflverse · jugada a jugada {temporada}', info, len(df),
              f"semana {int(df['week'].max())}" if len(df) else '')
    print(f'  jugada a jugada {temporada}: {len(df):,} jugadas ({info["estado"]})')
pbp = pd.concat(_partes, ignore_index=True)
del _partes

# --- Reportes de lesiones ---
_partes = []
for temporada in range(CONFIG['PRIMERA_TEMPORADA_PBP'], TEMPORADA + 1):
    actual = temporada == TEMPORADA
    ruta, info = descargar(f'{NFLVERSE}/injuries/injuries_{temporada}.csv', f'injuries_{temporada}.csv',
                           CONFIG['HORAS_CACHE_ACTUAL'] if actual else 24 * 365 * 20, obligatorio=False)
    if ruta is None:
        AVISOS_DATOS.append(f'Sin reporte de lesiones de {temporada}.')
        continue
    df = pd.read_csv(ruta, low_memory=False,
                     usecols=lambda c: c in ['season', 'game_type', 'team', 'week', 'gsis_id', 'position',
                                             'full_name', 'report_status'])
    _partes.append(df)
    if actual:
        registrar(f'nflverse · lesiones {temporada}', info, len(df),
                  f"semana {int(df['week'].max())}" if len(df) else '')
lesiones = pd.concat(_partes, ignore_index=True) if _partes else pd.DataFrame(
    columns=['season', 'game_type', 'team', 'week', 'gsis_id', 'position', 'full_name', 'report_status'])
del _partes

print('\n📚 Fuentes utilizadas:')
display(pd.DataFrame(FUENTES).drop(columns='URL'))

# %% [markdown]
# ## 4️⃣ Validación y limpieza
# **Qué hace:** unifica nombres de equipos que se mudaron (OAK→LV, SD→LAC, STL→LA), comprueba que los datos estén
# completos y actualizados, y muestra un **semáforo de calidad**. Nada se rellena con cifras inventadas: lo que falta
# queda marcado y reduce la confianza de las predicciones afectadas.

# %%
MUDANZAS = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}

cal = calendario[calendario['season'] >= CONFIG['PRIMERA_TEMPORADA_ELO']].copy()
cal['home_orig'], cal['away_orig'] = cal['home_team'], cal['away_team']
cal['home_team'] = cal['home_team'].replace(MUDANZAS)
cal['away_team'] = cal['away_team'].replace(MUDANZAS)
cal['gameday'] = pd.to_datetime(cal['gameday'])
cal['jugado'] = cal['home_score'].notna() & cal['away_score'].notna()
cal['neutral'] = (cal['location'] == 'Neutral').astype(int)
cal['playoff'] = (cal['game_type'] != 'REG').astype(int)
cal = cal.sort_values(['gameday', 'game_id']).reset_index(drop=True)

for col in ('posteam', 'defteam'):
    pbp[col] = pbp[col].replace(MUDANZAS)
lesiones['team'] = lesiones['team'].replace(MUDANZAS)

controles = []


def control(nombre, ok, detalle, grave=False):
    controles.append({'Control': nombre, 'Estado': '🟢 OK' if ok else ('🔴 Problema' if grave else '🟡 Revisar'),
                      'Detalle': detalle})
    if not ok:
        AVISOS_DATOS.append(f'{nombre}: {detalle}')


equipos = sorted(set(cal['home_team']) | set(cal['away_team']))
control('32 equipos reconocidos', len(equipos) == 32, f'{len(equipos)} equipos encontrados', grave=True)
control('Identificadores de partido únicos', cal['game_id'].is_unique, 'sin duplicados' if cal['game_id'].is_unique
        else 'hay partidos duplicados', grave=True)
_inconsist = cal[cal['jugado'] & ((cal['home_score'] - cal['away_score']) != cal['result'])]
control('Resultados coherentes', _inconsist.empty, f'{len(_inconsist)} partidos con marcador incoherente')

cal_act = cal[cal['season'] == TEMPORADA]
_ult = cal_act.loc[cal_act['jugado'], 'gameday'].max()
_dias = (pd.Timestamp(HOY.date()) - _ult).days if pd.notna(_ult) else None
_hay_pendientes = (~cal_act['jugado']).any()
control('Resultados actualizados', _dias is None or _dias <= 10 or not _hay_pendientes,
        f'último resultado: {_ult.date() if pd.notna(_ult) else "ninguno"} (hace {_dias} días)')
# Partidos ya disputados (fecha pasada) pero sin resultado cargado
_sin_res = cal_act[(~cal_act['jugado']) & (cal_act['gameday'] < pd.Timestamp(HOY.date()) - pd.Timedelta(days=1))]
control('Sin partidos pasados pendientes de resultado', _sin_res.empty,
        f'{len(_sin_res)} partidos ya disputados aún sin resultado en la fuente')

_jugados_pbp = cal[cal['jugado'] & (cal['season'] >= CONFIG['PRIMERA_TEMPORADA_PBP'])]['game_id']
_faltan_pbp = sorted(set(_jugados_pbp) - set(pbp['game_id'].unique()))
control('Jugada a jugada completo', len(_faltan_pbp) == 0,
        f'{len(_faltan_pbp)} partidos jugados sin datos jugada a jugada' + (f' (p. ej. {_faltan_pbp[-3:]})' if _faltan_pbp else ''))

_prox = cal_act[~cal_act['jugado']]
_sin_cuota = _prox[_prox['home_moneyline'].isna()]
control('Cuotas de referencia en próximos partidos', _prox.empty or len(_sin_cuota) < len(_prox),
        f'{len(_prox) - len(_sin_cuota)} de {len(_prox)} próximos partidos tienen líneas en nflverse')
_sem_les = lesiones.loc[lesiones['season'] == TEMPORADA, 'week'].max()
control('Reporte de lesiones de la temporada', pd.notna(_sem_les),
        f'último reporte: semana {int(_sem_les)}' if pd.notna(_sem_les) else 'no disponible')

display(HTML('<h3>🚦 Semáforo de calidad de datos</h3>'))
display(pd.DataFrame(controles))

# %% [markdown]
# ## 5️⃣ Ingeniería de variables (sin fuga de información)
# **Qué hace:** convierte los datos en medidas de fortaleza de cada equipo **tal como se conocían antes de cada
# partido**. Regla de oro: para un partido del domingo solo se usan partidos anteriores a ese domingo (todas las
# medias se calculan y luego se desplazan un partido hacia atrás).
#
# Variables principales:
# * **Elo** ajustado por localía y margen de victoria (con regresión a la media entre temporadas).
# * **EPA por jugada** (puntos esperados añadidos) y **success rate**, ofensivos y defensivos, con media móvil
#   exponencial (pesa más lo reciente) y **ajustados por la calidad del rival**.
# * Pase/carrera, yardas, pérdidas de balón, 3ª oportunidad, zona roja, puntos, forma de los últimos 3 partidos.
# * Rendimiento como local/visitante y contra rivales fuertes/débiles.
# * **Quarterback**: valor del QB titular (EPA por dropback de sus últimos 16 partidos, encogido hacia un nivel de
#   suplente cuando hay poca muestra), cambio de QB y su estado en el reporte de lesiones.
# * Lesionados relevantes (Out/Doubtful), días de descanso, kilómetros y husos horarios de viaje, viento y temperatura.

# %%
VIDA_MEDIA = 6          # partidos: media móvil exponencial (lo de hace 6 partidos pesa la mitad)
QB_PRIOR, QB_K = -0.10, 200   # nivel de un QB sin historial y "peso" de ese supuesto en dropbacks

# Coordenadas (lat, lon) y huso horario estándar (UTC) de cada sede, para distancias de viaje.
SEDES = {
    'ARI': (33.53, -112.26, -7), 'ATL': (33.76, -84.40, -5), 'BAL': (39.28, -76.62, -5), 'BUF': (42.77, -78.79, -5),
    'CAR': (35.23, -80.85, -5), 'CHI': (41.86, -87.62, -6), 'CIN': (39.10, -84.52, -5), 'CLE': (41.51, -81.70, -5),
    'DAL': (32.75, -97.09, -6), 'DEN': (39.74, -105.02, -7), 'DET': (42.34, -83.05, -5), 'GB': (44.50, -88.06, -6),
    'HOU': (29.68, -95.41, -6), 'IND': (39.76, -86.16, -5), 'JAX': (30.32, -81.64, -5), 'KC': (39.05, -94.48, -6),
    'LA': (33.95, -118.34, -8), 'LAC': (33.95, -118.34, -8), 'LV': (36.09, -115.18, -8), 'MIA': (25.96, -80.24, -5),
    'MIN': (44.97, -93.26, -6), 'NE': (42.09, -71.26, -5), 'NO': (29.95, -90.08, -6), 'NYG': (40.81, -74.07, -5),
    'NYJ': (40.81, -74.07, -5), 'PHI': (39.90, -75.17, -5), 'PIT': (40.45, -80.02, -5), 'SEA': (47.60, -122.33, -8),
    'SF': (37.40, -121.97, -8), 'TB': (27.98, -82.50, -5), 'TEN': (36.17, -86.77, -6), 'WAS': (38.91, -76.86, -5),
    'OAK': (37.75, -122.20, -8), 'SD': (32.78, -117.12, -8), 'STL': (38.63, -90.19, -6),
}


def distancia_km(a, b):
    (la1, lo1, _), (la2, lo2, _) = SEDES[a], SEDES[b]
    la1, lo1, la2, lo2 = map(math.radians, (la1, lo1, la2, lo2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


# ---------- 5.1 Elo ----------
def calcular_elo(cal, k=20, ventaja_local=48, regresion=1 / 3):
    rating, ult_temp, pre_h, pre_a = {}, {}, [], []
    for r in cal.itertuples(index=False):
        for t in (r.home_team, r.away_team):
            rating.setdefault(t, 1505.0)
            if ult_temp.get(t, r.season) != r.season:      # nueva temporada: regresión a la media
                rating[t] = rating[t] * (1 - regresion) + 1505 * regresion
            ult_temp[t] = r.season
        rh, ra = rating[r.home_team], rating[r.away_team]
        pre_h.append(rh); pre_a.append(ra)
        if r.jugado:
            diff = rh + (0 if r.neutral else ventaja_local) - ra
            esperado = 1 / (1 + 10 ** (-diff / 400))
            margen = r.home_score - r.away_score
            real = 1.0 if margen > 0 else 0.0 if margen < 0 else 0.5
            diff_ganador = diff if margen > 0 else -diff
            mult = math.log(abs(margen) + 1) * 2.2 / (diff_ganador * 0.001 + 2.2) if margen != 0 else 1.0
            cambio = k * mult * (real - esperado)
            rating[r.home_team] += cambio
            rating[r.away_team] -= cambio
    return np.array(pre_h), np.array(pre_a), rating


cal['elo_h'], cal['elo_a'], ELO_ACTUAL = calcular_elo(cal)
cal['elo_diff'] = cal['elo_h'] - cal['elo_a'] + np.where(cal['neutral'] == 1, 0, 48)
cal['p_elo'] = 1 / (1 + 10 ** (-cal['elo_diff'] / 400))
# Entorno anotador de la liga antes de cada partido (media móvil de puntos totales de partidos previos)
cal['liga_pts_pre'] = cal['total'].where(cal['jugado']).ewm(halflife=128, ignore_na=True).mean().shift(1).ffill()

# ---------- 5.2 Estadísticas por equipo y partido (jugada a jugada) ----------
jug = pbp[pbp['play_type'].isin(['pass', 'run']) & pbp['epa'].notna() & pbp['posteam'].notna()].copy()
jug['es_pase'] = jug['pass'] == 1
jug['perdida'] = jug['interception'].fillna(0) + jug['fumble_lost'].fillna(0)
g = jug.groupby(['game_id', 'posteam'])
ofe = pd.DataFrame({'jugadas': g.size(), 'epa': g['epa'].mean(), 'sr': g['success'].mean(),
                    'yardas': g['yards_gained'].sum(), 'perdidas': g['perdida'].sum()})
ofe['pass_epa'] = jug[jug['es_pase']].groupby(['game_id', 'posteam'])['epa'].mean()
ofe['rush_epa'] = jug[~jug['es_pase']].groupby(['game_id', 'posteam'])['epa'].mean()
_t = jug[jug['down'] == 3].groupby(['game_id', 'posteam'])['third_down_converted']
ofe['tercera'] = _t.sum() / _t.size()
_rz = (jug[jug['yardline_100'] <= 20].groupby(['game_id', 'posteam', 'fixed_drive'])['fixed_drive_result'].first())
ofe['zona_roja'] = (_rz == 'Touchdown').groupby(level=[0, 1]).sum() / _rz.groupby(level=[0, 1]).size()
ofe = ofe.reset_index().rename(columns={'posteam': 'team'})

# QB: dropbacks y EPA de cada pasador por partido
_qb = jug[(jug['qb_dropback'] == 1) & jug['passer_player_id'].notna()]
qb_partido = _qb.groupby(['game_id', 'passer_player_id']).agg(db=('qb_epa', 'size'), epa_sum=('qb_epa', 'sum')).reset_index()

# ---------- 5.3 Tabla equipo-partido ----------
_comunes = ['game_id', 'season', 'week', 'game_type', 'gameday', 'neutral', 'playoff', 'jugado', 'elo_h', 'elo_a']
_h = cal[_comunes].assign(team=cal['home_team'], opp=cal['away_team'], es_local=1, pts_for=cal['home_score'],
                          pts_against=cal['away_score'], rest=cal['home_rest'], qb_id=cal['home_qb_id'],
                          qb_name=cal['home_qb_name'], elo_opp=cal['elo_a'])
_a = cal[_comunes].assign(team=cal['away_team'], opp=cal['home_team'], es_local=0, pts_for=cal['away_score'],
                          pts_against=cal['home_score'], rest=cal['away_rest'], qb_id=cal['away_qb_id'],
                          qb_name=cal['away_qb_name'], elo_opp=cal['elo_h'])
eq = pd.concat([_h, _a], ignore_index=True).drop(columns=['elo_h', 'elo_a'])
eq['margin'] = eq['pts_for'] - eq['pts_against']
eq = eq.merge(ofe, on=['game_id', 'team'], how='left')
_def = ofe.rename(columns={'team': 'opp', **{c: 'def_' + c for c in ofe.columns if c not in ('game_id', 'team')}})
eq = eq.merge(_def, on=['game_id', 'opp'], how='left')
eq = eq.sort_values(['team', 'gameday', 'game_id']).reset_index(drop=True)


def pre_partido(df, col, grupo=('team',), metodo='ewm'):
    """Valor de `col` conocido ANTES de cada partido: se calcula con partidos previos ya jugados del mismo grupo."""
    grupo = list(grupo)
    validos = df[df[col].notna() & df['jugado']]
    if metodo == 'ewm':
        post = validos.groupby(grupo)[col].transform(lambda s: s.ewm(halflife=VIDA_MEDIA).mean())
    else:  # últimos 3 partidos
        post = validos.groupby(grupo)[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
    serie = pd.Series(np.nan, index=df.index)
    serie.loc[post.index] = post
    tmp = df[grupo].copy()
    tmp['v'] = serie
    tmp['v'] = tmp.groupby(grupo)['v'].shift(1)
    return tmp.groupby(grupo)['v'].ffill()


METRICAS = ['pts_for', 'pts_against', 'margin', 'epa', 'sr', 'pass_epa', 'rush_epa', 'yardas', 'perdidas',
            'tercera', 'zona_roja', 'jugadas', 'def_epa', 'def_sr', 'def_yardas', 'def_perdidas', 'def_tercera',
            'def_zona_roja']
for m in METRICAS:
    eq['pre_' + m] = pre_partido(eq, m)
eq['pre_l3_margin'] = pre_partido(eq, 'margin', metodo='ultimos3')
eq['pre_l3_epa'] = pre_partido(eq, 'epa', metodo='ultimos3')

# Ajuste por rival: rendimiento respecto a lo que ese rival solía permitir/producir antes del partido
_riv = eq[['game_id', 'team', 'pre_def_epa', 'pre_epa']].rename(
    columns={'team': 'opp', 'pre_def_epa': 'opp_pre_def_epa', 'pre_epa': 'opp_pre_epa'})
eq = eq.merge(_riv, on=['game_id', 'opp'], how='left').sort_values(['team', 'gameday', 'game_id']).reset_index(drop=True)
eq['epa_vs_esp'] = eq['epa'] - eq['opp_pre_def_epa']
eq['def_epa_vs_esp'] = eq['def_epa'] - eq['opp_pre_epa']
eq['pre_adj_off'] = pre_partido(eq, 'epa_vs_esp')
eq['pre_adj_def'] = pre_partido(eq, 'def_epa_vs_esp')
# Rendimiento según el contexto: como local/visitante y contra rivales fuertes/débiles (Elo >= 1550)
eq['rival_fuerte'] = (eq['elo_opp'] >= 1550).astype(int)
eq['pre_margen_sede'] = pre_partido(eq, 'margin', grupo=('team', 'es_local'))
eq['pre_margen_tipo_rival'] = pre_partido(eq, 'margin', grupo=('team', 'rival_fuerte'))
# Partidos jugados esta temporada antes del actual
eq['partidos_temp'] = eq.groupby(['team', 'season'])['jugado'].cumsum() - eq['jugado'].astype(int)

# ---------- 5.4 Quarterback ----------
eq['qb_confirmado'] = eq['qb_id'].notna()
eq['qb_id'] = eq.groupby('team')['qb_id'].ffill()            # sin dato: se asume el último titular conocido
eq['qb_name'] = eq.groupby('team')['qb_name'].ffill()
eq['qb_cambio'] = (eq['qb_id'] != eq.groupby('team')['qb_id'].shift(1)).astype(int)
_hist = qb_partido.merge(cal[['game_id', 'gameday']], on='game_id').sort_values(['passer_player_id', 'gameday'])
_hist['db16'] = _hist.groupby('passer_player_id')['db'].transform(lambda s: s.rolling(16, min_periods=1).sum())
_hist['epa16'] = _hist.groupby('passer_player_id')['epa_sum'].transform(lambda s: s.rolling(16, min_periods=1).sum())
_hist = _hist.groupby(['passer_player_id', 'gameday'], as_index=False)[['db16', 'epa16']].last()
_izq = eq[['qb_id', 'gameday']].reset_index().dropna(subset=['qb_id']).sort_values('gameday')
_q = pd.merge_asof(_izq, _hist.rename(columns={'passer_player_id': 'qb_id'}).sort_values('gameday'),
                   on='gameday', by='qb_id', allow_exact_matches=False).set_index('index')
eq['qb_db'] = _q['db16'].reindex(eq.index).fillna(0)
eq['qb_val'] = ((_q['epa16'].reindex(eq.index).fillna(0) + QB_PRIOR * QB_K) / (eq['qb_db'] + QB_K))

# ---------- 5.5 Lesiones ----------
_les = lesiones[lesiones['report_status'].isin(['Out', 'Doubtful'])
                & ~lesiones['position'].isin(['QB', 'K', 'P', 'LS'])].copy()
_les['peso'] = np.where(_les['report_status'] == 'Out', 1.0, 0.75)
_les_eq = _les.groupby(['season', 'week', 'team'])['peso'].sum().rename('lesionados').reset_index()
_semanas_con_reporte = lesiones[['season', 'week']].drop_duplicates().assign(reporte=1)
eq = eq.merge(_les_eq, on=['season', 'week', 'team'], how='left').merge(_semanas_con_reporte, on=['season', 'week'], how='left')
eq['reporte_lesiones'] = eq['reporte'].fillna(0).astype(int)
eq['lesionados'] = eq['lesionados'].where(eq['reporte_lesiones'] == 1, np.nan)
eq.loc[(eq['reporte_lesiones'] == 1) & eq['lesionados'].isna(), 'lesionados'] = 0.0
_qbles = lesiones[lesiones['position'] == 'QB'][['season', 'week', 'gsis_id', 'report_status']].rename(
    columns={'gsis_id': 'qb_id', 'report_status': 'qb_estado'}).drop_duplicates(['season', 'week', 'qb_id'])
eq = eq.merge(_qbles, on=['season', 'week', 'qb_id'], how='left').drop(columns='reporte')
eq = eq.sort_values(['team', 'gameday', 'game_id']).reset_index(drop=True)

# ---------- 5.6 Variables a nivel de partido ----------
VARS_EQUIPO = [c for c in eq.columns if c.startswith('pre_')] + [
    'qb_val', 'qb_db', 'qb_cambio', 'qb_confirmado', 'qb_estado', 'qb_id', 'qb_name', 'lesionados',
    'reporte_lesiones', 'partidos_temp', 'rest']
_H = eq[eq['es_local'] == 1].set_index('game_id')[VARS_EQUIPO].add_prefix('h_')
_A = eq[eq['es_local'] == 0].set_index('game_id')[VARS_EQUIPO].add_prefix('a_')
partidos = cal.set_index('game_id').join(_H).join(_A).reset_index()

for v in ['adj_off', 'adj_def', 'epa', 'def_epa', 'sr', 'def_sr', 'pass_epa', 'rush_epa', 'margin', 'l3_margin',
          'l3_epa', 'perdidas', 'def_perdidas', 'tercera', 'def_tercera', 'zona_roja', 'def_zona_roja', 'yardas',
          'def_yardas', 'pts_for', 'pts_against', 'margen_sede', 'margen_tipo_rival', 'jugadas']:
    partidos['d_' + v] = partidos['h_pre_' + v] - partidos['a_pre_' + v]
    partidos['s_' + v] = partidos['h_pre_' + v] + partidos['a_pre_' + v]
partidos['d_qb_val'] = partidos['h_qb_val'] - partidos['a_qb_val']
partidos['s_qb_val'] = partidos['h_qb_val'] + partidos['a_qb_val']
partidos['d_rest'] = partidos['home_rest'] - partidos['away_rest']
partidos['d_lesionados'] = partidos['h_lesionados'] - partidos['a_lesionados']
partidos['campo_local'] = 1 - partidos['neutral']
partidos['partidos_min'] = partidos[['h_partidos_temp', 'a_partidos_temp']].min(axis=1)
partidos['viaje_km'] = [np.nan if n else distancia_km(a, h)
                        for n, a, h in zip(partidos['neutral'], partidos['away_orig'], partidos['home_orig'])]
partidos['husos'] = [np.nan if n else SEDES[a][2] - SEDES[h][2]
                     for n, a, h in zip(partidos['neutral'], partidos['away_orig'], partidos['home_orig'])]
partidos['interior'] = partidos['roof'].isin(['dome', 'closed']).astype(int)
partidos['clima_conocido'] = (partidos['interior'] == 1) | partidos['wind'].notna()
partidos['viento'] = np.where(partidos['interior'] == 1, 0, partidos['wind'])
partidos['temp_ext'] = np.where(partidos['interior'] == 1, 70, partidos['temp'])   # 70 °F ≈ estadio cubierto
partidos['frio'] = (partidos['temp_ext'] < 35).astype(float).where(partidos['temp_ext'].notna())

VARS_GANADOR = ['elo_diff', 'campo_local', 'd_adj_off', 'd_adj_def', 'd_epa', 'd_def_epa', 'd_sr', 'd_def_sr',
                'd_pass_epa', 'd_rush_epa', 'd_margin', 'd_l3_margin', 'd_l3_epa', 'd_perdidas', 'd_def_perdidas',
                'd_tercera', 'd_def_tercera', 'd_zona_roja', 'd_def_zona_roja', 'd_yardas', 'd_def_yardas',
                'd_qb_val', 'h_qb_cambio', 'a_qb_cambio', 'd_rest', 'd_lesionados', 'd_margen_sede',
                'd_margen_tipo_rival', 'div_game', 'viaje_km', 'husos', 'partidos_min', 'playoff']
VARS_TOTAL = ['s_epa', 's_def_epa', 's_adj_off', 's_adj_def', 's_sr', 's_def_sr', 's_pass_epa', 's_pts_for',
              's_pts_against', 's_yardas', 's_def_yardas', 's_jugadas', 's_zona_roja', 's_def_zona_roja',
              's_perdidas', 's_qb_val', 'viento', 'temp_ext', 'interior', 'liga_pts_pre', 'playoff', 'partidos_min']

print(f'✅ Variables construidas: {len(VARS_GANADOR)} para ganador/margen y {len(VARS_TOTAL)} para total de puntos.')
print(f'   Partidos con variables: {len(partidos):,} (desde {partidos["season"].min()}).')
_muestra = partidos[(partidos['season'] == TEMPORADA)][['week', 'away_team', 'home_team', 'elo_diff', 'd_adj_off',
                                                         'd_adj_def', 'd_qb_val', 'd_rest']].tail(5)
display(_muestra.round(3))

# %% [markdown]
# ## 6️⃣ Entrenamiento de modelos
# **Qué hace:** define los modelos candidatos y los entrena con todos los partidos jugados para mostrar qué variables
# pesan más. La comparación justa (con partidos que el modelo **no** ha visto) se hace en la celda 7.
#
# | Modelo | Idea en una frase |
# |---|---|
# | Elo | Cada equipo tiene una puntuación que sube al ganar y baja al perder (referencia clásica). |
# | Regresión logística | Combina las variables de forma lineal; simple y difícil de sobreajustar. |
# | Random Forest | Muchos árboles de decisión que votan. |
# | HistGradientBoosting / XGBoost | Árboles que se corrigen unos a otros. |
# | Margen (Ridge) | Estima por cuántos puntos gana el local; sirve también para el **spread**. |
# | Total (Ridge) | Estima los puntos totales del partido; sirve para el **over/under**. |
# | Ensamble | Promedio de los anteriores. |

# %%
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

PRIMERA_TEMPORADA_ENTRENO = CONFIG['PRIMERA_TEMPORADA_PBP'] + 1   # la primera temporada de PBP solo "calienta" medias


def modelos_clasificacion():
    m = {
        'Regresión logística': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                             LogisticRegression(C=0.05, max_iter=3000)),
        'Random Forest': make_pipeline(SimpleImputer(strategy='median'),
                                       RandomForestClassifier(n_estimators=400, min_samples_leaf=25, max_features=0.33,
                                                              n_jobs=-1, random_state=42)),
        'HistGradientBoosting': HistGradientBoostingClassifier(learning_rate=0.03, max_iter=250, max_depth=3,
                                                               min_samples_leaf=40, l2_regularization=1.0,
                                                               random_state=42),
    }
    if XGBOOST_DISPONIBLE:
        from xgboost import XGBClassifier
        m['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0,
                                     eval_metric='logloss', random_state=42, n_jobs=-1)
    return m


def regresor():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=50.0))


def entrenar(train):
    """Entrena todos los modelos con los partidos de `train` (solo partidos ya jugados)."""
    sin_empate = train[train['result'] != 0]
    y = (sin_empate['result'] > 0).astype(int)
    clas = {n: m.fit(sin_empate[VARS_GANADOR], y) for n, m in modelos_clasificacion().items()}
    mg = regresor().fit(train[VARS_GANADOR], train['result'])
    tt = regresor().fit(train[VARS_TOTAL], train['total'])
    ajuste = lambda resid, p: float(np.sqrt(np.mean(resid ** 2) * len(resid) / (len(resid) - p - 1)))
    return {'clasificadores': clas, 'margen': mg, 'total': tt,
            'sigma_margen': ajuste(train['result'] - mg.predict(train[VARS_GANADOR]), len(VARS_GANADOR)),
            'sigma_total': ajuste(train['total'] - tt.predict(train[VARS_TOTAL]), len(VARS_TOTAL)),
            'n_train': len(train), 'hasta': str(train['gameday'].max().date())}


def predecir(mods, df):
    out = pd.DataFrame(index=df.index)
    for n, m in mods['clasificadores'].items():
        out['p_' + n] = m.predict_proba(df[VARS_GANADOR])[:, 1]
    out['margen_pred'] = mods['margen'].predict(df[VARS_GANADOR])
    out['p_Margen (Ridge)'] = norm.cdf(out['margen_pred'] / mods['sigma_margen'])
    out['p_Elo'] = df['p_elo'].values
    candidatos = [c for c in out.columns if c.startswith('p_')]
    out['p_Ensamble'] = out[candidatos].mean(axis=1)
    out['desacuerdo'] = out[candidatos].std(axis=1)          # dispersión entre modelos = incertidumbre
    out['total_pred'] = mods['total'].predict(df[VARS_TOTAL])
    out['sigma_margen'], out['sigma_total'] = mods['sigma_margen'], mods['sigma_total']
    return out


jugados = partidos[partidos['jugado'] & (partidos['season'] >= PRIMERA_TEMPORADA_ENTRENO)].copy()
_mods_todo = entrenar(jugados)
_coef = _mods_todo['clasificadores']['Regresión logística'][-1].coef_[0]
_imp = pd.DataFrame({'Variable': VARS_GANADOR, 'Peso (logística, estandarizado)': _coef}) \
    .assign(abs=lambda d: d.iloc[:, 1].abs()).sort_values('abs', ascending=False).drop(columns='abs').head(12)
print(f'Partidos de entrenamiento: {len(jugados):,} ({PRIMERA_TEMPORADA_ENTRENO}–{TEMPORADA}).')
print(f'Error típico del modelo de margen: ±{_mods_todo["sigma_margen"]:.1f} puntos; '
      f'del modelo de total: ±{_mods_todo["sigma_total"]:.1f} puntos.')
display(HTML('<b>Variables con más peso para elegir ganador (positivo = favorece al local):</b>'))
display(_imp.round(3))

# %% [markdown]
# ## 7️⃣ Validación temporal (walk-forward)
# **Qué hace:** simula cómo habría funcionado cada modelo en el pasado **sin trampas**: para predecir 2019 se entrena solo
# con 2017–2018; para 2020, con 2017–2019; y así sucesivamente. Se comparan con:
# * **Accuracy** (% de ganadores acertados), **Brier score** y **log loss** (miden si las probabilidades son realistas;
#   *más bajo es mejor*), y **calibración** (si dice 70 %, ¿gana ~70 % de las veces?).
# * La **probabilidad del mercado** (cuotas de cierre sin margen) como referencia: es muy difícil de superar.
#
# Se elige el modelo con **menor log loss** fuera de muestra (el más fiable en probabilidades), no el que más acierta.

# %%
import plotly.graph_objects as go
import plotly.express as px


def americano_a_decimal(x):
    x = pd.to_numeric(x, errors='coerce')
    return np.where(x > 0, 1 + x / 100, np.where(x < 0, 1 + 100 / np.abs(x), np.nan))


def prob_sin_margen(dec_a, dec_b):
    """Probabilidad implícita del lado A quitando el margen de la casa (normalización proporcional)."""
    pa, pb = 1 / np.asarray(dec_a, float), 1 / np.asarray(dec_b, float)
    return pa / (pa + pb)


partidos['ml_h_dec'] = americano_a_decimal(partidos['home_moneyline'])
partidos['ml_a_dec'] = americano_a_decimal(partidos['away_moneyline'])
partidos['sp_h_dec'] = americano_a_decimal(partidos['home_spread_odds'])
partidos['sp_a_dec'] = americano_a_decimal(partidos['away_spread_odds'])
partidos['ov_dec'] = americano_a_decimal(partidos['over_odds'])
partidos['un_dec'] = americano_a_decimal(partidos['under_odds'])
partidos['p_mercado'] = prob_sin_margen(partidos['ml_h_dec'], partidos['ml_a_dec'])

_oos = []
for S in range(CONFIG['PRIMERA_TEMPORADA_TEST'], TEMPORADA + 1):
    train = partidos[partidos['jugado'] & (partidos['season'] >= PRIMERA_TEMPORADA_ENTRENO) & (partidos['season'] < S)]
    test = partidos[partidos['jugado'] & (partidos['season'] == S)]
    if test.empty:
        continue
    _oos.append(pd.concat([test, predecir(entrenar(train), test)], axis=1))
    print(f'  {S}: entrenado con {len(train):,} partidos → evaluado en {len(test)} partidos')
oos = pd.concat(_oos)
CANDIDATOS = [c[2:] for c in oos.columns if c.startswith('p_') and c not in ('p_elo', 'p_mercado')]


def metricas(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {'Partidos': len(y), 'Accuracy': accuracy_score(y, p > 0.5), 'Brier': brier_score_loss(y, p),
            'Log loss': log_loss(y, p, labels=[0, 1])}


_ev = oos[oos['result'] != 0]
_y = (_ev['result'] > 0).astype(int)
tabla_modelos = pd.DataFrame({n: metricas(_y, _ev['p_' + n]) for n in CANDIDATOS}).T
_m = _ev['p_mercado'].notna()
tabla_modelos.loc['📈 Mercado (cuotas de cierre, referencia)'] = metricas(_y[_m], _ev.loc[_m, 'p_mercado'])
MODELO_ELEGIDO = tabla_modelos.loc[CANDIDATOS, 'Log loss'].idxmin()
tabla_modelos = tabla_modelos.sort_values('Log loss')
display(HTML(f'<h3>Comparación fuera de muestra ({oos["season"].min()}–{oos["season"].max()})</h3>'
             f'<p>✅ Modelo elegido: <b>{MODELO_ELEGIDO}</b> (menor log loss entre los candidatos).</p>'))
display(tabla_modelos.style.format({'Partidos': '{:.0f}', 'Accuracy': '{:.1%}', 'Brier': '{:.4f}', 'Log loss': '{:.4f}'})
        .highlight_min(subset=['Brier', 'Log loss'], color='#d8f0dd'))

_por_temp = pd.DataFrame({S: {**{f'{k} modelo': v for k, v in metricas((d['result'] > 0).astype(int), d['p_' + MODELO_ELEGIDO]).items()},
                              'Log loss mercado': metricas((d['result'] > 0).astype(int), d['p_mercado'])['Log loss']}
                          for S, d in _ev.groupby('season')}).T
display(HTML('<b>Rendimiento del modelo elegido por temporada (fuera de muestra):</b>'))
display(_por_temp.round(4))

# Calibración: ¿las probabilidades significan lo que dicen?
def curva_calibracion(p, y, bins=10):
    d = pd.DataFrame({'p': p, 'y': y}).dropna()
    d['bin'] = pd.cut(d['p'], np.linspace(0, 1, bins + 1))
    return d.groupby('bin', observed=True).agg(p_media=('p', 'mean'), frec_real=('y', 'mean'), n=('y', 'size')).reset_index()


fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Calibración perfecta', line=dict(dash='dot', color='gray')))
for nombre, col, color in [(MODELO_ELEGIDO, 'p_' + MODELO_ELEGIDO, '#2a78d6'), ('Mercado', 'p_mercado', '#e8890c')]:
    c = curva_calibracion(_ev[col], _y)
    fig.add_trace(go.Scatter(x=c['p_media'], y=c['frec_real'], mode='lines+markers', name=nombre,
                             marker=dict(size=np.clip(c['n'] / 12, 5, 18), color=color),
                             hovertemplate='Prob. estimada %{x:.0%}<br>Ganó de verdad %{y:.0%}<extra></extra>'))
fig.update_layout(title='Calibración: probabilidad estimada vs. frecuencia real de victoria local', height=420,
                  xaxis_title='Probabilidad estimada', yaxis_title='Frecuencia real', template='plotly_white',
                  xaxis_tickformat='.0%', yaxis_tickformat='.0%')
fig.show()

# %% [markdown]
# ## 8️⃣ Backtesting de la estrategia de apuestas
# **Qué hace:** con las predicciones fuera de muestra de la celda 7 y las **cuotas de cierre reales** publicadas por
# nflverse, simula apostar 1 unidad en cada oportunidad que pase los filtros, por mercado.
#
# * **Mezcla modelo/mercado:** la probabilidad final combina el modelo con la probabilidad sin margen del mercado.
#   El peso del modelo se elige en las temporadas de **calibración** (2019–21) minimizando el log loss.
# * **Umbral de valor esperado:** se elige por mercado en **calibración** y se comprueba en **prueba** (2022 en adelante).
#   Un mercado solo queda *habilitado* para recomendaciones si es rentable en ambos periodos.
# * Reglas de liquidación: empate en moneyline = devolución (push); spread o total exactamente en la línea = devolución.
#   El margen de la casa ya va incluido porque se usan las cuotas reales con su comisión.
#
# > 🟦 **Resultados HISTÓRICOS SIMULADOS — no son ganancias reales.**

# %%
def prob_linea(mu, sigma, linea):
    """P(resultado > línea) y P(resultado == línea) con aproximación normal (líneas enteras pueden dar push)."""
    mu, linea = np.asarray(mu, float), np.asarray(linea, float)
    entera = np.isclose(linea, np.round(linea))
    p_gana = np.where(entera, 1 - norm.cdf((linea + 0.5 - mu) / sigma), 1 - norm.cdf((linea - mu) / sigma))
    p_push = np.where(entera, norm.cdf((linea + 0.5 - mu) / sigma) - norm.cdf((linea - 0.5 - mu) / sigma), 0.0)
    return p_gana, p_push


def construir_opciones(df, col_p):
    """Una fila por partido, mercado y lado con la probabilidad del modelo, la del mercado y la cuota."""
    filas = []
    res = df['result'] if 'result' in df else pd.Series(np.nan, index=df.index)
    tot = df['total'] if 'total' in df else pd.Series(np.nan, index=df.index)
    # Moneyline
    def mercado(col, a, b):   # prob. sin margen ya calculada (consenso de casas) o a partir de las dos cuotas
        return df[col].values if col in df else prob_sin_margen(df[a], df[b])
    p = df[col_p].values
    pm = mercado('pm_ml', 'ml_h_dec', 'ml_a_dec')
    for lado, pp, pmk, dec, gana in [('local', p, pm, df['ml_h_dec'], res > 0), ('visitante', 1 - p, 1 - pm, df['ml_a_dec'], res < 0)]:
        filas.append(pd.DataFrame({'game_id': df['game_id'].values, 'mercado': 'Moneyline', 'lado': lado, 'linea': np.nan,
                                   'p_modelo': pp, 'p_push': 0.0, 'p_mercado': pmk, 'cuota': np.asarray(dec, float),
                                   'gana': np.where(res.isna(), np.nan, gana), 'push': np.where(res.isna(), np.nan, res == 0)}))
    # Spread (línea S = margen local que hay que superar: el local cubre si result > S)
    S = df['spread_line'].values
    pg, pu = prob_linea(df['margen_pred'], df['sigma_margen'], S)
    pv = 1 - pg - pu
    sp_h, sp_a = df['sp_h_dec'].values, df['sp_a_dec'].values
    pm = mercado('pm_sp', 'sp_h_dec', 'sp_a_dec')
    for lado, pw, pmk, dec, gana in [('local', pg, pm, sp_h, res > S), ('visitante', pv, 1 - pm, sp_a, res < S)]:
        filas.append(pd.DataFrame({'game_id': df['game_id'].values, 'mercado': 'Spread', 'lado': lado,
                                   'linea': np.where(lado == 'local', -S, S), 'p_modelo': pw / np.clip(1 - pu, 1e-9, None),
                                   'p_push': pu, 'p_mercado': pmk, 'cuota': dec,
                                   'gana': np.where(res.isna(), np.nan, gana), 'push': np.where(res.isna(), np.nan, res == S)}))
    # Total (over/under)
    L = df['total_line'].values
    po, pu = prob_linea(df['total_pred'], df['sigma_total'], L)
    pun = 1 - po - pu
    ov, un = df['ov_dec'].values, df['un_dec'].values
    pm = mercado('pm_tot', 'ov_dec', 'un_dec')
    for lado, pw, pmk, dec, gana in [('over', po, pm, ov, tot > L), ('under', pun, 1 - pm, un, tot < L)]:
        filas.append(pd.DataFrame({'game_id': df['game_id'].values, 'mercado': 'Total', 'lado': lado, 'linea': L,
                                   'p_modelo': pw / np.clip(1 - pu, 1e-9, None), 'p_push': pu, 'p_mercado': pmk,
                                   'cuota': dec, 'gana': np.where(tot.isna(), np.nan, gana),
                                   'push': np.where(tot.isna(), np.nan, tot == L)}))
    return pd.concat(filas, ignore_index=True)


def confianza_base(df):
    """Confianza 0-100 a partir de lo que también se conoce en el histórico (desacuerdo, inicio de temporada, datos)."""
    c = 100 - np.minimum(30, df['desacuerdo'].values * 300)
    c -= np.where(df['partidos_min'].values < 3, 15, 0)
    c -= np.where(df['h_pre_epa'].isna().values | df['a_pre_epa'].isna().values, 25, 0)
    return c


def evaluar(op, pesos):
    op = op.copy()
    w = op['mercado'].map(pesos)
    op['p_final'] = np.where(op['p_mercado'].notna(), w * op['p_modelo'] + (1 - w) * op['p_mercado'], op['p_modelo'])
    op['ev'] = (1 - op['p_push']) * (op['p_final'] * op['cuota'] - 1)
    op['ventaja'] = op['p_final'] - op['p_mercado']
    return op


def elegir_mejor_lado(op):
    """Como máximo una apuesta por partido y mercado: el lado con mayor valor esperado."""
    return op.sort_values('ev', ascending=False).drop_duplicates(['game_id', 'mercado'])


# Opciones fuera de muestra
op_hist = construir_opciones(oos, 'p_' + MODELO_ELEGIDO)
_info = oos[['game_id', 'season', 'week', 'gameday', 'desacuerdo', 'partidos_min', 'h_pre_epa', 'a_pre_epa', 'clima_conocido']]
op_hist = op_hist.merge(_info, on='game_id')
op_hist['confianza'] = confianza_base(op_hist) - np.where((op_hist['mercado'] == 'Total') & ~op_hist['clima_conocido'], 10, 0)
op_hist = op_hist[op_hist['cuota'].notna() & op_hist['p_mercado'].notna()]
CALIB = op_hist['season'].isin(CONFIG['TEMPORADAS_CALIBRACION'])

# 1) Peso del modelo en la mezcla, por mercado (solo temporadas de calibración)
PESOS_MODELO = {}
for mk in ['Moneyline', 'Spread', 'Total']:
    d = op_hist[CALIB & (op_hist['mercado'] == mk) & (op_hist['lado'].isin(['local', 'over'])) & (op_hist['push'] == 0)]
    perdidas = {w: log_loss(d['gana'].astype(int), np.clip(w * d['p_modelo'] + (1 - w) * d['p_mercado'], 1e-4, 1 - 1e-4), labels=[0, 1])
                for w in np.round(np.arange(0, 1.01, 0.05), 2)}
    PESOS_MODELO[mk] = min(perdidas, key=perdidas.get)
op_hist = evaluar(op_hist, PESOS_MODELO)


def liquidar(b):
    return np.where(b['push'] == 1, 0.0, np.where(b['gana'] == 1, b['cuota'] - 1, -1.0))


def filtrar(op, umbral_ev):
    return op[(op['ev'] >= umbral_ev) & (op['ventaja'] >= CONFIG['VENTAJA_MINIMA']) & (op['cuota'] <= CONFIG['CUOTA_MAXIMA'])
              & (op['confianza'] >= CONFIG['CONFIANZA_MINIMA'])]


def resumen_apuestas(b, semilla=7):
    if b.empty:
        return {'Apuestas': 0}
    b = b.sort_values(['gameday', 'game_id'])
    u = liquidar(b)
    acum = np.cumsum(u)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], acum]))[1:] - acum)) if len(u) else 0.0
    rng = np.random.default_rng(semilla)
    boot = [rng.choice(u, len(u)).mean() for _ in range(2000)]
    resueltas = b['push'] == 0
    p = np.clip(b.loc[resueltas, 'p_final'], 1e-4, 1 - 1e-4)
    y = b.loc[resueltas, 'gana'].astype(int)
    return {'Apuestas': len(b), 'Aciertos': (b['gana'] == 1).sum() / max(resueltas.sum(), 1), 'Beneficio (u)': u.sum(),
            'ROI': u.mean(), 'ROI IC95% inf': np.percentile(boot, 2.5), 'ROI IC95% sup': np.percentile(boot, 97.5),
            'EV medio estimado': b['ev'].mean(), 'Drawdown máx (u)': dd,
            'Brier': brier_score_loss(y, p) if len(y) else np.nan,
            'Log loss': log_loss(y, p, labels=[0, 1]) if len(y) else np.nan}


# 2) Umbral de EV por mercado (elegido en calibración)
REJILLA_EV = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
UMBRAL_EV, HABILITADO, _filas = {}, {}, []
mejores = elegir_mejor_lado(op_hist)
for mk in ['Moneyline', 'Spread', 'Total']:
    d = mejores[mejores['mercado'] == mk]
    candidatos = []
    for u_ev in REJILLA_EV:
        r = resumen_apuestas(filtrar(d[d['season'].isin(CONFIG['TEMPORADAS_CALIBRACION'])], u_ev))
        if r['Apuestas'] >= CONFIG['MIN_APUESTAS_BACKTEST']:
            candidatos.append((r['ROI'], u_ev))
    UMBRAL_EV[mk] = max(candidatos)[1] if candidatos else 0.05
    for periodo, mascara in [('Calibración', d['season'].isin(CONFIG['TEMPORADAS_CALIBRACION'])),
                             ('Prueba', ~d['season'].isin(CONFIG['TEMPORADAS_CALIBRACION']))]:
        _filas.append({'Mercado': mk, 'Periodo': periodo, 'Umbral EV': UMBRAL_EV[mk],
                       **resumen_apuestas(filtrar(d[mascara], UMBRAL_EV[mk]))})
    _filas.append({'Mercado': mk, 'Periodo': 'Prueba · sin filtros (todo EV>0)', 'Umbral EV': 0.0,
                   **resumen_apuestas(d[~d['season'].isin(CONFIG['TEMPORADAS_CALIBRACION']) & (d['ev'] > 0)])})
    cal_r = [f for f in _filas if f['Mercado'] == mk and f['Periodo'] == 'Calibración'][0]
    pru_r = [f for f in _filas if f['Mercado'] == mk and f['Periodo'] == 'Prueba'][0]
    HABILITADO[mk] = all(r.get('Apuestas', 0) >= CONFIG['MIN_APUESTAS_BACKTEST'] and r['ROI'] > 0 for r in (cal_r, pru_r))
    if not CONFIG['EXIGIR_RESPALDO_BACKTEST']:
        HABILITADO[mk] = True

tabla_backtest = pd.DataFrame(_filas)
display(HTML('<h3>🟦 Backtesting histórico simulado (1 unidad por apuesta, cuotas de cierre reales)</h3>'
             f'<p>Peso del modelo en la mezcla con el mercado: {", ".join(f"{k} {v:.0%}" for k, v in PESOS_MODELO.items())}. '
             'Un peso bajo significa que el mercado ya contenía casi toda la información del modelo.</p>'))
display(tabla_backtest.style.format({'Umbral EV': '{:.0%}', 'Aciertos': '{:.1%}', 'Beneficio (u)': '{:+.1f}', 'ROI': '{:+.1%}',
                                     'ROI IC95% inf': '{:+.1%}', 'ROI IC95% sup': '{:+.1%}', 'EV medio estimado': '{:+.1%}',
                                     'Drawdown máx (u)': '{:.1f}', 'Brier': '{:.4f}', 'Log loss': '{:.4f}', 'Apuestas': '{:.0f}'},
                                    na_rep='—'))
_hab = ', '.join(f'{k}: {"✅ habilitado" if v else "⛔ sin respaldo suficiente"}' for k, v in HABILITADO.items())
display(HTML(f'<p><b>Mercados para recomendaciones:</b> {_hab}</p>'
             '<p style="color:#555">Cómo leerlo: si el intervalo de confianza del ROI incluye el 0 %, el resultado podría deberse '
             'al azar. Las cuotas usadas son de <b>cierre</b> (las más eficientes); no hay cuotas históricas de apertura '
             'gratuitas, así que no se puede medir el CLV histórico. El registro (celda 14) sí lo mide en vivo.</p>'))

# Resultados por temporada y curva de beneficio acumulado (con los umbrales elegidos)
_apuestas_bt = pd.concat([filtrar(mejores[mejores['mercado'] == mk], UMBRAL_EV[mk]) for mk in UMBRAL_EV])
_apuestas_bt = _apuestas_bt.assign(unidades=liquidar(_apuestas_bt)).sort_values(['gameday', 'game_id'])
if not _apuestas_bt.empty:
    display(HTML('<b>Beneficio simulado por temporada y mercado (unidades):</b>'))
    display(_apuestas_bt.pivot_table(index='season', columns='mercado', values='unidades', aggfunc=['sum', 'count'])
            .round(1).fillna(0))
    fig = go.Figure()
    for mk, d in _apuestas_bt.groupby('mercado'):
        fig.add_trace(go.Scatter(x=d['gameday'], y=d['unidades'].cumsum(), mode='lines', name=mk))
    _corte = pd.Timestamp(f'{max(CONFIG["TEMPORADAS_CALIBRACION"]) + 1}-08-01')
    fig.add_vline(x=_corte.timestamp() * 1000, line_dash='dot', line_color='gray')
    fig.add_annotation(x=_corte, y=1, yref='paper', text='← calibración | prueba →', showarrow=False)
    fig.update_layout(title='Beneficio acumulado SIMULADO (unidades) — histórico, no real', template='plotly_white',
                      height=400, yaxis_title='Unidades')
    fig.show()

# %% [markdown]
# ## 9️⃣ Predicción de los partidos actuales y futuros
# **Qué hace:** entrena la versión final del modelo elegido con **todos** los partidos ya jugados y predice los de la
# temporada en curso.
#
# **Control de versiones (sin reentrenar por impulsos):** el modelo guardado solo se sustituye cuando hay al menos
# 16 partidos nuevos **y** la validación temporal de la celda 7 no empeora respecto a la versión anterior. Nunca se
# ajusta el modelo "para corregir" las últimas derrotas.

# %%
import joblib

_firma = hashlib.sha1(json.dumps([VARS_GANADOR, VARS_TOTAL, MODELO_ELEGIDO, VIDA_MEDIA, QB_PRIOR, QB_K,
                                  PRIMERA_TEMPORADA_ENTRENO]).encode()).hexdigest()[:8]
_logloss_wf = float(tabla_modelos.loc[MODELO_ELEGIDO, 'Log loss'])
_meta_ruta, _mod_ruta = CARPETA / 'modelos' / 'actual.json', CARPETA / 'modelos' / 'actual.joblib'
_meta = json.loads(_meta_ruta.read_text()) if _meta_ruta.exists() else None
_motivo = None
if _meta and _meta['firma'] == _firma and _mod_ruta.exists():
    _nuevos = len(jugados) - _meta['n_train']
    if _nuevos < CONFIG['MIN_PARTIDOS_NUEVOS_PARA_REENTRENAR']:
        _motivo = f'se mantiene la versión {_meta["version"]}: solo hay {_nuevos} partidos nuevos (mínimo {CONFIG["MIN_PARTIDOS_NUEVOS_PARA_REENTRENAR"]}).'
    elif _logloss_wf > _meta['logloss_wf'] + 0.005:
        _motivo = (f'se mantiene la versión {_meta["version"]}: la nueva validación temporal empeora '
                   f'({_logloss_wf:.4f} vs {_meta["logloss_wf"]:.4f}). Revisa los datos antes de sustituirla.')
if _motivo:
    MODELOS = joblib.load(_mod_ruta)
    VERSION_MODELO = _meta['version']
    print('ℹ️ Modelo:', _motivo)
else:
    MODELOS = entrenar(jugados)
    VERSION_MODELO = f'{_firma}-{MODELOS["hasta"]}'
    joblib.dump(MODELOS, _mod_ruta)
    _meta_ruta.write_text(json.dumps({'firma': _firma, 'version': VERSION_MODELO, 'n_train': len(jugados),
                                      'logloss_wf': _logloss_wf, 'modelo': MODELO_ELEGIDO,
                                      'entrenado': HOY.strftime('%Y-%m-%d %H:%M UTC')}))
    print(f'✅ Nueva versión del modelo: {VERSION_MODELO} ({MODELO_ELEGIDO}, {len(jugados):,} partidos).')

temp = partidos[partidos['season'] == TEMPORADA].copy()
temp = pd.concat([temp, predecir(MODELOS, temp)], axis=1)
temp['p_local'] = temp['p_' + MODELO_ELEGIDO]
SEMANAS = sorted(temp['week'].unique())
_pend = temp.loc[~temp['jugado'], 'week']
SEMANA_OBJETIVO = int(_pend.min()) if len(_pend) else int(max(SEMANAS))
print(f'📅 Próxima semana con partidos pendientes: {SEMANA_OBJETIVO}')
display(temp[temp['week'] == SEMANA_OBJETIVO][['gameday', 'away_team', 'home_team', 'h_qb_name', 'a_qb_name', 'p_local',
                                                'margen_pred', 'total_pred']]
        .rename(columns={'gameday': 'Fecha', 'away_team': 'Visitante', 'home_team': 'Local', 'h_qb_name': 'QB local',
                         'a_qb_name': 'QB visitante', 'p_local': 'Prob. local', 'margen_pred': 'Margen local estimado',
                         'total_pred': 'Puntos totales estimados'})
        .style.format({'Prob. local': '{:.0%}', 'Margen local estimado': '{:+.1f}', 'Puntos totales estimados': '{:.1f}',
                       'Fecha': lambda d: d.strftime('%d/%m')}))

# %% [markdown]
# ## 🔟 Descarga y análisis de cuotas
# **Qué hace:** reúne las cuotas en este orden de prioridad:
# 1. **CSV manual** (si activaste `USAR_CSV_MANUAL`).
# 2. **The Odds API** (si configuraste la clave): varias casas; se usa la **cuota mediana** del mercado para calcular el
#    valor (prudente) y se muestra aparte la mejor cuota encontrada.
# 3. **nflverse**: líneas de referencia publicadas con el calendario. Pueden llevar horas de retraso → se marca y se
#    reduce la confianza. **Comprueba siempre la cuota real en tu casa de apuestas antes de decidir.**
#
# **Errores comunes:** `401` = clave incorrecta; `429` = agotaste las consultas gratuitas del mes.

# %%
EQUIPOS_ODDS_API = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Los Angeles Rams': 'LA', 'Los Angeles Chargers': 'LAC', 'Las Vegas Raiders': 'LV', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'Seattle Seahawks': 'SEA',
    'San Francisco 49ers': 'SF', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
}
COLS_CUOTA = ['ml_h_dec', 'ml_a_dec', 'spread_line', 'sp_h_dec', 'sp_a_dec', 'total_line', 'ov_dec', 'un_dec']
_pub_nflverse = next((f['Publicado por la fuente'] for f in FUENTES if 'calendario' in f['Fuente']), '')

# --- Referencia nflverse (siempre disponible) ---
cuotas = temp[['game_id'] + COLS_CUOTA].copy()
for mk in ('ml', 'sp', 'tot'):
    cuotas[f'fuente_{mk}'] = np.where(cuotas['ml_h_dec' if mk == 'ml' else 'sp_h_dec' if mk == 'sp' else 'ov_dec'].notna(),
                                      'nflverse (referencia)', '')
    cuotas[f'casa_{mk}'] = ''
cuotas['actualizado'] = _pub_nflverse
cuotas['n_casas'] = 1


def buscar_partido(local, visitante, fecha=None):
    """Devuelve (game_id, invertido) del partido pendiente de la temporada con esos equipos."""
    c = temp[~temp['jugado'] & (((temp['home_team'] == local) & (temp['away_team'] == visitante)) |
                                ((temp['home_team'] == visitante) & (temp['away_team'] == local)))]
    if fecha is not None and len(c) > 1:
        c = c.iloc[(c['gameday'] - pd.Timestamp(fecha).tz_localize(None).normalize()).abs().argsort()]
    if c.empty:
        return None, False
    return c.iloc[0]['game_id'], c.iloc[0]['home_team'] != local


def consenso(precios_a, precios_b):
    """Prob. sin margen de la mediana de cuotas y mejor cuota de cada lado."""
    a, b = np.median(precios_a), np.median(precios_b)
    return a, b, float(prob_sin_margen(a, b))


ESTADO_CUOTAS = 'Líneas de referencia de nflverse (sin clave de API).'
if ODDS_API_KEY:
    try:
        r = requests.get('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds',
                         params={'apiKey': ODDS_API_KEY, 'regions': CONFIG['REGION_ODDS_API'],
                                 'markets': 'h2h,spreads,totals', 'oddsFormat': 'decimal', 'dateFormat': 'iso'},
                         timeout=30)
        r.raise_for_status()
        eventos = r.json()
        _n = 0
        for ev in eventos:
            loc, vis = EQUIPOS_ODDS_API.get(ev['home_team']), EQUIPOS_ODDS_API.get(ev['away_team'])
            gid, inv = buscar_partido(loc, vis, ev['commence_time'])
            if gid is None:
                continue
            if inv:   # la API puede listar al revés los partidos en campo neutral
                loc, vis = vis, loc
            casas = {'ml': {}, 'sp': {}, 'tot': {}}
            ultima = []
            for bk in ev.get('bookmakers', []):
                ultima.append(bk.get('last_update', ''))
                for m in bk.get('markets', []):
                    o = {EQUIPOS_ODDS_API.get(x['name'], x['name']): x for x in m['outcomes']}
                    if m['key'] == 'h2h' and loc in o and vis in o:
                        casas['ml'][bk['title']] = (None, o[loc]['price'], o[vis]['price'])
                    elif m['key'] == 'spreads' and loc in o and vis in o:
                        casas['sp'][bk['title']] = (-o[loc]['point'], o[loc]['price'], o[vis]['price'])
                    elif m['key'] == 'totals' and 'Over' in o and 'Under' in o:
                        casas['tot'][bk['title']] = (o['Over']['point'], o['Over']['price'], o['Under']['price'])
            i = cuotas.index[cuotas['game_id'] == gid][0]
            for mk, (ca, cb, cl) in {'ml': ('ml_h_dec', 'ml_a_dec', None), 'sp': ('sp_h_dec', 'sp_a_dec', 'spread_line'),
                                     'tot': ('ov_dec', 'un_dec', 'total_line')}.items():
                if not casas[mk]:
                    continue
                lineas = pd.Series([v[0] for v in casas[mk].values()])
                linea = None if cl is None else lineas.mode().iloc[0]          # línea de consenso
                sel = {k: v for k, v in casas[mk].items() if cl is None or v[0] == linea}
                a, b, pm = consenso([v[1] for v in sel.values()], [v[2] for v in sel.values()])
                cuotas.loc[i, [ca, cb]] = a, b
                if cl:
                    cuotas.loc[i, cl] = linea
                cuotas.loc[i, f'pm_{mk}'] = pm
                mejor_a = max(sel.items(), key=lambda kv: kv[1][1])
                mejor_b = max(sel.items(), key=lambda kv: kv[1][2])
                cuotas.loc[i, f'mejor_{mk}'] = f'{mejor_a[1][1]:.2f} ({mejor_a[0]}) / {mejor_b[1][2]:.2f} ({mejor_b[0]})'
                cuotas.loc[i, f'fuente_{mk}'] = 'The Odds API (mediana)'
                cuotas.loc[i, f'casa_{mk}'] = f'{len(sel)} casas'
            cuotas.loc[i, 'actualizado'] = max(ultima) if ultima else ''
            cuotas.loc[i, 'n_casas'] = len(ev.get('bookmakers', []))
            _n += 1
        ESTADO_CUOTAS = (f'The Odds API: {_n} partidos con cuotas en vivo. Consultas restantes este mes: '
                         f'{r.headers.get("x-requests-remaining", "?")}.')
        FUENTES.append({'Fuente': 'The Odds API · cuotas en vivo', 'Estado': 'descargado ahora', 'Publicado por la fuente': '',
                        'Descargado': HOY.strftime('%Y-%m-%d %H:%M UTC'), 'Filas': _n, 'Dato más reciente': '',
                        'URL': 'https://the-odds-api.com'})
    except Exception as e:
        codigo = getattr(getattr(e, 'response', None), 'status_code', '')
        ayuda = {401: 'clave incorrecta', 429: 'cuota mensual agotada'}.get(codigo, str(e)[:120])
        ESTADO_CUOTAS = f'⚠️ The Odds API falló ({ayuda}); se usan líneas de referencia de nflverse.'
        AVISOS_DATOS.append(ESTADO_CUOTAS)

# --- CSV manual (máxima prioridad) ---
PLANTILLA = CARPETA / 'plantilla_cuotas.csv'
_prox = temp[~temp['jugado'] & (temp['week'] == SEMANA_OBJETIVO)]
pd.DataFrame([{'local': g.home_team, 'visitante': g.away_team, 'mercado': m, 'lado': l, 'linea': '', 'cuota': '',
               'casa': '', 'fecha_hora': ''}
              for g in _prox.itertuples() for m, lados in [('moneyline', ['local', 'visitante']),
                                                          ('spread', ['local', 'visitante']), ('total', ['over', 'under'])]
              for l in lados]).to_csv(PLANTILLA, index=False)


def cargar_csv_cuotas(contenido):
    global ESTADO_CUOTAS
    m = pd.read_csv(io.BytesIO(contenido) if isinstance(contenido, bytes) else contenido)
    m.columns = [c.strip().lower() for c in m.columns]
    m['cuota_dec'] = [float(c) if abs(float(c)) < 50 and float(c) > 1 else float(americano_a_decimal(c))
                      for c in m['cuota']]
    n = 0
    for (loc, vis), d in m.groupby(['local', 'visitante']):
        gid, inv = buscar_partido(str(loc).upper(), str(vis).upper())
        if gid is None:
            print(f'⚠️ No se encontró el partido {vis} @ {loc} entre los pendientes.')
            continue
        i = cuotas.index[cuotas['game_id'] == gid][0]
        lado = lambda x: {'local': 'visitante', 'visitante': 'local'}.get(x, x) if inv else x
        f = {(r.mercado.lower(), lado(r.lado.lower())): r for r in d.itertuples()}
        if ('moneyline', 'local') in f and ('moneyline', 'visitante') in f:
            cuotas.loc[i, ['ml_h_dec', 'ml_a_dec', 'fuente_ml']] = f['moneyline', 'local'].cuota_dec, f['moneyline', 'visitante'].cuota_dec, 'CSV manual'
        if ('spread', 'local') in f and ('spread', 'visitante') in f:
            cuotas.loc[i, ['spread_line', 'sp_h_dec', 'sp_a_dec', 'fuente_sp']] = (
                -float(f['spread', 'local'].linea), f['spread', 'local'].cuota_dec, f['spread', 'visitante'].cuota_dec, 'CSV manual')
        if ('total', 'over') in f and ('total', 'under') in f:
            cuotas.loc[i, ['total_line', 'ov_dec', 'un_dec', 'fuente_tot']] = (
                float(f['total', 'over'].linea), f['total', 'over'].cuota_dec, f['total', 'under'].cuota_dec, 'CSV manual')
        for mk in ('ml', 'sp', 'tot'):
            if f'pm_{mk}' in cuotas:
                cuotas.loc[i, f'pm_{mk}'] = np.nan
        cuotas.loc[i, 'actualizado'] = str(d['fecha_hora'].iloc[0])
        n += 1
    ESTADO_CUOTAS = f'CSV manual: {n} partidos cargados (resto: {ESTADO_CUOTAS})'
    print('✅', ESTADO_CUOTAS)


if CONFIG['USAR_CSV_MANUAL']:
    if EN_COLAB:
        from google.colab import files
        print(f'Plantilla creada en {PLANTILLA}. Descárgala, rellénala y súbela:')
        files.download(str(PLANTILLA))
        _sub = files.upload()
        for _nombre, _contenido in _sub.items():
            cargar_csv_cuotas(_contenido)
    elif Path('mis_cuotas.csv').exists():
        cargar_csv_cuotas('mis_cuotas.csv')

print('💱 Cuotas:', ESTADO_CUOTAS)
_vista = cuotas.merge(temp[['game_id', 'week', 'home_team', 'away_team']], on='game_id')
_vista['handicap_local'] = -_vista['spread_line']
display(_vista[_vista['week'] == SEMANA_OBJETIVO][['away_team', 'home_team', 'ml_a_dec', 'ml_h_dec', 'handicap_local', 'sp_h_dec',
                                                   'total_line', 'ov_dec', 'un_dec', 'fuente_ml', 'actualizado']]
        .rename(columns={'away_team': 'Visitante', 'home_team': 'Local', 'ml_a_dec': 'Cuota visitante',
                         'ml_h_dec': 'Cuota local', 'handicap_local': 'Hándicap local', 'sp_h_dec': 'Cuota spread local',
                         'total_line': 'Línea total', 'ov_dec': 'Cuota over', 'un_dec': 'Cuota under',
                         'fuente_ml': 'Fuente', 'actualizado': 'Actualizado'}).round(2))

# %% [markdown]
# ## 1️⃣1️⃣ Cálculo del valor esperado y decisión
# Para cada partido y mercado se calcula: probabilidad del modelo, probabilidad final (mezcla con el mercado),
# probabilidad implícita de la cuota (con y sin margen de la casa), cuota justa, ventaja, valor esperado e incertidumbre.
#
# **Reglas de decisión** (en este orden):
# * 🔴 **Problema de datos**: falta la cuota o el QB previsto figura como baja (*Out*).
# * ⚪ **No apostar**: la cuota no paga lo suficiente (valor esperado o ventaja bajo el umbral), la cuota es demasiado alta,
#   o el mercado no demostró ventaja en el backtesting.
# * 🟡 **Esperar**: hay valor, pero falta información (QB sin confirmar o en duda, reporte de lesiones pendiente, cuota de
#   referencia sin verificar, clima desconocido) o la confianza es baja.
# * 🟢 **Apostar (cumple criterios)**: supera todos los filtros. *No significa que vaya a ganar.*

# %%
op = construir_opciones(temp.drop(columns=COLS_CUOTA).merge(cuotas, on='game_id'), 'p_local')
_ctx = temp[['game_id', 'season', 'week', 'gameday', 'jugado', 'home_team', 'away_team', 'desacuerdo', 'partidos_min',
             'h_pre_epa', 'a_pre_epa', 'clima_conocido', 'h_qb_confirmado', 'a_qb_confirmado', 'h_qb_estado', 'a_qb_estado',
             'h_qb_name', 'a_qb_name', 'h_reporte_lesiones', 'a_reporte_lesiones', 'result', 'total']]
op = op.merge(_ctx, on='game_id').merge(cuotas[['game_id', 'fuente_ml', 'fuente_sp', 'fuente_tot', 'actualizado']], on='game_id')
op['fuente_cuota'] = np.select([op['mercado'] == 'Moneyline', op['mercado'] == 'Spread'],
                               [op['fuente_ml'], op['fuente_sp']], op['fuente_tot'])
for _mk, _nombre in [('ml', 'Moneyline'), ('sp', 'Spread'), ('tot', 'Total')]:   # mejor cuota encontrada (The Odds API)
    if f'mejor_{_mk}' in cuotas:
        _mejor = op['game_id'].map(cuotas.set_index('game_id')[f'mejor_{_mk}'])
        _lado0 = op['lado'].isin(['local', 'over'])
        _txt = _mejor.str.split(' / ').str[0].where(_lado0, _mejor.str.split(' / ').str[1])
        _sel = (op['mercado'] == _nombre) & _txt.notna()
        op.loc[_sel, 'fuente_cuota'] = op.loc[_sel, 'fuente_cuota'] + ' · mejor: ' + _txt[_sel]
op['confianza'] = confianza_base(op)
op = evaluar(op, PESOS_MODELO)
op['p_implicita'] = 1 / op['cuota']                       # con margen de la casa
op['cuota_justa'] = 1 / op['p_final']
op['margen_casa'] = np.nan
for mk, (a, b) in {'Moneyline': ('ml_h_dec', 'ml_a_dec'), 'Spread': ('sp_h_dec', 'sp_a_dec'),
                   'Total': ('ov_dec', 'un_dec')}.items():
    _m = cuotas.set_index('game_id')
    op.loc[op['mercado'] == mk, 'margen_casa'] = (1 / _m[a] + 1 / _m[b] - 1).reindex(op.loc[op['mercado'] == mk, 'game_id']).values
# Incertidumbre: desviación de la probabilidad final según el desacuerdo entre modelos
op['incertidumbre'] = op['desacuerdo'] * op['mercado'].map(PESOS_MODELO).clip(lower=0.25)

try:
    HORAS_CUOTA_NFLVERSE = (pd.Timestamp(HOY) - pd.to_datetime(_pub_nflverse, utc=True)).total_seconds() / 3600
except Exception:
    HORAS_CUOTA_NFLVERSE = None


def señales(r):
    """Ajustes de confianza y avisos que solo existen en vivo (lesiones, QB, cuotas)."""
    pen, amarillas, rojas = 0, [], []
    for lado, conf, est, nom, rep in [('local', r.h_qb_confirmado, r.h_qb_estado, r.h_qb_name, r.h_reporte_lesiones),
                                      ('visitante', r.a_qb_confirmado, r.a_qb_estado, r.a_qb_name, r.a_reporte_lesiones)]:
        if est == 'Out':
            rojas.append(f'el QB previsto del {lado} ({nom}) figura como baja')
        elif est in ('Doubtful', 'Questionable'):
            pen += 25; amarillas.append(f'QB {lado} ({nom}) en duda ({est})')
        elif not conf:
            pen += 15; amarillas.append(f'QB titular {lado} sin confirmar (se asume {nom})')
        if not rep:
            pen += 5; amarillas.append(f'reporte de lesiones del {lado} aún no publicado')
    if 'nflverse' in str(r.fuente_cuota):
        pen += 10                                    # línea de referencia, no de una casa concreta
        if HORAS_CUOTA_NFLVERSE is None or HORAS_CUOTA_NFLVERSE > 12:
            amarillas.append('cuota de referencia con más de 12 h: verifícala en tu casa de apuestas')
    if r.mercado == 'Total' and not r.clima_conocido:
        pen += 10; amarillas.append('pronóstico del clima aún no disponible')
    return pen, amarillas, rojas


def decidir(r):
    if r.jugado:
        return 'Jugado', 'Partido ya disputado (modo revisión).'
    if pd.isna(r.cuota) or pd.isna(r.p_final):
        return 'Problema de datos', 'No hay cuota disponible para este mercado.'
    if r.rojas:
        return 'Problema de datos', 'Atención: ' + '; '.join(r.rojas) + '. Las cifras pueden no ser válidas.'
    umbral = UMBRAL_EV[r.mercado]
    if r.cuota > CONFIG['CUOTA_MAXIMA']:
        return 'No apostar', f'Cuota muy alta ({r.cuota:.2f}): demasiado riesgo y error de estimación.'
    if r.ev < umbral or r.ventaja < CONFIG['VENTAJA_MINIMA']:
        if r.ev <= 0:
            return 'No apostar', 'La cuota no paga lo suficiente para la probabilidad estimada (valor esperado negativo).'
        return 'No apostar', f'Valor esperado {r.ev:+.1%} y ventaja {r.ventaja:+.1%}: por debajo de los umbrales ({umbral:.0%} y {CONFIG["VENTAJA_MINIMA"]:.0%}).'
    if not HABILITADO[r.mercado]:
        return 'No apostar', f'Hay valor teórico ({r.ev:+.1%}), pero en el backtesting este mercado no demostró ventaja real.'
    if r.amarillas or r.confianza < CONFIG['CONFIANZA_MINIMA']:
        detalle = '; '.join(r.amarillas) if r.amarillas else f'confianza baja ({r.confianza:.0f}/100)'
        return 'Esperar', f'Valor esperado {r.ev:+.1%}, pero falta información: {detalle}.'
    nota = ' Comprueba que tu casa ofrece una cuota igual o mayor.' if 'nflverse' in str(r.fuente_cuota) else ''
    return 'Apostar', f'Supera los filtros: ventaja {r.ventaja:+.1%} sobre el mercado y valor esperado {r.ev:+.1%}.{nota}'


_s = op.apply(señales, axis=1, result_type='expand')
op['confianza'] = (op['confianza'] - _s[0]).clip(0, 100)
op['amarillas'], op['rojas'] = _s[1], _s[2]
mejores_op = elegir_mejor_lado(op)
mejores_op[['decision', 'motivo']] = mejores_op.apply(decidir, axis=1, result_type='expand')
mejores_op['nivel_confianza'] = pd.cut(mejores_op['confianza'], [-1, 49.9, 69.9, 101], labels=['Baja', 'Media', 'Alta']).astype(str)

# Importe hipotético: Kelly fraccional con topes por apuesta y por semana (solo para "Apostar")
mejores_op['kelly'] = (mejores_op['p_final'] * mejores_op['cuota'] - 1) / (mejores_op['cuota'] - 1)
mejores_op['fraccion'] = np.where(mejores_op['decision'] == 'Apostar',
                                  np.clip(mejores_op['kelly'] * CONFIG['FRACCION_KELLY'], 0, CONFIG['TOPE_POR_APUESTA']), 0.0)
for _sem, _d in mejores_op.groupby('week'):
    _tot = _d['fraccion'].sum()
    if _tot > CONFIG['TOPE_SEMANAL']:
        mejores_op.loc[_d.index, 'fraccion'] *= CONFIG['TOPE_SEMANAL'] / _tot
mejores_op['monto'] = (mejores_op['fraccion'] * CONFIG['CAPITAL_HIPOTETICO']).round(2)

ORDEN = {'Apostar': 0, 'Esperar': 1, 'Problema de datos': 2, 'No apostar': 3, 'Jugado': 4}
mejores_op['orden'] = mejores_op['decision'].map(ORDEN)
mejores_op['calidad'] = mejores_op['ev'].fillna(-1) * mejores_op['confianza'] / 100
mejores_op = mejores_op.sort_values(['week', 'orden', 'calidad'], ascending=[True, True, False]).reset_index(drop=True)
_sem = mejores_op[mejores_op['week'] == SEMANA_OBJETIVO]
print(f'Semana {SEMANA_OBJETIVO}: ' + ', '.join(f'{k}: {v}' for k, v in _sem['decision'].value_counts().items()))

# %% [markdown]
# ## 1️⃣2️⃣ 📊 PANEL — tu resumen semanal
# **Esta es la celda importante.** Elige la semana en la lista desplegable. Debajo tienes:
# **A.** resumen · **B.** tabla de recomendaciones · **C.** explicación de cada oportunidad · **D.** glosario ·
# **E.** análisis individual de cualquier partido.
#
# Colores: 🟢 cumple los criterios · 🟡 falta información · ⚪ sin ventaja suficiente · 🔴 problema de datos.
# **Ningún color significa que una apuesta sea segura.**

# %%
import ipywidgets as widgets
from IPython.display import clear_output

ESTILO = """<style>
.nfl{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#1f2328;background:#ffffff;padding:6px 4px;line-height:1.45}
.nfl h2{margin:6px 0 4px;font-size:22px}.nfl h3{margin:14px 0 6px;font-size:17px}
.nfl .tarjetas{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}
.nfl .tarjeta{flex:1 1 140px;border-radius:10px;padding:10px 14px;background:#f6f8fa;border:1px solid #d0d7de}
.nfl .tarjeta .num{font-size:30px;font-weight:700}.nfl .tarjeta .txt{font-size:13px;color:#57606a}
.nfl .t-verde{border-left:6px solid #2e9e5b}.nfl .t-amarillo{border-left:6px solid #d9a400}
.nfl .t-gris{border-left:6px solid #8c959f}.nfl .t-azul{border-left:6px solid #2a78d6}
.nfl table.rec{border-collapse:collapse;width:100%;font-size:13px}
.nfl table.rec th{background:#24292f;color:#fff;padding:7px 6px;text-align:left;font-weight:600}
.nfl table.rec td{padding:7px 6px;border-bottom:1px solid #d0d7de;vertical-align:top}
.nfl tr.verde td{background:#e9f7ee}.nfl tr.amarillo td{background:#fff7dc}
.nfl tr.gris td{background:#f6f8fa;color:#57606a}.nfl tr.rojo td{background:#fdecec}.nfl tr.jugado td{background:#eef3fb}
.nfl .chip{display:inline-block;padding:2px 9px;border-radius:12px;font-weight:600;font-size:12px;white-space:nowrap}
.nfl .c-verde{background:#2e9e5b;color:#fff}.nfl .c-amarillo{background:#d9a400;color:#1f2328}
.nfl .c-gris{background:#8c959f;color:#fff}.nfl .c-rojo{background:#d64545;color:#fff}.nfl .c-azul{background:#2a78d6;color:#fff}
.nfl .aviso{background:#fff8e1;border-left:4px solid #d9a400;padding:8px 12px;margin:8px 0;font-size:13px}
.nfl .nota{background:#f6f8fa;border-left:4px solid #8c959f;padding:8px 12px;margin:8px 0;font-size:13px;color:#424a53}
.nfl .noapostar{background:#f6f8fa;border:2px solid #424a53;padding:16px;font-size:18px;font-weight:700;text-align:center;border-radius:10px;margin:10px 0}
.nfl .expl{border:1px solid #d0d7de;border-radius:10px;padding:10px 16px;margin:10px 0}
.nfl .expl.verde{border-left:6px solid #2e9e5b}.nfl .expl.amarillo{border-left:6px solid #d9a400}.nfl .expl.gris{border-left:6px solid #8c959f}
.nfl .expl p{margin:4px 0}.nfl .barra{height:10px;border-radius:5px;background:#d0d7de;overflow:hidden;width:90px;display:inline-block;vertical-align:middle}
.nfl .barra>span{display:block;height:100%;background:#2a78d6}
.nfl details{border:1px solid #d0d7de;border-radius:8px;padding:6px 12px;margin:6px 0;background:#fff}
.nfl summary{cursor:pointer;font-weight:600}
.nfl .num-pos{color:#1a7f37;font-weight:600}.nfl .num-neg{color:#cf222e}
</style>"""

CLASE = {'Apostar': 'verde', 'Esperar': 'amarillo', 'No apostar': 'gris', 'Problema de datos': 'rojo', 'Jugado': 'jugado'}
CHIP = {'Apostar': ('c-verde', '🟢 Apostar (cumple criterios)'), 'Esperar': ('c-amarillo', '🟡 Esperar / revisar'),
        'No apostar': ('c-gris', '⚪ No apostar'), 'Problema de datos': ('c-rojo', '🔴 Problema de datos'),
        'Jugado': ('c-azul', '🔵 Jugado')}
INFO_PARTIDO = temp.set_index('game_id')
_ml_local = op[(op['mercado'] == 'Moneyline') & (op['lado'] == 'local')].set_index('game_id')['p_mercado']
INFO_PARTIDO['p_mercado'] = _ml_local.reindex(INFO_PARTIDO.index)          # mercado de la fuente de cuotas usada


DIAS = ['lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom']


def fecha_corta(d):
    return f'{DIAS[d.weekday()]} {d:%d/%m}'


def pct(x, signo=False):
    return '—' if pd.isna(x) else (f'{x:+.1%}' if signo else f'{x:.0%}')


def num(x, f='{:.2f}'):
    return '—' if pd.isna(x) else f.format(x)


def describir(r):
    eqp = r.home_team if r.lado == 'local' else r.away_team
    if r.mercado == 'Moneyline':
        return f'<b>Gana {eqp}</b><br><small>Moneyline (ganador del partido)</small>'
    if r.mercado == 'Spread':
        h = r.linea
        if pd.isna(h):
            return f'{eqp} (spread sin línea)'
        if h < 0:
            n = math.floor(-h) + 1
            cond = f'{eqp} debe ganar por {n} o más' + (' (por exactamente ' + f'{-h:g} = devolución)' if float(h).is_integer() else '')
        elif h > 0:
            cond = f'{eqp} gana, empata o pierde por {math.ceil(h) - 1} o menos' + (' (por exactamente ' + f'{h:g} = devolución)' if float(h).is_integer() else '')
        else:
            cond = f'gana {eqp} (empate = devolución)'
        return f'<b>{eqp} {h:+g}</b><br><small>Spread: {cond}</small>'
    txt = 'Más' if r.lado == 'over' else 'Menos'
    return f'<b>{txt} de {r.linea:g} puntos</b><br><small>Total ({r.lado}): suma de puntos de ambos equipos</small>'


def resultado_jugado(r):
    if r.mercado == 'Total':
        txt = f'Total real: {r.total:.0f}.'
    else:
        txt = f'Marcador: {r.away_team} {INFO_PARTIDO.loc[r.game_id, "away_score"]:.0f} – {INFO_PARTIDO.loc[r.game_id, "home_score"]:.0f} {r.home_team}.'
    if pd.isna(r.gana):
        return txt
    return txt + (' Esta opción habría sido devolución.' if r.push == 1 else ' Esta opción habría ganado.' if r.gana == 1 else ' Esta opción habría perdido.')


def resumen_semana(sem):
    d = mejores_op[mejores_op['week'] == sem]
    pend = d[d['decision'] != 'Jugado']
    por_partido = pend.groupby('game_id')['decision'].agg(lambda s: 'Apostar' if (s == 'Apostar').any() else
                                                          'Info' if s.isin(['Esperar', 'Problema de datos']).any() else 'No')
    return d, pend, por_partido


def html_panel(sem, mostrar_todos=False):
    d, pend, por_partido = resumen_semana(sem)
    h = [ESTILO, '<div class="nfl">', f'<h2>🏈 Semana {sem} · Temporada {TEMPORADA}</h2>']
    # ---- A. Resumen ----
    verdes = pend[pend['decision'] == 'Apostar']
    tops = ', '.join(f'{r.away_team} @ {r.home_team} ({r.mercado})' for r in verdes.head(3).itertuples()) or 'ninguna'
    h.append('<div class="tarjetas">'
             f'<div class="tarjeta t-azul"><div class="num">{d["game_id"].nunique()}</div><div class="txt">partidos en la semana '
             f'({len(por_partido)} pendientes analizados)</div></div>'
             f'<div class="tarjeta t-verde"><div class="num">{(por_partido == "Apostar").sum()}</div><div class="txt">con oportunidades que '
             f'superan los filtros</div></div>'
             f'<div class="tarjeta t-amarillo"><div class="num">{(por_partido == "Info").sum()}</div><div class="txt">requieren más '
             f'información</div></div>'
             f'<div class="tarjeta t-gris"><div class="num">{(por_partido == "No").sum()}</div><div class="txt">no justifican una '
             f'apuesta</div></div></div>')
    h.append(f'<p><b>Mejor evaluadas:</b> {tops}.</p>')
    fuentes_txt = '; '.join(f'{f["Fuente"]} (publicado: {f["Publicado por la fuente"] or "—"})' for f in FUENTES
                            if 'jugada' not in f['Fuente'] or str(TEMPORADA) in f['Fuente'])
    avisos = ''.join(f'<li>{a}</li>' for a in dict.fromkeys(AVISOS_DATOS))
    h.append(f'<div class="nota">🕒 <b>Datos actualizados:</b> análisis generado el {HOY.strftime("%d/%m/%Y %H:%M UTC")}. '
             f'Fuentes: {fuentes_txt}.<br>💱 <b>Cuotas:</b> {ESTADO_CUOTAS}<br>🧠 Modelo {MODELO_ELEGIDO}, versión {VERSION_MODELO}.'
             + (f'<br>⚠️ <b>Limitaciones detectadas:</b><ul style="margin:2px 0">{avisos}</ul>' if avisos else '') + '</div>')
    if not HABILITADO or not any(HABILITADO.values()):
        h.append('<div class="aviso">ℹ️ En el backtesting ningún mercado mostró una ventaja estable frente a las cuotas de cierre, '
                 'así que el sistema es especialmente exigente: puede marcar <b>No apostar</b> en todos los partidos. '
                 'Eso es el comportamiento correcto, no un error.</div>')
    if pend.empty:
        h.append('<div class="nota">Todos los partidos de esta semana ya se jugaron: se muestran en <b>modo revisión</b>.</div>')
    elif verdes.empty:
        h.append('<div class="noapostar">NO APOSTAR: no se encontraron oportunidades con suficiente respaldo estadístico.</div>')
    # ---- B. Tabla ----
    filas = d if (mostrar_todos or pend.empty) else d[d['decision'] != 'No apostar']
    h.append('<h3>📋 Tabla de recomendaciones</h3>')
    if filas.empty:
        h.append('<p>No hay oportunidades destacadas. Marca <i>"Mostrar también partidos sin ventaja"</i> para ver el análisis completo.</p>')
    else:
        monto = CONFIG['MOSTRAR_MONTOS_SUGERIDOS']
        h.append('<table class="rec"><tr><th>Partido</th><th>Recomendación</th><th>Prob. estimada</th><th>Cuota disponible</th>'
                 '<th>Cuota justa</th><th>Ventaja</th><th>Valor esperado</th><th>Confianza</th><th>Decisión</th><th>Motivo</th>'
                 + ('<th>Importe hipotético</th>' if monto else '') + '</tr>')
        for r in filas.itertuples():
            cls, txt = CHIP[r.decision]
            motivo = resultado_jugado(r) if r.decision == 'Jugado' else r.motivo
            ev_cls = 'num-pos' if (r.ev or 0) > 0 else 'num-neg'
            h.append(f'<tr class="{CLASE[r.decision]}"><td><b>{r.away_team} @ {r.home_team}</b><br><small>{fecha_corta(r.gameday)}</small></td>'
                     f'<td>{describir(r)}</td><td>{pct(r.p_final)}<br><small>± {r.incertidumbre * 100:.1f} pts</small></td>'
                     f'<td>{num(r.cuota)}<br><small>{r.fuente_cuota}</small></td><td>{num(r.cuota_justa)}</td>'
                     f'<td>{pct(r.ventaja, True)}</td><td class="{ev_cls}">{pct(r.ev, True)}<br><small>por unidad</small></td>'
                     f'<td><span class="barra"><span style="width:{r.confianza:.0f}%"></span></span> {r.nivel_confianza}<br>'
                     f'<small>{r.confianza:.0f}/100</small></td><td><span class="chip {cls}">{txt}</span></td><td>{motivo}</td>'
                     + (f'<td>{"%.2f" % r.monto if r.decision == "Apostar" else "—"}</td>' if monto else '') + '</tr>')
        h.append('</table>')
        if CONFIG['MOSTRAR_MONTOS_SUGERIDOS']:
            h.append(f'<p style="font-size:12px;color:#57606a">Importe hipotético sobre un capital de {CONFIG["CAPITAL_HIPOTETICO"]:.0f}: '
                     f'Kelly fraccional ({CONFIG["FRACCION_KELLY"]:.0%} de Kelly), máximo {CONFIG["TOPE_POR_APUESTA"]:.0%} por apuesta y '
                     f'{CONFIG["TOPE_SEMANAL"]:.0%} por semana. Puedes ocultarlo con MOSTRAR_MONTOS_SUGERIDOS = False.</p>')
    # ---- C. Explicaciones ----
    destacadas = pend[pend['decision'].isin(['Apostar', 'Esperar'])].head(6)
    h.append('<h3>💬 Explicación de las oportunidades</h3>')
    if destacadas.empty and not pend.empty:
        cercanas = pend[pend['cuota'].notna()].sort_values('ev', ascending=False).head(3)
        h.append('<p>No hay oportunidades recomendables. Para que aprendas a leer el análisis, estas son las <b>3 opciones '
                 'más cercanas</b> a cumplir los filtros (<b>no son recomendaciones</b>):</p>')
        destacadas = cercanas
    for r in destacadas.itertuples():
        h.append(explicacion(r))
    h.append('</div>')
    return ''.join(h)


# ---- Factores que explican cada predicción ----
FACTORES_LADO = [  # variable, nombre, signo (+1 = valor alto favorece al local), columnas local/visitante, formato
    ('elo_diff', 'Rating Elo (fuerza general, incluye ventaja de campo)', 1, 'elo_h', 'elo_a', '{:.0f}'),
    ('d_adj_off', 'Ataque ajustado por la calidad de los rivales (EPA por jugada)', 1, 'h_pre_adj_off', 'a_pre_adj_off', '{:+.3f}'),
    ('d_adj_def', 'Defensa ajustada por rivales (EPA permitido; más bajo = mejor)', -1, 'h_pre_adj_def', 'a_pre_adj_def', '{:+.3f}'),
    ('d_qb_val', 'Calidad del quarterback (EPA por dropback, últimos 16 partidos)', 1, 'h_qb_val', 'a_qb_val', '{:+.3f}'),
    ('d_sr', 'Success rate ofensivo (% de jugadas exitosas)', 1, 'h_pre_sr', 'a_pre_sr', '{:.1%}'),
    ('d_l3_margin', 'Forma reciente (margen medio en los últimos 3 partidos)', 1, 'h_pre_l3_margin', 'a_pre_l3_margin', '{:+.1f}'),
    ('d_perdidas', 'Pérdidas de balón por partido (más bajo = mejor)', -1, 'h_pre_perdidas', 'a_pre_perdidas', '{:.2f}'),
    ('d_rest', 'Días de descanso', 1, 'home_rest', 'away_rest', '{:.0f}'),
    ('d_lesionados', 'Bajas relevantes en el reporte de lesiones (más bajo = mejor)', -1, 'h_lesionados', 'a_lesionados', '{:.1f}'),
]
FACTORES_TOTAL = [  # variable, nombre, signo (+1 = valor alto favorece el over), formato
    ('s_adj_off', 'Ataques combinados ajustados por rivales', 1, '{:+.3f}'),
    ('s_adj_def', 'Defensas combinadas (EPA permitido; más alto = defensas más débiles)', 1, '{:+.3f}'),
    ('s_pts_for', 'Puntos anotados por partido (suma de ambos)', 1, '{:.1f}'),
    ('s_jugadas', 'Ritmo de juego (jugadas ofensivas por partido, suma)', 1, '{:.0f}'),
    ('viento', 'Viento previsto (mph; 0 si es estadio cubierto)', -1, '{:.0f}'),
    ('s_qb_val', 'Calidad combinada de los quarterbacks', 1, '{:+.3f}'),
]
_REF = jugados[[f[0] for f in FACTORES_LADO] + [f[0] for f in FACTORES_TOTAL]].agg(['mean', 'std'])


def factores(r):
    """Lista de (z, texto) con z > 0 si el factor favorece la opción analizada."""
    g = INFO_PARTIDO.loc[r.game_id]
    out = []
    if r.mercado == 'Total':
        dir_ = 1 if r.lado == 'over' else -1
        for var, nombre, signo, fmt in FACTORES_TOTAL:
            v = g[var]
            if pd.isna(v):
                continue
            z = dir_ * signo * (v - _REF.loc['mean', var]) / _REF.loc['std', var]
            out.append((z, f'{nombre}: {fmt.format(v)} (media histórica {fmt.format(_REF.loc["mean", var])})'))
        dif = g['total_pred'] - r.linea
        out.append((dir_ * dif / 3, f'El modelo estima {g["total_pred"]:.1f} puntos frente a una línea de {r.linea:g}'))
    else:
        dir_ = 1 if r.lado == 'local' else -1
        for var, nombre, signo, ch, ca, fmt in FACTORES_LADO:
            v = g[var]
            if pd.isna(v) or pd.isna(_REF.loc['std', var]) or _REF.loc['std', var] == 0:
                continue
            z = dir_ * signo * (v - _REF.loc['mean', var]) / _REF.loc['std', var]
            extra = f' ({g["h_qb_name"]} vs {g["a_qb_name"]})' if var == 'd_qb_val' else ''
            out.append((z, f'{nombre}: {r.home_team} {num(g[ch], fmt)} vs {r.away_team} {num(g[ca], fmt)}{extra}'))
    return sorted(out, key=lambda t: -t[0])


def explicacion(r):
    g = INFO_PARTIDO.loc[r.game_id]
    fs = factores(r)
    a_favor = [t for z, t in fs if z > 0.15][:3] or ['Ningún factor destaca claramente; la diferencia viene de pequeñas ventajas acumuladas.']
    en_contra = [t for z, t in sorted(fs, key=lambda t: t[0]) if z < -0.15][:2]
    eqp = r.home_team if r.lado == 'local' else r.away_team if r.lado == 'visitante' else None
    mercado_txt = {'Moneyline': f'si <b>{eqp}</b> gana el partido',
                   'Spread': f'si <b>{eqp}</b> cubre el hándicap de {r.linea:+g} puntos',
                   'Total': f'si se anotan <b>{"más" if r.lado == "over" else "menos"} de {r.linea:g}</b> puntos entre los dos equipos'}[r.mercado]
    riesgos = en_contra + list(r.amarillas) + [
        f'Incluso con buenos datos, los partidos individuales son muy aleatorios: el margen final suele desviarse '
        f'±{g["sigma_margen"]:.0f} puntos de lo previsto.',
        'El modelo no conoce noticias de última hora (cambios de alineación, clima extremo, motivación) si no están en los datos.']
    cls = {'Apostar': 'verde', 'Esperar': 'amarillo'}.get(r.decision, 'gris')
    gana10, neto10 = 10 * r.cuota, 10 * (r.cuota - 1)
    valor = 'la casa paga <b>más</b> de lo que el modelo considera justo → posible valor' if r.cuota > r.cuota_justa else \
        'la casa paga <b>menos</b> de lo que el modelo considera justo → no hay valor a este precio'
    veredicto = {'Apostar': f'Sí, según los filtros estadísticos. {r.motivo} Aun así, con una probabilidad estimada '
                            f'del {r.p_final:.0%}, <b>se perdería aproximadamente {1 - r.p_final:.0%} de las veces</b>. '
                            'Úsala solo con un importe pequeño y dentro de tu límite.',
                 'Esperar': f'{r.motivo} Vuelve a ejecutar el cuaderno cuando se publique esa información.'}.get(
        r.decision, f'No supera los filtros: {r.motivo}')
    return (f'<div class="expl {cls}"><p style="font-size:15px"><b>{r.away_team} @ {r.home_team}</b> · {r.mercado} · '
            f'<span class="chip {CHIP[r.decision][0]}">{CHIP[r.decision][1]}</span></p>'
            f'<p><b>¿Qué está analizando el modelo?</b> El partido {r.away_team} (visitante) contra {r.home_team} (local) del '
            f'{r.gameday:%d/%m}. Se analiza {mercado_txt}. Probabilidad estimada: <b>{r.p_final:.0%}</b>.</p>'
            f'<p><b>¿Por qué considera interesante esta opción?</b></p><ul>{"".join(f"<li>{t}</li>" for t in a_favor)}</ul>'
            f'<p><b>¿Qué podría salir mal?</b></p><ul>{"".join(f"<li>{t}</li>" for t in riesgos)}</ul>'
            f'<p><b>¿Qué significa la cuota?</b> Cuota {r.cuota:.2f}: si apuestas 10 y aciertas cobras {gana10:.2f} '
            f'(ganancia neta {neto10:.2f}); si fallas pierdes los 10. Esa cuota implica una probabilidad del {r.p_implicita:.0%} '
            f'({r.p_mercado:.0%} sin el margen de la casa). Según el modelo, la cuota justa sería {r.cuota_justa:.2f}: {valor}. '
            f'Valor esperado: {r.ev:+.1%} por unidad (en promedio, a largo plazo, {10 * r.ev:+.2f} por cada 10 apostados).</p>'
            f'<p><b>¿Vale la pena considerarla?</b> {veredicto}</p></div>')


GLOSARIO = ESTILO + """<div class="nfl"><h3>📖 Glosario para principiantes (haz clic para desplegar)</h3>
<details><summary>Probabilidad estimada</summary>Lo que calcula el modelo sobre la posibilidad de que ocurra un resultado.
<i>Ejemplo:</i> 60 % significa que, en 100 partidos con condiciones parecidas, ese resultado ocurriría unas 60 veces… y fallaría unas 40.</details>
<details><summary>Cuota</summary>El multiplicador que usa la casa para calcular el pago (formato decimal). <i>Ejemplo:</i> apuestas 10 a cuota 2.50 →
si aciertas cobras 25 (ganancia neta 15); si fallas pierdes 10. En formato americano, -150 equivale a 1.67 y +150 a 2.50.</details>
<details><summary>Probabilidad implícita y margen de la casa</summary>La probabilidad que "esconde" una cuota: 1 ÷ cuota. Cuota 1.91 → 52.4 %.
Si sumas las dos opciones de un partido (1.91 y 1.91) obtienes 104.8 %: ese 4.8 % extra es el margen de la casa. Quitándolo se obtiene la
probabilidad "sin margen" (50 % y 50 %).</details>
<details><summary>Cuota justa</summary>La cuota teórica correspondiente a una probabilidad: 1 ÷ probabilidad. <i>Ejemplo:</i> si el modelo da 55 %,
la cuota justa es 1 ÷ 0.55 = 1.82. Si la casa ofrece 2.00 (más que 1.82), hay posible valor; si ofrece 1.70, no.</details>
<details><summary>Ventaja estimada</summary>Diferencia entre la probabilidad del modelo y la del mercado sin margen. <i>Ejemplo:</i> modelo 55 %, mercado 51 % → ventaja +4 puntos.</details>
<details><summary>Valor esperado (EV)</summary>El resultado económico promedio teórico si repitieras muchas veces la misma apuesta en las mismas
condiciones: EV = probabilidad × cuota − 1. <i>Ejemplo:</i> 55 % × 2.00 − 1 = +0.10 → +10 % (en promedio +1 por cada 10 apostados).</details>
<details><summary>Valor esperado positivo</summary>Una <b>señal</b> de posible ventaja estadística, <b>no una garantía</b>. Con EV +5 % y
probabilidad 50 % puedes perder 5 apuestas seguidas con bastante facilidad (ocurre ~3 % de las veces por puro azar).</details>
<details><summary>Bankroll (capital)</summary>El dinero separado exclusivamente para apostar, que podrías perder sin afectar a tus gastos.
<i>Ejemplo:</i> si separas 500, una apuesta del 1 % son 5.</details>
<details><summary>ROI</summary>Rendimiento porcentual sobre el dinero apostado. <i>Ejemplo:</i> apuestas 100 en total y terminas con +4 → ROI +4 %.</details>
<details><summary>Drawdown</summary>La caída desde el máximo anterior del capital. <i>Ejemplo:</i> llegas a 1 200 y luego bajas a 900 → drawdown de 300 (25 %).</details>
<details><summary>Moneyline</summary>Apostar a quién gana el partido. En la NFL un empate (muy raro) suele devolverse.</details>
<details><summary>Spread (hándicap)</summary>Una línea de puntos que modifica las condiciones. <i>Ejemplo:</i> DAL −3.5 significa que DAL debe ganar por 4 o más;
TB +3.5 gana la apuesta si TB gana o pierde por 3 o menos. Con líneas enteras (−3) un resultado exacto se devuelve (push).</details>
<details><summary>Over/under (total)</summary>Apostar si el total de puntos supera (over) o queda por debajo (under) de una línea.
<i>Ejemplo:</i> línea 44.5; el partido acaba 24-21 (45 puntos) → gana el over.</details>
<details><summary>Confianza del modelo</summary>Puntuación 0-100 que baja cuando los modelos no se ponen de acuerdo, es inicio de temporada, falta el QB
confirmado, el reporte de lesiones, el clima o la cuota no está verificada.</details>
<details><summary>Kelly fraccional</summary>Fórmula que sugiere qué parte del capital apostar según la ventaja. Aquí se usa solo 1/4 de Kelly y con
tope del 2 % por apuesta, porque Kelly completo es muy agresivo si el modelo se equivoca.</details>
<details><summary>Backtesting</summary>Simular la estrategia en temporadas pasadas con las cuotas reales de entonces. Son resultados
<b>simulados</b>, no ganancias reales, y el pasado no garantiza el futuro.</details>
</div>"""


# ---- E. Análisis individual ----
def comparar_equipos(gid):
    g = INFO_PARTIDO.loc[gid]
    filas = [('Elo (fuerza general)', 'elo_h', 'elo_a', '{:.0f}', 1), ('Ataque ajustado (EPA/jugada)', 'h_pre_adj_off', 'a_pre_adj_off', '{:+.3f}', 1),
             ('Defensa ajustada (EPA permitido)', 'h_pre_adj_def', 'a_pre_adj_def', '{:+.3f}', -1),
             ('EPA ofensiva por jugada', 'h_pre_epa', 'a_pre_epa', '{:+.3f}', 1), ('EPA defensiva permitida', 'h_pre_def_epa', 'a_pre_def_epa', '{:+.3f}', -1),
             ('Success rate ofensivo', 'h_pre_sr', 'a_pre_sr', '{:.1%}', 1), ('Success rate permitido', 'h_pre_def_sr', 'a_pre_def_sr', '{:.1%}', -1),
             ('EPA en pases', 'h_pre_pass_epa', 'a_pre_pass_epa', '{:+.3f}', 1), ('EPA en carreras', 'h_pre_rush_epa', 'a_pre_rush_epa', '{:+.3f}', 1),
             ('Puntos anotados / partido', 'h_pre_pts_for', 'a_pre_pts_for', '{:.1f}', 1), ('Puntos recibidos / partido', 'h_pre_pts_against', 'a_pre_pts_against', '{:.1f}', -1),
             ('Yardas / partido', 'h_pre_yardas', 'a_pre_yardas', '{:.0f}', 1), ('Yardas permitidas / partido', 'h_pre_def_yardas', 'a_pre_def_yardas', '{:.0f}', -1),
             ('Pérdidas de balón / partido', 'h_pre_perdidas', 'a_pre_perdidas', '{:.2f}', -1), ('Recuperaciones / partido', 'h_pre_def_perdidas', 'a_pre_def_perdidas', '{:.2f}', 1),
             ('3ª oportunidad convertida', 'h_pre_tercera', 'a_pre_tercera', '{:.1%}', 1), ('TD en zona roja', 'h_pre_zona_roja', 'a_pre_zona_roja', '{:.1%}', 1),
             ('Margen como local/visitante', 'h_pre_margen_sede', 'a_pre_margen_sede', '{:+.1f}', 1),
             ('Forma: margen últimos 3', 'h_pre_l3_margin', 'a_pre_l3_margin', '{:+.1f}', 1),
             ('Valor del QB', 'h_qb_val', 'a_qb_val', '{:+.3f}', 1), ('Días de descanso', 'home_rest', 'away_rest', '{:.0f}', 1),
             ('Bajas relevantes (Out/Doubtful)', 'h_lesionados', 'a_lesionados', '{:.1f}', -1)]
    h = [f'<table class="rec"><tr><th>Estadística (antes del partido)</th><th>{g["home_team"]} (local)</th><th>{g["away_team"]} (visitante)</th></tr>']
    for nombre, ch, ca, fmt, signo in filas:
        vh, va = g[ch], g[ca]
        mejor_h = pd.notna(vh) and pd.notna(va) and signo * (vh - va) > 0
        mejor_a = pd.notna(vh) and pd.notna(va) and signo * (va - vh) > 0
        h.append(f'<tr><td>{nombre}</td><td>{num(vh, fmt)} {"▲" if mejor_h else ""}</td><td>{num(va, fmt)} {"▲" if mejor_a else ""}</td></tr>')
    h.append('</table><p style="font-size:12px;color:#57606a">▲ = mejor en esa estadística. Medias móviles que pesan más los '
             'partidos recientes; QB: ' + f'{g["h_qb_name"]} vs {g["a_qb_name"]}' + '.</p>')
    return ''.join(h)


def grafico_evolucion(gid):
    g = INFO_PARTIDO.loc[gid]
    fig = go.Figure()
    for t, color in [(g['home_team'], '#2a78d6'), (g['away_team'], '#e8890c')]:
        d = eq[(eq['team'] == t) & eq['jugado'] & (eq['gameday'] < g['gameday'])].tail(17)
        x = np.arange(-len(d), 0)                       # partidos hacia atrás (sin huecos por la pretemporada)
        info = [f'{dd:%d/%m/%Y} vs {o}' for dd, o in zip(d['gameday'], d['opp'])]
        fig.add_trace(go.Bar(x=x, y=d['margin'], name=f'{t}: margen del partido', marker_color=color, opacity=0.35,
                             hovertext=info, hovertemplate='%{hovertext}: %{y:+.0f}<extra></extra>'))
        fuerza = (d['epa_vs_esp'] - d['def_epa_vs_esp']).ewm(halflife=VIDA_MEDIA).mean() * 100
        fig.add_trace(go.Scatter(x=x, y=fuerza, name=f'{t}: fuerza neta ajustada (tendencia)', line=dict(color=color, width=3),
                                 hovertext=info, hovertemplate='%{hovertext}: %{y:+.1f}<extra></extra>'))
    fig.update_layout(title='Evolución reciente: margen de cada partido (barras) y tendencia de fuerza neta (línea, EPA ajustado ×100)',
                      template='plotly_white', height=380, barmode='group', legend=dict(orientation='h', y=-0.25),
                      xaxis_title='Partidos anteriores (−1 = el más reciente)')
    return fig


def grafico_probabilidad(gid):
    g = INFO_PARTIDO.loc[gid]
    p = g['p_local']
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['Victoria'], x=[p], orientation='h', name=g['home_team'], marker_color='#2a78d6',
                         text=[f'{g["home_team"]} {p:.0%}'], textposition='inside'))
    fig.add_trace(go.Bar(y=['Victoria'], x=[1 - p], orientation='h', name=g['away_team'], marker_color='#e8890c',
                         text=[f'{g["away_team"]} {1 - p:.0%}'], textposition='inside'))
    if pd.notna(g.get('p_mercado', np.nan)):
        fig.add_vline(x=g['p_mercado'], line_dash='dot', line_color='black')
        fig.add_annotation(x=g['p_mercado'], y=1.25, yref='paper', text=f'mercado {g["p_mercado"]:.0%}', showarrow=False)
    fig.update_layout(barmode='stack', height=170, template='plotly_white', showlegend=False, xaxis=dict(tickformat='.0%', range=[0, 1]),
                      margin=dict(l=10, r=10, t=60, b=20), title=f'Probabilidad estimada de victoria (modelo {MODELO_ELEGIDO}; línea = mercado)')
    return fig


def sensibilidad(gid):
    g = INFO_PARTIDO.loc[[gid]]
    base = float(g['p_local'].iloc[0])
    h = ['<table class="rec"><tr><th>Escenario</th><th>Prob. victoria local</th><th>Cambio</th></tr>']
    escenarios = [('Predicción actual', g)]
    for lado, pref in [('local', 'h_'), ('visitante', 'a_')]:
        alt = g.copy()
        alt[pref + 'qb_val'] = QB_PRIOR
        alt[pref + 'qb_cambio'] = 1
        alt['d_qb_val'] = alt['h_qb_val'] - alt['a_qb_val']
        alt['s_qb_val'] = alt['h_qb_val'] + alt['a_qb_val']
        escenarios.append((f'Si el QB {lado} ({g[pref + "qb_name"].iloc[0]}) no juega y entra un suplente sin historial', alt))
    alt = g.copy(); alt['d_rest'] = 0
    escenarios.append(('Si ambos equipos tuvieran el mismo descanso', alt))
    alt = g.copy(); alt['d_lesionados'] = 0
    escenarios.append(('Si ambos equipos tuvieran las mismas bajas', alt))
    for nombre, df in escenarios:
        p = float(predecir(MODELOS, df)['p_' + MODELO_ELEGIDO].iloc[0])
        h.append(f'<tr><td>{nombre}</td><td>{p:.1%}</td><td>{p - base:+.1%}</td></tr>')
    h.append('</table>')
    # Sensibilidad a la cuota
    d = op[(op['game_id'] == gid) & op['cuota'].notna()]
    if not d.empty:
        h.append('<p><b>¿Y si cambia la cuota?</b> Valor esperado de cada opción si la cuota baja o sube:</p>'
                 '<table class="rec"><tr><th>Opción</th><th>Cuota −10 %</th><th>−5 %</th><th>Actual</th><th>+5 %</th><th>+10 %</th>'
                 '<th>Cuota mínima para cumplir el umbral</th></tr>')
        for r in d.itertuples():
            evs = [(1 - r.p_push) * (r.p_final * r.cuota * (1 + k) - 1) for k in (-0.10, -0.05, 0, 0.05, 0.10)]
            minima = (1 + UMBRAL_EV[r.mercado] / max(1 - r.p_push, 1e-9)) / r.p_final
            celdas = ''.join(f'<td class="{"num-pos" if e > 0 else "num-neg"}">{e:+.1%}</td>' for e in evs)
            h.append(f'<tr><td>{describir(r)}</td>{celdas}<td>{minima:.2f}</td></tr>')
        h.append('</table>')
    w = PESOS_MODELO.get('Moneyline', 0)
    pm = INFO_PARTIDO.loc[gid, 'p_mercado'] if 'p_mercado' in INFO_PARTIDO else np.nan
    if pd.notna(pm):
        h.append(f'<p style="font-size:13px">Peso del modelo en la probabilidad final de moneyline: {w:.0%} (elegido en el backtesting). '
                 f'Solo modelo: {base:.0%} · solo mercado: {pm:.0%} · mezcla usada: {w * base + (1 - w) * pm:.0%}.</p>')
    return ''.join(h)


def panel_partido(gid):
    g = INFO_PARTIDO.loc[gid]
    display(HTML(ESTILO + f'<div class="nfl"><h2>{g["away_team"]} @ {g["home_team"]} · semana {g["week"]} · {g["gameday"]:%d/%m/%Y}</h2>'
                 f'<p>Margen estimado: <b>{g["home_team"]} {g["margen_pred"]:+.1f}</b> · Puntos totales estimados: '
                 f'<b>{g["total_pred"]:.1f}</b> · Estadio: {g.get("stadium", "—")} ({g["roof"] if pd.notna(g["roof"]) else "techo desconocido"})'
                 + (f' · Viento {g["wind"]:.0f} mph, {g["temp"]:.0f} °F' if pd.notna(g['wind']) else '') + '</p></div>'))
    grafico_probabilidad(gid).show()
    d = op[op['game_id'] == gid]
    filas = ''.join(f'<tr><td>{describir(r)}</td><td>{pct(r.p_final)}</td><td>{num(r.cuota)}</td><td>{num(r.cuota_justa)}</td>'
                    f'<td>{pct(r.ev, True)}</td></tr>' for r in d.itertuples())
    mejores_d = mejores_op[mejores_op['game_id'] == gid]
    fs = factores(mejores_d.iloc[0]) if len(mejores_d) and mejores_d.iloc[0]['mercado'] != 'Total' else \
        factores(d[(d['mercado'] == 'Moneyline') & (d['lado'] == 'local')].iloc[0])
    favor_local = [t for z, t in fs if z > 0.15][:4] if fs else []
    contra = [t for z, t in sorted(fs, key=lambda x: x[0]) if z < -0.15][:4]
    lado_txt = mejores_d.iloc[0]['lado'] if len(mejores_d) and mejores_d.iloc[0]['mercado'] != 'Total' else 'local'
    eq_ref = g['home_team'] if lado_txt == 'local' else g['away_team']
    display(HTML(ESTILO + '<div class="nfl"><h3>💱 Mercados y cuotas</h3><table class="rec"><tr><th>Opción</th><th>Prob. estimada</th>'
                 f'<th>Cuota</th><th>Cuota justa</th><th>Valor esperado</th></tr>{filas}</table>'
                 f'<h3>⚖️ Principales factores para {eq_ref}</h3><div style="display:flex;gap:16px;flex-wrap:wrap">'
                 f'<div style="flex:1 1 300px"><b>A favor</b><ul>{"".join(f"<li>{t}</li>" for t in favor_local) or "<li>—</li>"}</ul></div>'
                 f'<div style="flex:1 1 300px"><b>En contra</b><ul>{"".join(f"<li>{t}</li>" for t in contra) or "<li>—</li>"}</ul></div></div>'
                 '<h3>📊 Comparación de equipos</h3>' + comparar_equipos(gid) + '</div>'))
    grafico_evolucion(gid).show()
    display(HTML(ESTILO + '<div class="nfl"><h3>🔬 Sensibilidad de la predicción</h3>' + sensibilidad(gid) + '</div>'))


# ---- Widgets ----
sel_semana = widgets.Dropdown(options=[(f'Semana {w}', int(w)) for w in SEMANAS], value=SEMANA_OBJETIVO, description='Semana:')
chk_todos = widgets.Checkbox(value=False, description='Mostrar también partidos sin ventaja', indent=False)
out_panel = widgets.Output()


def refrescar_panel(_=None):
    with out_panel:
        clear_output(wait=True)
        display(HTML(html_panel(sel_semana.value, chk_todos.value)))


sel_semana.observe(refrescar_panel, 'value')
chk_todos.observe(refrescar_panel, 'value')

sel_semana2 = widgets.Dropdown(options=[(f'Semana {w}', int(w)) for w in SEMANAS], value=SEMANA_OBJETIVO, description='Semana:')
sel_partido = widgets.Dropdown(description='Partido:', layout=widgets.Layout(width='380px'))
out_partido = widgets.Output()


def opciones_partidos(_=None):
    d = temp[temp['week'] == sel_semana2.value].sort_values('gameday')
    sel_partido.options = [(f'{r.away_team} @ {r.home_team} ({r.gameday:%d/%m})', r.game_id) for r in d.itertuples()]


def refrescar_partido(_=None):
    with out_partido:
        clear_output(wait=True)
        if sel_partido.value:
            panel_partido(sel_partido.value)


sel_semana2.observe(opciones_partidos, 'value')
sel_partido.observe(refrescar_partido, 'value')
opciones_partidos()

display(widgets.HBox([sel_semana, chk_todos]), out_panel)
refrescar_panel()
display(HTML(GLOSARIO))
display(HTML(ESTILO + '<div class="nfl"><h2>🔎 Análisis individual de un partido</h2><p>Elige semana y partido:</p></div>'))
display(widgets.HBox([sel_semana2, sel_partido]), out_partido)
refrescar_partido()

# %% [markdown]
# ## 1️⃣3️⃣ 🎲 Simulador de riesgo (todo hipotético)
# Prueba qué podría pasar con tu capital. Puedes cargar una opción analizada o escribir tus propios números.
#
# * **Criterio de riesgo:** *monto fijo*, *porcentaje fijo del capital* o *Kelly fraccional con tope*.
# * El simulador **nunca** aumenta la apuesta para recuperar pérdidas (no usa martingala).
# * Se simulan 2 000 trayectorias posibles con la probabilidad indicada para ver el rango de resultados.
#
# > Recuerda: la probabilidad es una **estimación** del modelo. Si el modelo se equivoca, el resultado real será peor.

# %%
_opciones_sim = [('— Escribir mis propios números —', None)] + [
    (f'S{r.week} · {r.away_team} @ {r.home_team} · {r.mercado} {r.lado} · cuota {r.cuota:.2f} · prob {r.p_final:.0%}', i)
    for i, r in mejores_op[(~mejores_op['jugado']) & mejores_op['cuota'].notna()].iterrows()]
w_op = widgets.Dropdown(options=_opciones_sim, description='Opción:', layout=widgets.Layout(width='620px'))
w_mercado = widgets.Dropdown(options=['Moneyline', 'Spread', 'Total'], description='Mercado:')
w_capital = widgets.FloatText(value=CONFIG['CAPITAL_HIPOTETICO'], description='Capital:')
w_cuota = widgets.BoundedFloatText(value=1.91, min=1.01, max=50, step=0.01, description='Cuota:')
w_prob = widgets.BoundedFloatText(value=52.5, min=1, max=99, step=0.5, description='Prob. (%):')
w_monto = widgets.BoundedFloatText(value=10, min=0, max=1e7, description='Monto fijo:')
w_pct = widgets.BoundedFloatText(value=1.0, min=0.1, max=5, step=0.1, description='% capital:')
w_n = widgets.BoundedIntText(value=100, min=1, max=2000, description='Nº apuestas:')
w_crit = widgets.Dropdown(options=['Monto fijo', 'Porcentaje fijo del capital', 'Kelly fraccional (con tope)'],
                          value='Porcentaje fijo del capital', description='Criterio:')
w_boton = widgets.Button(description='Simular', button_style='primary', icon='play')
out_sim = widgets.Output()


def cargar_opcion(_=None):
    if w_op.value is None:
        return
    r = mejores_op.loc[w_op.value]
    w_cuota.value, w_prob.value, w_mercado.value = round(float(r['cuota']), 2), round(float(r['p_final']) * 100, 1), r['mercado']


def simular(capital, cuota, p, n, criterio, monto, pct_cap, trayectorias=2000, semilla=1):
    rng = np.random.default_rng(semilla)
    gana = rng.random((trayectorias, n)) < p
    cap = np.full(trayectorias, float(capital))
    historia = np.empty((trayectorias, n + 1)); historia[:, 0] = cap
    kelly = max(0.0, (p * cuota - 1) / (cuota - 1)) * CONFIG['FRACCION_KELLY']
    for j in range(n):
        if criterio == 'Monto fijo':
            st = np.full(trayectorias, monto)
        elif criterio == 'Porcentaje fijo del capital':
            st = cap * pct_cap / 100
        else:
            st = cap * min(kelly, CONFIG['TOPE_POR_APUESTA'])
        st = np.minimum(st, np.maximum(cap, 0))            # nunca se apuesta más de lo que queda
        cap = cap + np.where(gana[:, j], st * (cuota - 1), -st)
        historia[:, j + 1] = cap
    return historia, kelly


def ejecutar_simulacion(_=None):
    with out_sim:
        clear_output(wait=True)
        p, cuota, capital, n = w_prob.value / 100, w_cuota.value, w_capital.value, w_n.value
        historia, kelly = simular(capital, cuota, p, n, w_crit.value, w_monto.value, w_pct.value)
        final = historia[:, -1]
        ev = p * cuota - 1
        primera = {'Monto fijo': w_monto.value, 'Porcentaje fijo del capital': capital * w_pct.value / 100}.get(
            w_crit.value, capital * min(kelly, CONFIG['TOPE_POR_APUESTA']))
        pico = np.maximum.accumulate(historia, axis=1)
        dd = np.max((pico - historia) / np.where(pico > 0, pico, 1), axis=1)
        aviso_ev = ('<div class="aviso">⚠️ Con estos números el valor esperado es <b>negativo</b>: a largo plazo se pierde dinero.</div>'
                    if ev <= 0 else '')
        texto_kelly = (f'Kelly completo sugeriría {max(0, (p * cuota - 1) / (cuota - 1)):.1%} del capital; se usa {CONFIG["FRACCION_KELLY"]:.0%} de eso '
                       f'= {kelly:.2%}, limitado a un máximo de {CONFIG["TOPE_POR_APUESTA"]:.0%} → {min(kelly, CONFIG["TOPE_POR_APUESTA"]):.2%} por apuesta.')
        display(HTML(ESTILO + f'<div class="nfl"><h3>Resultado de la simulación (HIPOTÉTICO) · {w_mercado.value}</h3>{aviso_ev}'
                     '<div class="tarjetas">'
                     f'<div class="tarjeta t-azul"><div class="num">{ev:+.1%}</div><div class="txt">valor esperado por unidad apostada</div></div>'
                     f'<div class="tarjeta t-azul"><div class="num">{primera * cuota:.2f}</div><div class="txt">pago potencial de la 1ª apuesta '
                     f'(apostando {primera:.2f})</div></div>'
                     f'<div class="tarjeta t-verde"><div class="num">{primera * (cuota - 1):+.2f}</div><div class="txt">ganancia neta si acierta la 1ª</div></div>'
                     f'<div class="tarjeta t-gris"><div class="num">{-primera:.2f}</div><div class="txt">pérdida si falla la 1ª</div></div></div>'
                     '<div class="tarjetas">'
                     f'<div class="tarjeta t-amarillo"><div class="num">{(final < capital).mean():.0%}</div><div class="txt">probabilidad de terminar '
                     f'con pérdidas tras {n} apuestas</div></div>'
                     f'<div class="tarjeta t-amarillo"><div class="num">{(final < capital * 0.5).mean():.1%}</div><div class="txt">probabilidad de '
                     f'perder más de la mitad del capital</div></div>'
                     f'<div class="tarjeta t-gris"><div class="num">{np.median(final):,.0f}</div><div class="txt">capital final mediano '
                     f'(5 %–95 %: {np.percentile(final, 5):,.0f} – {np.percentile(final, 95):,.0f})</div></div>'
                     f'<div class="tarjeta t-gris"><div class="num">{np.median(dd):.0%}</div><div class="txt">caída máxima típica (drawdown mediano)</div></div></div>'
                     f'<p style="font-size:13px">{texto_kelly if w_crit.value.startswith("Kelly") else ""} '
                     'Con probabilidades reales inferiores a las estimadas, estos resultados empeoran rápidamente.</p></div>'))
        x = np.arange(n + 1)
        fig = go.Figure()
        for lo, hi, alpha in [(5, 95, 0.15), (25, 75, 0.30)]:
            fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([np.percentile(historia, hi, axis=0),
                                                                                       np.percentile(historia, lo, axis=0)[::-1]]),
                                     fill='toself', fillcolor=f'rgba(42,120,214,{alpha})', line=dict(width=0),
                                     name=f'{lo}%–{hi}% de los casos', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x, y=np.median(historia, axis=0), name='Caso mediano', line=dict(color='#2a78d6', width=3)))
        for k in range(3):
            fig.add_trace(go.Scatter(x=x, y=historia[k], name=f'Ejemplo {k + 1}', line=dict(width=1, dash='dot'), opacity=0.7))
        fig.add_hline(y=capital, line_dash='dash', line_color='gray')
        fig.update_layout(title='Evolución del capital HIPOTÉTICO en 2 000 simulaciones', xaxis_title='Número de apuesta',
                          yaxis_title='Capital', template='plotly_white', height=420)
        fig.show()


w_op.observe(cargar_opcion, 'value')
w_boton.on_click(ejecutar_simulacion)
display(HTML(ESTILO + '<div class="nfl"><div class="aviso">🎲 Simulador <b>hipotético</b>. Sin martingala: el importe nunca aumenta para '
             'recuperar pérdidas. No uses dinero necesario para tus gastos.</div></div>'))
display(widgets.VBox([w_op, widgets.HBox([w_mercado, w_crit]), widgets.HBox([w_capital, w_cuota, w_prob]),
                      widgets.HBox([w_monto, w_pct, w_n]), w_boton, out_sim]))
ejecutar_simulacion()

# %% [markdown]
# ## 1️⃣4️⃣ Registro de predicciones, resultados y exportación
# **Qué hace:** guarda automáticamente cada predicción de partidos pendientes (fecha, partido, mercado, cuota,
# probabilidad, decisión, versión del modelo). Cuando el partido se juega, completa el resultado, la ganancia/pérdida
# **hipotética** (1 unidad y el importe sugerido) y el **CLV**: compara tu cuota con la de cierre (si consigues
# sistemáticamente mejores cuotas que el cierre, es buena señal).
#
# Revisa aquí cada semana la **calibración**: si el modelo dice 60 %, ¿se acierta ~60 %? El registro se puede descargar
# en CSV. Activa `GUARDAR_EN_DRIVE` en la celda 2 para conservarlo entre sesiones de Colab.

# %%
REGISTRO = CARPETA / 'registro' / 'registro_predicciones.csv'
CLAVE = ['game_id', 'mercado', 'lado', 'version_modelo']
_nuevo = mejores_op[~mejores_op['jugado'] & mejores_op['cuota'].notna()].assign(
    fecha_prediccion=HOY.strftime('%Y-%m-%d %H:%M UTC'), version_modelo=VERSION_MODELO,
    partido=lambda d: d['away_team'] + ' @ ' + d['home_team'])[
    ['fecha_prediccion', 'version_modelo', 'game_id', 'season', 'week', 'gameday', 'partido', 'mercado', 'lado', 'linea', 'cuota',
     'fuente_cuota', 'p_modelo', 'p_final', 'p_mercado', 'ev', 'confianza', 'decision', 'motivo', 'monto']]
_nuevo = _nuevo.rename(columns={'monto': 'monto_hipotetico'})
if REGISTRO.exists():
    registro = pd.read_csv(REGISTRO)
    registro['gameday'] = pd.to_datetime(registro['gameday'])
    _jugados_ids = set(cal.loc[cal['jugado'], 'game_id'])
    # Se actualiza la foto de los partidos que aún no empezaron; los ya jugados quedan congelados.
    _clave_nueva = _nuevo.set_index(CLAVE).index
    _mantener = registro['game_id'].isin(_jugados_ids) | ~registro.set_index(CLAVE).index.isin(_clave_nueva)
    registro = pd.concat([registro[_mantener.values], _nuevo], ignore_index=True)
else:
    registro = _nuevo.copy()

# Liquidación de partidos terminados + CLV frente a la cuota de cierre de nflverse
_res = partidos.set_index('game_id')
for col in ('resultado', 'unidades', 'ganancia_hipotetica', 'cuota_cierre', 'clv'):
    if col not in registro:
        registro[col] = np.nan
registro['resultado'] = registro['resultado'].astype(object)
for i, r in registro[registro['resultado'].isna()].iterrows():
    if r['game_id'] not in _res.index or not _res.loc[r['game_id'], 'jugado']:
        continue
    g = _res.loc[r['game_id']]
    if r['mercado'] == 'Moneyline':
        v = g['result'] if r['lado'] == 'local' else -g['result']
        cierre = g['ml_h_dec'] if r['lado'] == 'local' else g['ml_a_dec']
        linea_cierre = np.nan
    elif r['mercado'] == 'Spread':
        v = g['result'] + r['linea'] if r['lado'] == 'local' else -g['result'] + r['linea']
        cierre = g['sp_h_dec'] if r['lado'] == 'local' else g['sp_a_dec']
        linea_cierre = -g['spread_line'] if r['lado'] == 'local' else g['spread_line']
    else:
        v = g['total'] - r['linea'] if r['lado'] == 'over' else r['linea'] - g['total']
        cierre = g['ov_dec'] if r['lado'] == 'over' else g['un_dec']
        linea_cierre = g['total_line']
    u = 0.0 if v == 0 else (r['cuota'] - 1 if v > 0 else -1.0)
    registro.loc[i, 'resultado'] = 'Devolución' if v == 0 else ('Ganada' if v > 0 else 'Perdida')
    registro.loc[i, 'unidades'] = u
    registro.loc[i, 'ganancia_hipotetica'] = u * r['monto_hipotetico'] if r['decision'] == 'Apostar' else 0.0
    registro.loc[i, 'cuota_cierre'] = cierre
    misma_linea = r['mercado'] == 'Moneyline' or np.isclose(r['linea'], linea_cierre)
    registro.loc[i, 'clv'] = r['cuota'] / cierre - 1 if misma_linea and pd.notna(cierre) else np.nan

registro.sort_values(['gameday', 'partido', 'mercado']).to_csv(REGISTRO, index=False)
print(f'💾 Registro guardado en {REGISTRO} ({len(registro)} filas).')

_liq = registro[registro['resultado'].notna()]
_ap = _liq[_liq['decision'] == 'Apostar']
display(HTML(ESTILO + '<div class="nfl"><h3>📒 Resultados registrados (en vivo, no simulados)</h3>'
             f'<p>Predicciones liquidadas: {len(_liq)} · Recomendadas como "Apostar": {len(_ap)} · '
             f'Unidades de las recomendadas: {_ap["unidades"].sum():+.2f} · ganancia hipotética: {_ap["ganancia_hipotetica"].sum():+.2f} · '
             f'CLV medio: {pct(_ap["clv"].mean(), True) if len(_ap) else "—"}</p>'
             '<p style="font-size:13px;color:#57606a">Con pocas apuestas, estos números dependen sobre todo del azar. Evalúa el sistema '
             'con cientos de apuestas, no con una semana.</p></div>'))

# Calibración semanal de todas las predicciones liquidadas (no solo las apostadas)
_cal = _liq[_liq['resultado'] != 'Devolución'].copy()
if len(_cal) >= 20:
    _cal['acierto'] = (_cal['resultado'] == 'Ganada').astype(int)
    _sem = _cal.groupby(['season', 'week']).agg(predicciones=('acierto', 'size'), prob_media=('p_final', 'mean'),
                                                  aciertos=('acierto', 'mean'))
    _sem['brier'] = _cal.groupby(['season', 'week']).apply(lambda d: brier_score_loss(d['acierto'], d['p_final']))
    display(_sem.style.format({'prob_media': '{:.1%}', 'aciertos': '{:.1%}', 'brier': '{:.3f}'}))
    c = curva_calibracion(_cal['p_final'], _cal['acierto'], bins=5)
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dot', color='gray'), name='Perfecta'),
                     go.Scatter(x=c['p_media'], y=c['frec_real'], mode='lines+markers', name='Registro en vivo', text=c['n'],
                                hovertemplate='Estimado %{x:.0%} · real %{y:.0%} · n=%{text}<extra></extra>')])
    fig.update_layout(title='Calibración del registro en vivo', template='plotly_white', height=360,
                      xaxis_tickformat='.0%', yaxis_tickformat='.0%')
    fig.show()
else:
    print(f'ℹ️ La calibración semanal aparecerá cuando haya al menos 20 predicciones liquidadas (ahora: {len(_cal)}).')

display(registro.tail(10))
if EN_COLAB:
    boton_descarga = widgets.Button(description='Descargar registro CSV', icon='download')
    boton_descarga.on_click(lambda _: __import__('google.colab.files', fromlist=['download']).download(str(REGISTRO)))
    display(boton_descarga)

# %% [markdown]
# ## ✅ Lista de comprobación y limitaciones
# * **Datos:** nflverse es una fuente abierta mantenida por la comunidad; se actualiza varias veces al día durante la
#   temporada, pero puede tener retrasos. Las fechas de publicación se muestran en la celda 3 y en el panel.
# * **Cuotas:** sin clave de API se usan líneas de referencia de nflverse; **verifica siempre la cuota real** en tu casa.
#   Las cuotas cambian y difieren entre casas.
# * **Backtesting:** usa cuotas de **cierre**, las más difíciles de batir. No hay cuotas históricas de apertura gratuitas.
#   Los resultados simulados no son ganancias reales y el pasado no garantiza el futuro.
# * **Modelo:** no conoce noticias de última hora, cambios de entrenador o motivación. El valor del QB suplente sin
#   historial es un supuesto (`QB_PRIOR`), no un dato.
# * **Disciplina:** el sistema puede (y suele) decir **NO APOSTAR**. Es la decisión correcta cuando el mercado ya refleja
#   la información disponible.
#
# 🔁 **Uso semanal recomendado:** ejecuta el cuaderno el martes/miércoles (nuevas líneas), y otra vez el sábado o el
# domingo por la mañana (lesiones finales, QB confirmados, clima). Revisa la celda 14 cada semana.
