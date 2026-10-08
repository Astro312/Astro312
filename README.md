# 🏈 Asistente de análisis de apuestas NFL (Google Colab)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astro312/astro312/blob/claude/nfl-betting-prediction-model-86gxt8/NFL_Apuestas_2026.ipynb)

Cuaderno de Google Colab que descarga datos reales de la NFL, estima probabilidades con varios modelos validados en el
tiempo, las compara con las cuotas y muestra en un panel sencillo **solo** las oportunidades con respaldo estadístico.
Si no hay ninguna, lo dice: **NO APOSTAR**.

> ⚠️ Herramienta de análisis, no una garantía. Todas las cantidades son hipotéticas. No apuestes dinero que necesites.

## Cómo usarlo

1. Pulsa el botón **Abrir en Colab** (o sube `NFL_Apuestas_2026.ipynb` a Colab).
2. Menú **Entorno de ejecución → Ejecutar todas**. La primera vez tarda unos 5–10 minutos (descarga ~10 temporadas).
3. Baja hasta la **celda 12 (Panel)** y usa las listas desplegables. No hace falta escribir código.

Opcional: cuotas en vivo con una clave gratuita de [The Odds API](https://the-odds-api.com), guardada en
**Secretos de Colab (icono 🔑)** con el nombre `ODDS_API_KEY`. O sube tus cuotas en CSV (`USAR_CSV_MANUAL = True`).

## Estructura del cuaderno

| Celda | Contenido |
|---|---|
| 1 | Instalación de dependencias |
| 2 | Configuración, claves API (Secretos de Colab) y reglas de prudencia |
| 3 | Descarga de datos con caché y registro de fuente y fecha |
| 4 | Validación y limpieza (semáforo de calidad) |
| 5 | Ingeniería de variables pre-partido (sin fuga de información) |
| 6 | Entrenamiento: Elo, regresión logística, Random Forest, HistGradientBoosting, XGBoost, Ridge (margen y total) |
| 7 | Validación temporal walk-forward por temporada (accuracy, Brier, log loss, calibración) frente al mercado |
| 8 | Backtesting con cuotas de cierre reales: mezcla modelo/mercado y umbrales elegidos en 2019-21, comprobados en 2022+ |
| 9 | Predicción de la temporada en curso con control de versiones del modelo |
| 10 | Cuotas: CSV manual → The Odds API → líneas de referencia de nflverse |
| 11 | Valor esperado, cuota justa, ventaja, incertidumbre, confianza y decisión |
| 12 | **Panel**: resumen semanal, tabla con colores, explicaciones, glosario y análisis individual |
| 13 | Simulador de riesgo Monte Carlo (monto fijo, % fijo o Kelly fraccional con tope; sin martingala) |
| 14 | Registro de predicciones, liquidación, CLV, calibración semanal y exportación CSV |

## Fuentes de datos

| Fuente | Qué aporta | Clave |
|---|---|---|
| [nflverse](https://github.com/nflverse/nflverse-data) `schedules/games.csv` | Calendario, resultados, descanso, QB titular, clima, estadio, moneyline/spread/total con precios (de cierre en partidos jugados) | No |
| nflverse `pbp/play_by_play_AAAA.parquet` | EPA, success rate, 3ª oportunidad, zona roja, pérdidas, yardas | No |
| nflverse `injuries/injuries_AAAA.csv` | Reporte oficial de lesiones | No |
| [The Odds API](https://the-odds-api.com) v4 | Cuotas en vivo de varias casas (opcional) | Sí (gratis) |
| CSV manual | Tus propias cuotas (opcional) | No |

No se hace *scraping* de casas de apuestas ni se rellenan datos faltantes con cifras inventadas.

## Resultados de la validación (ejecución del 08/10/2026)

Los resultados son los de la ejecución con datos reales hasta la semana 4 de 2026; cambiarán al actualizar los datos.

* Validación walk-forward 2019–2026 (2 018 partidos): el mejor modelo (Ensamble) alcanza **65.1 % de aciertos,
  log loss 0.628**; las cuotas de cierre del mercado, **66.3 % y 0.610**. **El mercado de cierre es más preciso que
  cualquier modelo.**
* Por eso la mezcla óptima elegida en calibración da al modelo un peso de 0 % (moneyline) y 10 % (spread y total), y
  en el backtesting **ningún mercado queda habilitado**: el sistema recomienda **NO APOSTAR** hasta que haya
  evidencia de ventaja. Es el comportamiento disciplinado que se pidió, no un fallo.
* El backtest usa cuotas de **cierre** (las más eficientes). No hay cuotas históricas de apertura gratuitas, así que
  el CLV histórico no puede medirse; el registro de la celda 14 sí lo mide en vivo.

## Desarrollo

El cuaderno se genera desde `fuente/nfl_apuestas.py` (celdas en formato `# %%`):

```bash
pip install nbformat
python herramientas/construir_notebook.py
```
