# ML1_ExamenAplicado_Mesina_Sebastián

Examen aplicado — Machine Learning I. Análisis de regresión sobre el dataset California Housing: exploración de datos, preprocesamiento sin data leakage, PCA, K-Means, y comparación de dos modelos supervisados.

## Descripción del dataset

- **Nombre:** California Housing
- **Fuente:** `sklearn.datasets.fetch_california_housing` (StatLib, U.S. Census 1990)
- **URL:** https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Filas:** 20,640 | **Columnas:** 9 (8 predictoras + 1 objetivo)
- **Variable objetivo:** `MedHouseVal` (valor mediano de vivienda por distrito, en cientos de miles de USD)
- **Tipo de tarea:** Regresión

## Metodología resumida

1. Carga y exploración inicial (`shape`, `dtypes`, `info`, `describe`).
2. Análisis de valores faltantes (0% — no requiere imputación).
3. Detección y tratamiento de outliers (método IQR, capping/winsorizing) sobre `MedInc`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`.
4. Análisis de la distribución del objetivo (skewness) y visualizaciones exploratorias (heatmap de correlación, scatter, violin plot).
5. División train/test (80/20, `random_state=42`) **antes** de cualquier transformación.
6. `ColumnTransformer` con `SimpleImputer` (mediana) + `StandardScaler`, ajustado solo sobre `X_train`.
7. PCA (varianza explicada 80–90%) y K-Means (K óptimo por silhouette score).
8. Modelado supervisado: `RidgeCV` (penalizado) y `RandomForestRegressor` con `GridSearchCV` (basado en árboles).
9. Evaluación sobre test con RMSE, MAE, R² y MAPE; comparación train vs. test para detectar sobreajuste.
10. Interpretación de importancia de variables y análisis de las observaciones con mayor error.

## Resultados del mejor modelo

| Modelo | RMSE | MAE | R² | MAPE | Tiempo entrenamiento (s) |
|---|---|---|---|---|---|
| **Random Forest** | 0.5054 | 0.4044 | 0.7263 | 37.36% | 184.72 |
| Ridge (RidgeCV) | 0.5072 | 0.4063 | 0.7244 | 36.78% | 0.08 |

**Modelo seleccionado:** Random Forest (menor RMSE), con la advertencia de que Ridge es casi igual de competitivo y mucho más rápido/interpretable — ver justificación completa en el notebook.

**Variables más importantes (Random Forest):** `MedInc` (~89% de la importancia total), `HouseAge`, `AveOccup`, `Latitude`, `AveBedrms` — ver detalle e interpretación de negocio en el notebook.

## Estructura del repositorio

```
├── ML1_ExamenAplicado.ipynb   # Notebook ejecutado con todo el análisis
├── README.md
├── requirements.txt
├── model_comparison.csv       # Tabla comparativa de métricas
└── figures/                   # Gráficos exportados (dpi=150)
```

## Cómo reproducir el análisis

```bash
pip install -r requirements.txt
jupyter notebook ML1_ExamenAplicado.ipynb
```

## Declaración de uso de IA generativa

Este trabajo fue desarrollado con asistencia de **Claude (Anthropic)** para la estructuración del pipeline de análisis (EDA, preprocesamiento, PCA/K-Means, modelado y evaluación) a partir de datos reales del dataset California Housing. La IA se utilizó como apoyo en la organización del código, la redacción de las justificaciones metodológicas en Markdown y la generación de material de apoyo para la grabación del video (guion y presentación). El análisis, la ejecución del código y la interpretación de resultados fueron revisados por el autor.

## Video

https://youtu.be/7oXZbq9Y2Pc
