# Detección de fraude transaccional con aprendizaje automático

## Descripción

Este repositorio contiene un flujo reproducible de ciencia de datos para el dataset IEEE-CIS Fraud Detection. El proyecto aborda detección de fraude transaccional mediante auditoría de datos, análisis exploratorio, partición temporal, preprocesamiento, modelos supervisados, explicabilidad, calibración, predicción conformal y análisis de errores.

El entregable principal es el notebook `notebooks/ieee_cis_fraud_detection_master.ipynb`, acompañado por documentación, figuras exportadas y una carpeta preparada para salidas de predicción.

## Objetivo general

Construir y documentar un flujo técnico para detectar fraude transaccional con aprendizaje automático, comparando un baseline interpretable con un modelo principal LightGBM e incorporando análisis de desempeño, explicabilidad, incertidumbre y errores por segmentos.

## Estructura del repositorio

```text
ieee-cis-fraud-detection/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- README.md
|-- documentacion/
|   |-- md/
|   |   |-- diagnostico_v1_1_0.md
|   |   |-- documentacion_detallada.md
|   |   |-- plan_v1_1_0.md
|   |   `-- resumen_ejecutivo.md
|   `-- docx/
|       |-- documentacion_detallada.docx
|       `-- resumen_ejecutivo.docx
|-- notebooks/
|   `-- ieee_cis_fraud_detection_master.ipynb
|-- reports/
|   |-- figures/
|   `-- submissions/
|       `-- .gitkeep
`-- src/
    `-- utils.py
```

## Datos requeridos

Los archivos CSV originales no se incluyen en Git. Para ejecutar el notebook, deben colocarse manualmente en `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Los archivos `train_transaction.csv` y `train_identity.csv` se usan para entrenamiento, validación y evaluación interna. Los archivos oficiales de test se reservan para generar predicciones finales y no deben utilizarse para calcular métricas de desempeño.

## Notebook principal

El notebook `notebooks/ieee_cis_fraud_detection_master.ipynb` contiene el flujo de trabajo completo:

- configuración de rutas y carga inicial de datos;
- auditoría del dataset unido;
- análisis exploratorio profundo;
- partición temporal y preprocesamiento reproducible;
- baseline con `DummyClassifier` y regresión logística;
- modelo principal con LightGBM;
- explicabilidad con SHAP;
- calibración isotónica y análisis de umbral;
- predicción conformal con MAPIE;
- análisis de errores y robustez;
- conclusiones finales.

## Documentación

La carpeta `documentacion/` separa los documentos por formato:

- `documentacion/md/`: documentos Markdown para revisión en GitHub.
- `documentacion/docx/`: versiones Word de los documentos principales.

La documentación principal incluye:

- `documentacion_detallada`: informe técnico y metodológico del proyecto.
- `resumen_ejecutivo`: síntesis breve para lectura rápida.
- `diagnostico_v1_1_0` y `plan_v1_1_0`: documentos de preparación de la versión `v1.1.0`.

## Figuras y predicciones

La carpeta `reports/figures/` contiene figuras exportadas desde el notebook y usadas por la documentación.

La carpeta `reports/submissions/` queda preparada para archivos de predicción final. Los CSV generados en esa carpeta están excluidos por `.gitignore`; solo se versiona `.gitkeep` para conservar la estructura.

## Ejecución local

1. Coloque los cinco CSV requeridos en `data/raw/`.
2. Instale las dependencias declaradas en `requirements.txt`.
3. Ejecute el notebook principal desde la raíz del proyecto.

```bash
pip install -r requirements.txt
```

## Ejecución en Google Colab

1. Suba o clone este repositorio en Colab.
2. Coloque los cinco CSV requeridos dentro de `data/raw/`.
3. Si faltan dependencias en el entorno, ejecute:

```python
%pip install -r requirements.txt
```

El notebook incluye una celda inicial de orientación para detectar ejecución local o en Google Colab y recordar la ubicación esperada de los archivos CSV.

## Versiones

- `v1.0.0`: versión inicial estable.
- `v1.1.0`: versión en preparación, orientada a limpieza del repositorio, soporte documental para el dataset completo y preparación de salidas de predicción.

## Regla de mantenimiento

Toda modificación futura del notebook debe reflejarse también, cuando corresponda, en la documentación, el `README.md`, las versiones `.docx` públicas y las figuras exportadas. Esta regla mantiene consistencia entre el análisis reproducible, la documentación y los materiales asociados al proyecto.
