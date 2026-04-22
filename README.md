# Proyecto final: detección de fraude transaccional con aprendizaje automático

## Descripción

Este repositorio contiene un proyecto de ciencia de datos aplicado al dataset IEEE-CIS Fraud Detection. El trabajo desarrolla un flujo reproducible para estudiar la detección de fraude transaccional mediante auditoría de datos, análisis exploratorio, preprocesamiento, modelado supervisado, explicabilidad, calibración, predicción conformal y análisis de errores.

El entregable principal es el notebook `notebooks/ieee_cis_fraud_detection_master.ipynb`, acompañado por documentación formal, figuras exportadas y archivos de apoyo para facilitar su revisión en GitHub.

## Objetivo general

Construir y documentar un flujo metodológico reproducible para detectar fraude transaccional con aprendizaje automático, evaluando un baseline interpretable y un modelo principal LightGBM, e incorporando lectura crítica mediante explicabilidad, calibración, análisis de umbral, incertidumbre conformal y análisis de errores por segmentos.

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
|   |   |-- documentacion_detallada.md
|   |   `-- resumen_ejecutivo.md
|   `-- docx/
|       |-- documentacion_detallada.docx
|       `-- resumen_ejecutivo.docx
|-- notebooks/
|   `-- ieee_cis_fraud_detection_master.ipynb
|-- reports/
|   `-- figures/
`-- src/
    `-- utils.py
```

### Archivos esperados en `data/raw/`

Los datos originales no se incluyen en el repositorio. Para ejecutar el notebook, deben colocarse manualmente estos archivos en `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

### Figuras exportadas

La carpeta `reports/figures/` contiene las figuras clave exportadas desde el notebook. En el árbol principal se muestra como carpeta para mantener la estructura limpia, sin enumerar cada archivo `.png`.

## Notebook principal

El notebook `notebooks/ieee_cis_fraud_detection_master.ipynb` contiene el flujo analítico completo del proyecto:

- configuración de rutas y carga inicial de datos;
- auditoría del dataset unido;
- análisis exploratorio profundo;
- diseño metodológico con partición temporal;
- preprocesamiento reproducible con `ColumnTransformer`;
- baseline interpretable con `DummyClassifier` y regresión logística;
- modelo principal avanzado con LightGBM;
- explicabilidad con SHAP;
- calibración isotónica y análisis de umbral;
- predicción conformal con MAPIE;
- análisis de errores y robustez;
- conclusiones finales.

## Documentación

La carpeta `documentacion/` separa los documentos por formato:

- `documentacion/md/`: versiones Markdown para GitHub y revisión en texto plano.
- `documentacion/docx/`: versiones Word para lectura, entrega o edición en procesadores de texto.

La documentación pública principal incluye:

- `documentacion_detallada`: informe técnico y metodológico completo del proyecto.
- `resumen_ejecutivo`: síntesis breve para lectura rápida.

## Ejecución local

1. Coloque `train_transaction.csv` y `train_identity.csv` en `data/raw/`.
2. Instale las dependencias declaradas en `requirements.txt`.
3. Ejecute el notebook principal desde la raíz del proyecto.

```bash
pip install -r requirements.txt
```

## Ejecución en Google Colab

1. Suba o clone este repositorio en Colab.
2. Coloque `train_transaction.csv` y `train_identity.csv` dentro de `data/raw/`.
3. Si faltan dependencias en el entorno, ejecute:

```python
%pip install -r requirements.txt
```

El notebook incluye una celda inicial de orientación para detectar ejecución local o en Google Colab y recordar la ubicación esperada de los archivos CSV.

## Figuras del proyecto

La carpeta `reports/figures/` contiene gráficos representativos exportados desde el análisis del proyecto. Estos archivos se usan en la documentación y permiten mantener referencias visuales estables fuera del notebook.

## Regla de mantenimiento

Toda modificación futura del notebook debe reflejarse también, cuando corresponda, en la documentación, el `README.md`, las versiones `.docx` públicas y las figuras exportadas. Esta regla mantiene consistencia entre el análisis reproducible, la documentación formal y los materiales asociados al proyecto.
