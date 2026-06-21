# Historial de cambios

Este proyecto utiliza versionado semántico para organizar cambios mayores, mejoras compatibles y correcciones de mantenimiento.

## [v1.1.0] - En preparación

### Agregado

- Preparación de `reports/submissions/` para almacenar salidas de predicción generadas localmente.
- Documentos de diagnóstico y planificación para orientar la construcción de `v1.1.0`.
- Referencia al dataset completo con archivos de entrenamiento, prueba oficial y plantilla de envío.

### Cambiado

- README actualizado con una estructura más clara, instrucciones de ejecución y explicación del uso de archivos `train` y `test`.
- Descripción de datos ajustada para considerar los cinco CSV esperados en `data/raw/`.
- Reglas de mantenimiento reforzadas para conservar consistencia entre notebook, documentación y figuras.

### Corregido

- Limpieza de referencias obsoletas a materiales de presentación eliminados.
- Revisión de exclusiones para evitar versionar datos locales, archivos comprimidos y salidas generadas.

### Pendiente

- Integrar el test oficial como bloque de inferencia sin usarlo para calcular métricas.
- Validar generación opcional de predicciones finales con formato compatible con `sample_submission.csv`.
- Revisar que las figuras exportadas sigan alineadas con la versión final del notebook.
- Confirmar que el estado de Git incluya solo cambios intencionales antes del release.

## [v1.0.0] - Versión Inicial Estable

- Notebook principal con flujo completo de análisis para IEEE-CIS Fraud Detection.
- Auditoría inicial del dataset unido y análisis exploratorio del comportamiento de fraude.
- Preprocesamiento reproducible con partición temporal, tratamiento de faltantes y codificación categórica.
- Modelos de referencia con `DummyClassifier` y regresión logística.
- Modelo principal con LightGBM y comparación frente al baseline.
- Explicabilidad con SHAP para interpretación global y local del modelo.
- Calibración de probabilidades y análisis de umbral de decisión.
- Predicción conformal con MAPIE como componente de incertidumbre.
- Análisis de errores y robustez por segmentos relevantes.
- Documentación principal en Markdown y DOCX, junto con figuras exportadas en `reports/figures/`.
