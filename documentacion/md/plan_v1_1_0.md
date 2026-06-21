# Plan técnico para v1.1.0

## 1. Objetivo de v1.1.0

Construir una versión ampliada de `ieee-cis-fraud-detection` que integre el dataset completo, compare más modelos, permita predicciones finales sobre el test oficial y fortalezca la documentación técnica. Esta versión podrá modificar notebook, código, métricas, resultados, documentación y figuras cuando sea necesario para mejorar el proyecto.

## 2. Alcance ampliado

- Validar formalmente los cinco CSV: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` y `sample_submission.csv`.
- Usar los datos etiquetados de train para entrenamiento, validación y prueba interna.
- Mantener el test oficial separado de las métricas porque no contiene etiquetas reales.
- Integrar `test_transaction.csv` y `test_identity.csv` para generar predicciones finales.
- Usar `sample_submission.csv` como estructura de salida.
- Normalizar nombres en el test oficial: `train_identity` usa columnas tipo `id_01`, mientras `test_identity` puede usar `id-01`; debe contemplarse conversión de `id-` a `id_`.
- Comparar `DummyClassifier`, regresión logística, Random Forest, LightGBM, TabPFN y un modelo final entrenado con más datos etiquetados.
- Agregar una sección o documento explicativo sobre modelos, función, lectura de resultados y límites.
- Actualizar documentación, figuras y resultados cuando cambie el flujo.

## 3. Modelos y comparación

- `DummyClassifier`: referencia mínima basada en prevalencia. Aporta un piso de comparación; no aprende patrones. Se comparará contra todos los modelos mediante métricas de clase desbalanceada.
- Regresión logística: baseline interpretable y lineal. Aporta lectura de señales generales; puede limitarse ante interacciones y no linealidad. Se comparará contra modelos de árboles y TabPFN.
- Random Forest: modelo de ensamble robusto para relaciones no lineales. Aporta contraste frente a LightGBM; puede ser costoso y menos competitivo en alta dimensionalidad.
- LightGBM: modelo tabular principal por eficiencia y capacidad de capturar interacciones. Aporta desempeño fuerte; requiere control de sobreajuste e interpretación cuidadosa.
- TabPFN: modelo moderno para tabular. Aporta comparación adicional; puede tener restricciones de tamaño, memoria o compatibilidad.
- Modelo final con más datos etiquetados: se entrenará después de seleccionar el mejor enfoque. Aporta una versión final para inferencia sobre test oficial; no debe usarse para redefinir métricas internas de manera retrospectiva.

La comparación debe priorizar PR-AUC, ROC-AUC, precision, recall, F1 y matriz de confusión. El análisis debe explicar diferencias entre modelos, no solo mostrar tablas.

## 4. Orden recomendado de implementación

1. Ejecutar `python scripts/validate_dataset.py`.
2. Normalizar columnas `id-` a `id_` para el test oficial sin alterar los archivos originales.
3. Revisar notebook, rutas, dependencias y documentación antes de ampliar el flujo.
4. Integrar los cinco CSV en el diseño de trabajo.
5. Mantener evaluación interna solo con datos etiquetados de train.
6. Añadir Random Forest y TabPFN como comparación controlada.
7. Seleccionar el mejor enfoque con métricas internas.
8. Entrenar el modelo final con más datos etiquetados cuando el criterio de selección esté definido.
9. Generar predicciones finales sobre test oficial con formato de `sample_submission.csv`.
10. Actualizar documentación, figuras, resultados y checklist del release.

## 5. Riesgos técnicos

- Alto consumo de memoria por tamaño del dataset y modelos adicionales.
- Incompatibilidad de nombres `id-` frente a `id_` en identity de test.
- Uso indebido del test oficial para calcular métricas.
- Comparaciones no equivalentes si los modelos usan particiones o preprocesamientos distintos.
- TabPFN puede requerir muestreo, reducción de variables o ajustes por limitaciones operativas.
- Predicciones finales pueden versionarse por error si no se respeta `.gitignore`.

## 6. Pruebas requeridas

- Validar dataset completo antes de ejecutar el flujo.
- Confirmar que CSV y predicciones generadas no estén preparados para commit.
- Validar notebook, rutas, dependencias y documentación antes del release.
- Verificar que las métricas se calculen solo con datos etiquetados.
- Confirmar que las predicciones finales tengan columnas y orden compatibles con `sample_submission.csv`.

## 7. Criterios de aceptación

- Los cinco CSV están validados y documentados.
- La normalización `id-` a `id_` está contemplada para test oficial.
- La evaluación interna usa solo train etiquetado.
- El test oficial se usa únicamente para predicciones finales.
- Los modelos están comparados con métricas adecuadas para clases desbalanceadas.
- La interpretación explica diferencias, fortalezas y límites de cada enfoque.
- El modelo final se entrena bajo un criterio explícito y genera salida compatible con `sample_submission.csv`.
- La documentación explicativa de modelos queda integrada.
- Notebook, código, documentación, dependencias y rutas pasan revisión previa.
- `git status --short` muestra solo cambios intencionales.

## 8. Checklist previo al release v1.1.0

- [ ] Ejecutar validación formal del dataset.
- [ ] Confirmar exclusión de CSV y predicciones generadas.
- [ ] Revisar normalización de columnas identity en test.
- [ ] Verificar particiones internas y separación del test oficial.
- [ ] Comparar Dummy, regresión logística, Random Forest, LightGBM y TabPFN.
- [ ] Definir y entrenar modelo final con más datos etiquetados.
- [ ] Generar predicciones finales si el flujo queda aprobado.
- [ ] Actualizar documentación explicativa y figuras necesarias.
- [ ] Revisar reproducibilidad mínima del proyecto.
- [ ] Preparar release `v1.1.0` sin marcarlo como final antes de la revisión.
