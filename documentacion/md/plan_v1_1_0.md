# Plan de trabajo para v1.1.0

## 1. Objetivo de v1.1.0

Preparar una versión más limpia, reproducible y completa del proyecto `ieee-cis-fraud-detection`, incorporando soporte documental y operativo para el dataset completo de IEEE-CIS Fraud Detection. La versión debe mantener intactos los resultados ya obtenidos y mejorar la claridad de uso, revisión y publicación del repositorio.

## 2. Alcance incluido en v1.1.0

- Limpieza del repositorio: verificar que no existan archivos temporales, salidas pesadas innecesarias ni datos agregados al índice de Git.
- README mejorado: actualizar instrucciones, estructura y datos requeridos para reflejar el uso de los cinco CSV.
- Documentación complementaria: mantener alineados `documentacion/md/` y `documentacion/docx/` con los cambios documentales relevantes.
- Validación del dataset completo: confirmar presencia, encabezados mínimos y consistencia básica de los cinco archivos en `data/raw/`.
- Uso de los cinco CSV: documentar y preparar el flujo para `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` y `sample_submission.csv`.
- Integración del test oficial: incorporar el conjunto oficial de prueba como datos para inferencia, sin usarlo para métricas ni para ajustar decisiones del modelo.
- Generación opcional de predicciones finales: permitir una salida tipo submission cuando el usuario decida ejecutarla.
- Revisión del notebook y del código: comprobar rutas, comentarios, dependencias, celdas clave y consistencia de nombres sin cambiar resultados previos.
- Pruebas mínimas de reproducibilidad: validar que el proyecto abre, detecta datos y ejecuta verificaciones livianas sin correr todo el flujo completo.
- Preparación del release `v1.1.0`: dejar checklist, cambios documentados y estado de Git listo para revisión.

## 3. Mejoras excluidas o dejadas para después

- Reentrenamiento completo de modelos.
- Cambio de métricas, gráficos o resultados ya reportados.
- Ajuste amplio de hiperparámetros.
- Nuevas técnicas de modelado.
- Cambios estructurales grandes en el notebook.
- Automatización completa de despliegue o integración continua.
- Publicación de datasets locales en el repositorio.

## 4. Orden recomendado de implementación

1. Confirmar estado base con `git status --short` y revisar que la rama sea `mejora-v1.1.0`.
2. Validar los cinco CSV mediante encabezados, tamaño, columnas clave y lectura de una fila.
3. Actualizar `README.md` y `data/README.md` para describir el dataset completo y el rol de cada archivo.
4. Revisar `.gitignore` para asegurar exclusión de CSV, ZIP, cachés y temporales.
5. Ajustar documentación complementaria para reflejar el flujo con archivos de prueba y salida opcional.
6. Revisar notebook sin ejecución completa: rutas, celdas de configuración, nombres de variables y notas de uso.
7. Diseñar la integración del test oficial como bloque de inferencia separado, sin afectar validación ni prueba interna.
8. Preparar una ruta opcional para generar predicciones finales con formato compatible con `sample_submission.csv`.
9. Ejecutar pruebas mínimas de reproducibilidad: formato del notebook, existencia de rutas, dependencias declaradas y validación de datos.
10. Verificar que solo queden cambios intencionales antes de cerrar la versión.

## 5. Riesgos técnicos

- Los archivos completos son grandes y pueden exigir memoria elevada durante ejecución.
- Dependencias como MAPIE, LightGBM o SHAP pueden cambiar comportamiento entre versiones.
- Usar el test oficial para decisiones de evaluación produciría una lectura metodológica inválida.
- Mezclar predicciones finales con métricas internas puede generar confusión en la interpretación.
- Si las figuras no se regeneran junto con cambios del notebook, podrían quedar desalineadas.
- La sincronización manual entre Markdown y DOCX puede dejar diferencias si no se revisa al cierre.

## 6. Criterios de aceptación

- `README.md` y `data/README.md` describen correctamente los cinco CSV.
- `.gitignore` mantiene excluidos los datos locales y archivos temporales.
- La validación liviana confirma que los cinco CSV existen y tienen columnas esperadas.
- El notebook conserva resultados, salidas y lógica previa.
- La integración del test oficial queda separada de métricas de validación y prueba interna.
- La generación de predicciones finales, si se incorpora, usa el formato de `sample_submission.csv`.
- La documentación pública queda sincronizada con el estado del proyecto.
- `git status --short` muestra únicamente cambios esperados para la versión.

## 7. Checklist previo al release v1.1.0

- [ ] Confirmar rama `mejora-v1.1.0`.
- [ ] Revisar que no haya CSV ni ZIP preparados para commit.
- [ ] Validar los cinco archivos de `data/raw/`.
- [ ] Revisar README y documentación de datos.
- [ ] Verificar dependencias declaradas en `requirements.txt`.
- [ ] Comprobar que las figuras referenciadas existen.
- [ ] Revisar notebook sin ejecutar entrenamiento completo.
- [ ] Confirmar separación entre evaluación interna e inferencia sobre test oficial.
- [ ] Probar generación opcional de predicciones finales si se implementa.
- [ ] Revisar sincronía entre Markdown y DOCX.
- [ ] Confirmar estado final con `git status --short`.
- [ ] Preparar tag `v1.1.0` solo después de la revisión final.
