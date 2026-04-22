# Resumen ejecutivo

**Título del proyecto:** Proyecto final: detección de fraude transaccional con aprendizaje automático  
**Autor:** Fernando Corrales Quirós  
**Curso:** Fundamentos de Analítica de Datos  
**Institución:** Universidad CENFOTEC  
**Periodo:** Primer cuatrimestre, 2026  
**Dataset:** IEEE-CIS Fraud Detection

## Propósito del proyecto

Este proyecto desarrolla un flujo reproducible de ciencia de datos para abordar la detección de fraude transaccional con aprendizaje automático. El trabajo se basa en el dataset IEEE-CIS Fraud Detection y combina auditoría de datos, análisis exploratorio, preprocesamiento, modelado supervisado, explicabilidad, calibración, incertidumbre y análisis de errores.

El objetivo no fue solamente obtener un modelo con buen desempeño, sino construir una lectura metodológica sólida del problema: entender la estructura del dataset, controlar riesgos de fuga de información, comparar modelos de forma justa y reconocer límites reales del sistema predictivo.

## Dataset y problema abordado

El dataset contiene información transaccional y variables complementarias de identidad. Las tablas se integran mediante `TransactionID`, conservando la tabla de transacciones como base. La variable objetivo es `isFraud`, que identifica si una transacción fue clasificada como fraudulenta.

El problema presenta cuatro retos principales:

- fuerte desbalance entre fraude y no fraude;
- alta dimensionalidad;
- presencia considerable de valores faltantes;
- comportamiento temporal relativo capturado por `TransactionDT`.

Estos factores hacen que la exactitud no sea una métrica suficiente y que sea necesario evaluar el modelo con métricas sensibles al desempeño sobre la clase fraude.

## Metodología resumida

El flujo metodológico utilizó una partición temporal en entrenamiento, validación y prueba, evitando un split aleatorio puro. Esta decisión busca una evaluación más realista, ya que los patrones de fraude pueden variar con el tiempo.

El preprocesamiento se implementó con un pipeline reproducible que incluye depuración inicial de variables, imputación numérica y categórica, indicadores de ausencia y codificación one-hot robusta. Todo el pipeline se ajustó únicamente sobre entrenamiento para reducir riesgos de fuga de información.

## Comparación de modelos

Se evaluaron tres referencias:

- `DummyClassifier`, como línea mínima basada en prevalencia;
- regresión logística regularizada, como baseline interpretable;
- LightGBM, como modelo principal avanzado para datos tabulares.

LightGBM superó claramente a la regresión logística. En validación, la PR-AUC aumentó de `0.3758` a `0.5960`; en prueba, aumentó de `0.2181` a `0.5440`. La ROC-AUC también mejoró, de `0.8326` a `0.9235` en validación y de `0.8249` a `0.9049` en prueba.

Este resultado indica que LightGBM captura relaciones no lineales e interacciones que el baseline lineal no representa con la misma flexibilidad.

## Explicabilidad, calibración e incertidumbre

SHAP permitió identificar variables transaccionales, temporales y agregados de comportamiento como señales relevantes para LightGBM. La interpretación se mantuvo como una lectura predictiva, no causal.

La calibración isotónica se usó para evaluar confiabilidad probabilística. Las probabilidades originales del modelo ya mostraban una alineación razonable en prueba, y la calibración no produjo una mejora clara en los indicadores resumidos.

La predicción conformal añadió una capa moderna de incertidumbre. La cobertura global quedó cerca de los niveles nominales de `90%` y `95%`, aunque la cobertura de la clase fraude fue menor que la de no fraude. Bajo la configuración usada, la incertidumbre apareció principalmente como abstención `{}`.

## Análisis de errores y robustez

El análisis de errores mostró que la principal debilidad del modelo está en los falsos negativos. En prueba, LightGBM obtuvo 1,108 verdaderos positivos, 84,748 verdaderos negativos, 267 falsos positivos y 1,957 falsos negativos.

La revisión por segmentos evidenció que el desempeño no es uniforme entre rangos de monto, bloques temporales, `ProductCD` y `DeviceType`. Esta heterogeneidad es relevante porque un buen desempeño global no garantiza estabilidad en todos los perfiles operativos.

## Conclusión ejecutiva

LightGBM es el modelo más sólido dentro del alcance del proyecto y mejora sustancialmente al baseline interpretable. Sin embargo, la detección de fraude sigue siendo un problema difícil por el desbalance, la incompletitud, los cambios temporales y la persistencia de falsos negativos.

El principal aporte del proyecto es integrar desempeño predictivo con una evaluación crítica: explicabilidad, calibración, incertidumbre y análisis de errores. Esta combinación produce una solución técnicamente defendible y evita una interpretación excesivamente optimista del modelo.

## Nota de mantenimiento

Si el notebook principal cambia, también deben revisarse este resumen, la documentación detallada, el `README.md`, las versiones `.docx` públicas y las figuras asociadas cuando corresponda.
