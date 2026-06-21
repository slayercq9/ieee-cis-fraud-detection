# Validación del dataset completo

## Propósito

Este documento describe la validación formal del dataset completo requerido para la versión `v1.1.0` del proyecto `ieee-cis-fraud-detection`. La validación permite confirmar que los archivos locales existen, tienen dimensiones esperadas y mantienen relaciones básicas consistentes antes de ejecutar el flujo principal.

## Archivos requeridos

Los archivos deben colocarse manualmente en `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Estos CSV no se versionan en Git. Deben mantenerse como archivos locales porque provienen del dataset original y tienen un tamaño elevado.

## Qué valida el script

El script `scripts/validate_dataset.py` revisa:

- existencia de los cinco archivos requeridos;
- número esperado de filas y columnas;
- presencia de `TransactionID` en todos los archivos;
- presencia de `isFraud` en `train_transaction.csv`;
- ausencia de `isFraud` en `test_transaction.csv`;
- columnas `TransactionID` e `isFraud` en `sample_submission.csv`;
- contencion de `TransactionID` de identity dentro de transaction, tanto en train como en test;
- coincidencia entre `TransactionID` de `sample_submission.csv` y `test_transaction.csv`;
- cobertura de identity en train y test;
- diferencia de nombres entre columnas `id_01` y `id-01`, relevante para normalizar el test oficial.

El script no modifica archivos de datos ni genera archivos de salida.

## Ejecución

Desde la raíz del proyecto:

```bash
python scripts/validate_dataset.py
```

La ejecución termina con código de salida `0` si las validaciones críticas son correctas. Si falta un archivo, una dimensión no coincide o se detecta una inconsistencia crítica, el script imprime un error claro y termina con código distinto de `0`.

## Uso del test oficial

El test oficial no contiene etiquetas reales de fraude. Por esa razón, debe usarse para generar predicciones finales y no para calcular métricas de desempeño. Las métricas del proyecto deben mantenerse sobre los conjuntos internos definidos a partir de los datos de entrenamiento.
