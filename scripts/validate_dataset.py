"""Validación formal del dataset completo IEEE-CIS Fraud Detection.

El script revisa archivos locales en data/raw/ sin modificar datos ni generar
salidas. Termina con código 0 si las validaciones críticas son correctas.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = {
    "train_transaction.csv": {
        "rows": 590_540,
        "cols": 394,
        "required": {"TransactionID", "isFraud", "TransactionDT"},
        "forbidden": set(),
    },
    "train_identity.csv": {
        "rows": 144_233,
        "cols": 41,
        "required": {"TransactionID"},
        "forbidden": set(),
    },
    "test_transaction.csv": {
        "rows": 506_691,
        "cols": 393,
        "required": {"TransactionID", "TransactionDT"},
        "forbidden": {"isFraud"},
    },
    "test_identity.csv": {
        "rows": 141_907,
        "cols": 41,
        "required": {"TransactionID"},
        "forbidden": set(),
    },
    "sample_submission.csv": {
        "rows": 506_691,
        "cols": 2,
        "required": {"TransactionID", "isFraud"},
        "forbidden": set(),
    },
}


def scan_csv(path: Path) -> dict:
    """Lee encabezado, cuenta filas y conserva TransactionID para cruces."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        id_index = header.index("TransactionID") if "TransactionID" in header else None

        row_count = 0
        malformed_rows = 0
        transaction_ids: list[str] = []

        for row in reader:
            row_count += 1
            if len(row) != len(header):
                malformed_rows += 1
            if id_index is not None and len(row) > id_index:
                transaction_ids.append(row[id_index])

    return {
        "header": header,
        "rows": row_count,
        "cols": len(header),
        "malformed_rows": malformed_rows,
        "transaction_ids": transaction_ids,
    }


def format_bool(value: bool) -> str:
    """Entrega una etiqueta compacta para el resumen de consola."""
    return "OK" if value else "ERROR"


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    results: dict[str, dict] = {}

    print("Validación del dataset IEEE-CIS Fraud Detection")
    print(f"Raíz del proyecto: {PROJECT_ROOT}")
    print(f"Directorio de datos: {DATA_DIR}\n")

    # Primero se valida la existencia para evitar errores menos claros.
    for filename in EXPECTED_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            errors.append(f"Falta el archivo requerido: {path}")

    if errors:
        print("Errores críticos:")
        for error in errors:
            print(f"- {error}")
        return 1

    # Escaneo liviano: no carga el dataset completo como dataframe.
    for filename, expected in EXPECTED_FILES.items():
        path = DATA_DIR / filename
        result = scan_csv(path)
        results[filename] = result

        if result["rows"] != expected["rows"]:
            errors.append(
                f"{filename}: filas esperadas={expected['rows']}, "
                f"filas observadas={result['rows']}"
            )
        if result["cols"] != expected["cols"]:
            errors.append(
                f"{filename}: columnas esperadas={expected['cols']}, "
                f"columnas observadas={result['cols']}"
            )

        header = set(result["header"])
        missing_required = expected["required"] - header
        forbidden_present = expected["forbidden"] & header

        if missing_required:
            errors.append(
                f"{filename}: faltan columnas clave: "
                f"{', '.join(sorted(missing_required))}"
            )
        if forbidden_present:
            errors.append(
                f"{filename}: contiene columnas no esperadas: "
                f"{', '.join(sorted(forbidden_present))}"
            )
        if result["malformed_rows"]:
            errors.append(
                f"{filename}: filas con cantidad irregular de columnas="
                f"{result['malformed_rows']}"
            )

    train_transaction_ids = set(results["train_transaction.csv"]["transaction_ids"])
    train_identity_ids = set(results["train_identity.csv"]["transaction_ids"])
    test_transaction_ids = set(results["test_transaction.csv"]["transaction_ids"])
    test_identity_ids = set(results["test_identity.csv"]["transaction_ids"])
    sample_ids = results["sample_submission.csv"]["transaction_ids"]
    sample_id_set = set(sample_ids)

    if not train_identity_ids.issubset(train_transaction_ids):
        errors.append("train_identity.csv contiene TransactionID fuera de train_transaction.csv")
    if not test_identity_ids.issubset(test_transaction_ids):
        errors.append("test_identity.csv contiene TransactionID fuera de test_transaction.csv")
    if sample_id_set != test_transaction_ids:
        errors.append("sample_submission.csv no coincide con TransactionID de test_transaction.csv")

    if sample_ids != results["test_transaction.csv"]["transaction_ids"]:
        warnings.append(
            "sample_submission.csv contiene los mismos TransactionID que test_transaction.csv, "
            "pero el orden no coincide."
        )

    train_identity_coverage = len(train_identity_ids) / len(train_transaction_ids)
    test_identity_coverage = len(test_identity_ids) / len(test_transaction_ids)

    train_identity_header = results["train_identity.csv"]["header"]
    test_identity_header = results["test_identity.csv"]["header"]
    train_has_underscore = any(col.startswith("id_") for col in train_identity_header)
    test_has_hyphen = any(col.startswith("id-") for col in test_identity_header)

    if train_has_underscore and test_has_hyphen:
        warnings.append(
            "test_identity.csv usa columnas tipo id-01 mientras train_identity.csv "
            "usa id_01; se requiere normalización de nombres para el test oficial."
        )

    print("Resumen por archivo:")
    print("Archivo | Filas | Columnas | Columnas clave")
    print("--- | ---: | ---: | ---")
    for filename, expected in EXPECTED_FILES.items():
        result = results[filename]
        header = set(result["header"])
        required_ok = expected["required"].issubset(header)
        forbidden_ok = not (expected["forbidden"] & header)
        keys_ok = required_ok and forbidden_ok
        print(
            f"{filename} | {result['rows']} | {result['cols']} | "
            f"{format_bool(keys_ok)}"
        )

    print("\nConsistencia de TransactionID:")
    print(f"- Cobertura de identity en train: {train_identity_coverage:.2%}")
    print(f"- Cobertura de identity en test: {test_identity_coverage:.2%}")
    print(f"- train_identity contenido en train_transaction: {format_bool(train_identity_ids.issubset(train_transaction_ids))}")
    print(f"- test_identity contenido en test_transaction: {format_bool(test_identity_ids.issubset(test_transaction_ids))}")
    print(f"- sample_submission coincide con test_transaction: {format_bool(sample_id_set == test_transaction_ids)}")

    if warnings:
        print("\nAdvertencias:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nErrores críticos:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nValidación completada correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
