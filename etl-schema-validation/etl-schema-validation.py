def validate_records(records: list, schema: list) -> list:
    """
    Returns a list of result dictionaries.
    """
    checks = {"int": lambda value: type(value) is int, "float": lambda value: type(value) in (int, float), "str": lambda value: type(value) is str}
    results = []
    for index, record in enumerate(records):
        errors = []
        for field in schema:
            column = field["column"]
            if column not in record:
                errors.append(f"{column}: missing")
                continue
            value = record[column]
            if value is None:
                if not field["nullable"]:
                    errors.append(f"{column}: null")
                continue
            if not checks[field["type"]](value):
                errors.append(f"{column}: expected {field['type']}, got {type(value).__name__}")
                continue
            if ("min" in field and value < field["min"]) or ("max" in field and value > field["max"]):
                errors.append(f"{column}: out of range")
        results.append({"record_index": index, "is_valid": not errors, "errors": errors})
    return results
