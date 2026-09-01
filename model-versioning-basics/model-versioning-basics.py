def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    return max(models, key=lambda model: (model["accuracy"], -model["latency"], model["timestamp"]))["name"]
