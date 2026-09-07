def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    return 0.0 if n_items == 0 else len(set(item for items in recommendations for item in items))/n_items
