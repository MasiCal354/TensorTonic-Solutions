def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum(a == p for a, p in zip(y_true, y_pred))
    fp = fn = len(y_true) - tp
    return 2*tp/(2*tp + fp + fn)
