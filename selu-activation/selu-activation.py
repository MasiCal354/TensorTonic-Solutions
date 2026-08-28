import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    lamb = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return [round(lamb * i, 4) if i > 0 else round(lamb * alpha * (math.exp(i) - 1), 4) for i in x]