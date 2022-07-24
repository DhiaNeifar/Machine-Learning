def n_decimal_places(X, n) -> float:
    d = pow(10, n)
    return int(X * d) / d
