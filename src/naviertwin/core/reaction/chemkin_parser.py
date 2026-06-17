"""CHEMKIN-style reaction parser — "A + B = C + D  k0 beta Ea".

Examples:
    >>> from naviertwin.core.reaction.chemkin_parser import parse_reaction
    >>> r = parse_reaction("H2 + O = OH + H  1.04e8 0.0 6.96")
    >>> r['reactants']
    ['H2', 'O']
"""

from __future__ import annotations

import re

_SPLIT = re.compile(r"\s*\+\s*")


def parse_reaction(line: str) -> dict:
    body, *params = line.split("=", 1)
    if not params:
        raise ValueError("expected '=' in reaction")
    reactants = list(map(str.strip, _SPLIT.split(body.strip())))
    rest = params[0].strip()
    # split products from rate constants by 2+ spaces or by detecting numeric tokens
    tokens = rest.split()
    # find first numeric token
    n_idx = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        try:
            float(t)
            n_idx = i
            break
        except ValueError:
            i += 1
            continue
    if n_idx is None:
        products_str = rest
        rates = (0.0, 0.0, 0.0)
    else:
        products_str = " ".join(tokens[:n_idx])
        rates_tokens = tokens[n_idx:n_idx + 3]
        rates = tuple(map(float, rates_tokens)) + (0.0,) * (3 - len(rates_tokens))
    products = list(filter(None, map(str.strip, _SPLIT.split(products_str.strip()))))
    A, beta, Ea = rates
    return {
        "reactants": reactants, "products": products,
        "A": float(A), "beta": float(beta), "Ea": float(Ea),
    }


__all__ = ["parse_reaction"]
