"""LunarRisk research package (P2.1: joint posterior, rules, risk integration).

Kept separate from ``scripts/train_baseline.py`` on purpose: the P1.8 baselines and their outputs
must stay reproducible byte for byte, so new probabilistic work lives here and only *reuses* the
architecture shape (a 98,464-parameter fixed-mask CNN front end).
"""
