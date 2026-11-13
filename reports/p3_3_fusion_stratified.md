# P3.3 fusion by latitude and learned block gating

2026-11-12. **`FUSION_STRATIFIED_COMPLETE`.** The fusion gain is not carried by one region: inside the three Val latitude bands the concatenated model beats Moonstone alone in every band on average (**+0.0152 equatorial, +0.0140 mid, +0.0179 polar**), though only the mid band is positive for all three seeds. Giving the two blocks learnable scalar weights instead of fixed 1.0 does **not** improve on plain concatenation (overall 0.2695 vs 0.2683), and the optimizer settles at roughly **1.62 for the Moonstone block and 0.75 for the NASA block**.

| Band (Val rows) | Moonstone | Fusion | Gated fusion | Fusion - Moonstone | Per-seed deltas |
|---|---:|---:|---:|---:|---|
| Equatorial 9-30 (355) | 0.190933 | 0.206086 | 0.205830 | +0.015152 | -0.0017 / +0.0135 / +0.0336 (2/3 positive) |
| Mid 30-60 (450) | 0.129616 | 0.143631 | 0.144312 | +0.014015 | +0.0138 / +0.0171 / +0.0111 (3/3 positive) |
| Polar 60-90 (350) | 0.114087 | 0.131967 | 0.131462 | +0.017879 | +0.0389 / +0.0220 / -0.0073 (2/3 positive) |
| Overall (1,155) | 0.235176 | 0.268314 | 0.269538 | +0.033138 | +0.0437 / +0.0368 / +0.0190 (3/3 positive) |

Cell values are means over seeds 42/43/44; band macro-F1 uses the full 49-class denominator, so bands are lower than the overall value and only the deltas should be compared across rows.

## What it says

- **The complementary signal is not a single-region artefact.** All three bands move in the same direction on average, with the mid-latitude band the most consistent. The equatorial and polar bands each have one seed that goes slightly negative, so the per-band claim is directional, not established.
- **Learning the block weights adds nothing.** The gated model lands within 0.001 of plain concatenation overall; the only thing it learns is that the 256-d Moonstone block deserves about twice the weight of the 768-d NASA block (~1.62 vs ~0.75, consistent across seeds). That is a useful diagnostic: the cross-domain block contributes but is not an equal partner.
- **Absolute band differences are small relative to the overall gain**, which is expected because each band has 350-450 Val rows and the 49-class macro-F1 over a subset is dominated by many sparse classes.

## Setup

Latitude comes only from the authenticated metadata export keyed by patch id; Val is northern hemisphere only (9° to 89°), so there is no southern-hemisphere band. Training reuses the frozen budget (20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, seeds 42/43/44, earliest-tie Val selection) on the same caches as the fusion matrix; the overall per-cell values reproduce the earlier matrix exactly, which validates the re-run. The gated model multiplies the Moonstone and NASA blocks by two learnable scalars initialised at 1.0 and otherwise shares the same head and budget. No Test ID is read or scored.

## Caveats

- Bands hold 350-450 Val rows and three seeds; the per-band numbers are exploratory and not a significance test.
- No regional, geological or latitudinal causal interpretation is supported; the bands are only a coarse robustness check.
- The Moonstone encoder's supervision on the same cohort (noted in the composition report) still inflates its side of every comparison here.

Evidence: [summary](evidence/p33_20261112/fusion_stratified_01/summary.json) with every per-seed band value and the learned weights; protocol [pre-registered](../protocols/p33_fusion_stratified_plan.json).
