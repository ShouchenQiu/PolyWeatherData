# McSampler uses empirical covariance, not AR(1)

Weather forecast errors across a 48-hour window are correlated in ways an AR(1) model
cannot capture (e.g. a morning bias persists through the afternoon). We estimate a
48x48 empirical correlation matrix from validation errors standardized by the TFT's
per-cycle uncertainty. Sampling uses
`diag(calibrated_sigma) × correlation × diag(calibrated_sigma)`, preserving both
empirical temporal dependence and the TFT's dynamic interval width.
The `mc_rho` and `mc_bias_corr_c` settings fields are removed; `n_samp` defaults to 10000.

**Considered option**: Keep AR(1) and tune `mc_rho` per station.
Rejected because rho has no stable physical interpretation across stations and seasons,
and the empirical matrix is estimated automatically from prediction residuals.

For contract-v2 models, sigma scales, correlation, and daily-high max bias are stored
in a checkpoint-hash-bound calibration bundle fitted from the baseline TFT's validation
errors. The TFT feature contract is unchanged from v1; v2 denotes mandatory calibrated
MC evidence, not a different TFT input set. Missing or mismatched calibration is a load
error; only legacy v1 models may use the old covariance/diagonal fallback.

Daily-high max bias and calibration evidence use only complete station-local calendar
days as defined by ADR-0010. Contiguous-fold validation persists one out-of-fold row per
`(init_time, target_date)` cycle-day; a global 48-hour maximum is not valid evidence.
