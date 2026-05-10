# McSampler uses empirical covariance, not AR(1)

Weather forecast errors across a 48-hour window are correlated in ways an AR(1) model
cannot capture (e.g. a morning bias persists through the afternoon). We replace the AR(1)
transition matrix with a 48x48 empirical covariance matrix estimated from historical
residuals, giving McSampler realistic joint samples across all decoder hours.
The `mc_rho` and `mc_bias_corr_c` settings fields are removed; `n_samp` defaults to 10000.

**Considered option**: Keep AR(1) and tune `mc_rho` per station.
Rejected because rho has no stable physical interpretation across stations and seasons,
and the empirical matrix is estimated automatically from the same training residuals
already produced by the TFT step.
