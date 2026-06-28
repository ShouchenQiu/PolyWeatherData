# Keep a fixed 48-hour contract with a dynamic trusted decoder horizon

**Status**: accepted

TFT training and production retain a fixed f01..f48 interface, but non-MAJOR
hours use the nearest prior MAJOR overlap and last-value persistence for the
final uncovered 1..5 hours. Those tail cells retain provenance through
`is_extrapolated` and `extrapolation_step`, carry zero loss weight, and are
excluded from trading calculations; production presents them by persisting the
last trusted TFT quantiles.

**Considered options**: A global 43-hour model was rejected because it discards
real f44..f48 data from MAJOR and shorter-offset cycles and would break the
48-hour model/covariance contract. Full-weight training on synthetic tail inputs
was rejected because low-quality covariates could distort shared parameters.

**Consequences**: Trusted horizon length varies from 43 to 48 hours by init
offset, evaluation metrics ignore zero-weight tail cells, and ADR-0012's linear
tail extrapolation is superseded.
