# Only f01 is stored per init_time in ParquetStore; Cleaner owns all feature engineering

Two related decisions are recorded together because they share the same rationale.

**f01-only storage**: The full fxx range (f01..f48) is redundant in Parquet because only
the 1-hour-ahead value is used as the encoder feature for each init_time. Storing f01
only reduces storage by ~47x and keeps the Parquet schema flat.
Non-MAJOR init_times that only reach f18 are extended to f48 via HRRR fill before being
fed to the decoder; the fill result is never persisted.

**Cleaner owns feature engineering**: Wind component decomposition, log1p smoothing of
heavy-tail variables, Kelvin-to-Celsius conversion, and temp_error computation all live
exclusively in `data/storage/cleaner.py`. No other module (model, loop, pipeline) performs
any feature transform. This enforces the layer boundary: `model/` never touches raw units
or raw column names.

**Considered option**: Compute features inside `model/pipeline.py` (as in the original
`tft_step9` script). Rejected because it couples model code to raw data units and makes
it impossible to validate features independently of the model.
