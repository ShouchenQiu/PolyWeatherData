# Encoder window is 72 hours, not 24

The original `tft_step7_dataloader.py` used a -23..0 window despite `tft_step6_build_72h.py`
preparing -72..0 data ¡ª a latent bug. Because the model target is max daily temperature,
72 hours of context is needed to capture diurnal cycles and multi-day trends; 24 hours is
insufficient and causes the encoder to silently train on a misaligned window.
`MAX_ENCODER_LENGTH = 72` in `model/tft.py`, and all DataLoader and Cleaner construction
must use the same constant.

**Considered option**: Keep 24 h and re-train from scratch on a shorter window.
Rejected because the existing Parquet dataset already stores 72 h of f01 history and
there is no benefit to discarding that signal.
