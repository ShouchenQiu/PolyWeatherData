"""Tests for model/tft.py -- TftModel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.tft import MAX_ENCODER_LENGTH, TftModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_max_encoder_length():
    assert MAX_ENCODER_LENGTH == 72


# ---------------------------------------------------------------------------
# TftModel.predict() -- un-residualise regression test
# ---------------------------------------------------------------------------

def _make_encoder_df(n: int = 72) -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    return pd.DataFrame({
        "valid_time": [base + pd.Timedelta(hours=i) for i in range(n)],
        "TMP_2m":     [5.0] * n,
        "time_idx":   list(range(n)),
    })


def _make_decoder_df(init_utc: pd.Timestamp, tmp2m_val: float = 10.0) -> pd.DataFrame:
    """48 decoder rows with a fixed TMP_2m for easy residual arithmetic."""
    return pd.DataFrame({
        "fxx":        list(range(1, 49)),
        "init_time":  [init_utc] * 48,
        "valid_time": [init_utc + pd.Timedelta(hours=f) for f in range(1, 49)],
        "TMP_2m":     [tmp2m_val] * 48,
        "time_idx":   list(range(72, 120)),
    })


class TestTftModelUnResidualize:
    """predict() must add TMP_2m back to the raw TFT residual output."""

    def _build_model(self) -> TftModel:
        """Return a TftModel with _ensure_loaded bypassed and _tft mocked."""
        model = TftModel.__new__(TftModel)
        model._checkpoint_path = None
        model._training_dataset_pkl = None
        model._tft = None
        model._training_ds = None
        return model

    def _mock_inner_tft(self, residual_row: list[float]):
        """Return a mock inner TFT whose predict() returns fixed residuals."""
        import types

        inner = types.SimpleNamespace()

        # preds_q is a list; preds_q[0] is the (48, 3) array
        residuals_arr = np.array(
            [residual_row for _ in range(48)], dtype=float
        )  # shape (48, 3)

        inner.predict = lambda loader, mode: [residuals_arr]
        return inner

    def test_residuals_plus_tmp2m(self):
        """Output quantiles = residual + TMP_2m for each fxx."""
        residual_row = [1.0, 2.0, 3.0]   # p10, p50, p90 residuals
        tmp2m_val    = 10.0               # TMP_2m in decoder

        model = self._build_model()

        # Patch _ensure_loaded and the TimeSeriesDataSet call path
        model._tft        = self._mock_inner_tft(residual_row)
        model._training_ds = object()  # non-None sentinel

        # Patch out the TimeSeriesDataSet / DataLoader calls so we bypass
        # pytorch-forecasting entirely -- only the numpy arithmetic matters.
        import model.tft as tft_module
        from unittest.mock import MagicMock, patch

        fake_loader = object()

        def fake_from_dataset(*args, **kwargs):
            ds = MagicMock()
            ds.to_dataloader.return_value = fake_loader
            return ds

        with patch.object(tft_module, "TimeSeriesDataSet",
                          create=True) as mock_tsd:
            mock_tsd.from_dataset.side_effect = fake_from_dataset

            enc = _make_encoder_df()
            dec = _make_decoder_df(
                pd.Timestamp("2025-01-01 12:00:00", tz="UTC"),
                tmp2m_val=tmp2m_val,
            )
            result = model.predict(enc, dec)

        combined = mock_tsd.from_dataset.call_args.args[1]
        assert combined["decoder_weight"].eq(1.0).all()
        # Every fxx should have absolute = residual + TMP_2m
        for fxx in range(1, 49):
            assert result[fxx]["p10"] == pytest.approx(residual_row[0] + tmp2m_val)
            assert result[fxx]["p50"] == pytest.approx(residual_row[1] + tmp2m_val)
            assert result[fxx]["p90"] == pytest.approx(residual_row[2] + tmp2m_val)

    def test_missing_tmp2m_raises_value_error(self):
        """When TMP_2m is absent from decoder, predict() must raise ValueError."""
        residual_row = [5.0, 6.0, 7.0]

        model = self._build_model()
        model._tft        = self._mock_inner_tft(residual_row)
        model._training_ds = object()

        enc = _make_encoder_df()
        dec = pd.DataFrame({
            "fxx":       list(range(1, 49)),
            "time_idx":  list(range(72, 120)),
            "valid_time": [
                pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
                + pd.Timedelta(hours=f)
                for f in range(1, 49)
            ],
        })
        with pytest.raises(ValueError, match="TMP_2m"):
            model.predict(enc, dec)


# ---------------------------------------------------------------------------
# V2V05 input validation
# ---------------------------------------------------------------------------

class TestTftModelValidation:
    """predict() must validate inputs before reaching pytorch code."""

    def _build_model(self) -> TftModel:
        model = TftModel.__new__(TftModel)
        model._checkpoint_path = None
        model._training_dataset_pkl = None
        model._tft = None
        model._training_ds = None
        return model

    def test_raises_if_encoder_missing_time_idx(self):
        model = self._build_model()
        enc = _make_encoder_df()
        enc = enc.drop(columns=["time_idx"])
        dec = _make_decoder_df(pd.Timestamp("2025-01-01 12:00:00", tz="UTC"))
        with pytest.raises(ValueError, match="time_idx"):
            model.predict(enc, dec)

    def test_raises_if_decoder_missing_time_idx(self):
        model = self._build_model()
        enc = _make_encoder_df()
        dec = _make_decoder_df(pd.Timestamp("2025-01-01 12:00:00", tz="UTC"))
        dec = dec.drop(columns=["time_idx"])
        with pytest.raises(ValueError, match="time_idx"):
            model.predict(enc, dec)

    def test_raises_if_encoder_wrong_row_count(self):
        model = self._build_model()
        enc = _make_encoder_df(n=71)
        dec = _make_decoder_df(pd.Timestamp("2025-01-01 12:00:00", tz="UTC"))
        with pytest.raises(ValueError, match="72"):
            model.predict(enc, dec)

    def test_raises_if_decoder_wrong_row_count(self):
        model = self._build_model()
        enc = _make_encoder_df()
        dec = _make_decoder_df(pd.Timestamp("2025-01-01 12:00:00", tz="UTC"))
        dec = dec.iloc[:47]  # drop one row
        with pytest.raises(ValueError, match="48"):
            model.predict(enc, dec)

    def test_raises_if_decoder_fxx_incomplete(self):
        model = self._build_model()
        enc = _make_encoder_df()
        dec = _make_decoder_df(pd.Timestamp("2025-01-01 12:00:00", tz="UTC"))
        # Replace fxx=1 with fxx=49 to break the set
        dec = dec.copy()
        dec.loc[dec["fxx"] == 1, "fxx"] = 49
        with pytest.raises(ValueError, match="fxx"):
            model.predict(enc, dec)

    def test_raises_if_decoder_tmp2m_has_nan(self):
        model = self._build_model()
        enc = _make_encoder_df()
        dec = _make_decoder_df(pd.Timestamp("2025-01-01 12:00:00", tz="UTC"))
        dec = dec.copy()
        dec.loc[0, "TMP_2m"] = float("nan")
        with pytest.raises(ValueError, match="TMP_2m"):
            model.predict(enc, dec)
