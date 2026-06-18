from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphatrade.scripts import build_m1_sample_index
from alphatrade.scripts import build_m1_sample_index_v2 as m1_index
from alphatrade.scripts import build_sample_index


INDEX_BUILDERS = [m1_index, build_m1_sample_index, build_sample_index]


def _bars(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eob": pd.date_range("2024-01-01 09:00:00", periods=n, freq="1min"),
            "close": np.arange(100.0, 100.0 + n),
            "segment_id": np.ones(n, dtype=np.int32),
        }
    )


@pytest.mark.parametrize("builder", INDEX_BUILDERS)
def test_sample_index_records_max_target_eob(builder) -> None:
    bars = builder.compute_labels(_bars(), horizons=[1, 2])

    samples = builder.generate_sample_indices(
        bars, lookback=2, horizons=[1, 2], stride=1, require_same_segment=True
    )

    first = samples.iloc[0]
    assert first["t"] == 1
    assert first["eob"] == bars.loc[1, "eob"]
    assert first["target_eob"] == bars.loc[3, "eob"]


@pytest.mark.parametrize("builder", INDEX_BUILDERS)
def test_split_by_time_drops_samples_whose_target_crosses_split_boundary(builder) -> None:
    bars = builder.compute_labels(_bars(), horizons=[1, 2])
    samples = builder.generate_sample_indices(
        bars, lookback=2, horizons=[1, 2], stride=1, require_same_segment=True
    )

    train, val, test = builder.split_by_time(
        samples,
        train_start="2024-01-01 09:00:00",
        train_end="2024-01-01 09:05:00",
        val_start="2024-01-01 09:05:00",
        val_end="2024-01-01 09:08:00",
        test_start="2024-01-01 09:08:00",
        test_end="2024-01-01 09:10:00",
    )

    assert not train.empty
    assert (train["eob"] < pd.Timestamp("2024-01-01 09:05:00")).all()
    assert (train["target_eob"] < pd.Timestamp("2024-01-01 09:05:00")).all()
    assert pd.Timestamp("2024-01-01 09:03:00") not in set(train["eob"])
    assert (val["target_eob"] < pd.Timestamp("2024-01-01 09:08:00")).all()
    assert test.empty
