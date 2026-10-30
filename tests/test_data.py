import numpy as np
import pandas as pd
import pytest

from covid_analysis import stats
from covid_analysis.data import add_features, check_quality, clean, load_clean, load_raw, to_number


@pytest.fixture(scope="module")
def df():
    return load_clean()


def test_to_number_handles_space_separators():
    values = pd.Series([12, "1 139", "61 927", "1 080"])
    assert to_number(values).tolist() == [12, 1139, 61927, 1080]


def test_clean_gives_expected_shape_and_types():
    cleaned = clean(load_raw())
    assert len(cleaned) == 100
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    numeric = cleaned.drop(columns="date")
    assert all(pd.api.types.is_integer_dtype(t) for t in numeric.dtypes)


def test_first_day_counts_first_case():
    cleaned = clean(load_raw())
    assert cleaned.loc[0, "new_cases"] == cleaned.loc[0, "total_cases"] == 1


def test_new_cases_sum_to_final_total(df):
    assert df["new_cases"].sum() == df["total_cases"].iloc[-1]
    assert df["new_deaths"].sum() == df["total_deaths"].iloc[-1]


def test_all_quality_checks_pass():
    assert check_quality(clean(load_raw()))["passed"].all()


def test_quality_check_catches_bad_active_count():
    broken = clean(load_raw())
    broken.loc[50, "active_cases"] += 10
    result = check_quality(broken).set_index("check")
    assert not result.loc["Active = cases - deaths - recovered", "passed"]


def test_every_day_has_a_phase(df):
    assert df["phase"].notna().all()
    assert df["phase"].iloc[0] == "Before lockdown"
    assert df["phase"].iloc[-1] == "Level 3"


def test_doubling_time_maths():
    assert stats.doubling_time(np.log(2)) == pytest.approx(1)
    assert np.isinf(stats.doubling_time(0))


def test_growth_slows_after_lockdown(df):
    growth = stats.phase_growth(df)
    assert growth.loc["Before lockdown", "doubling_days"] < growth.loc["Level 5", "doubling_days"]


def test_fatality_rates_are_ordered(df):
    rates = stats.fatality_rates(df)["rate_pct"]
    assert rates.iloc[0] < rates.iloc[1]


def test_add_features_does_not_change_input():
    cleaned = clean(load_raw())
    before = cleaned.copy()
    add_features(cleaned)
    pd.testing.assert_frame_equal(cleaned, before)
