"""Loading, cleaning, checking and enriching the raw data."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "covid.json"

COLUMN_NAMES = {
    "Date": "date",
    "Total Confirmed Cases": "total_cases",
    "Total Deaths": "total_deaths",
    "Total Recovered": "total_recovered",
    "Active Cases": "active_cases",
    "Daily Confirmed Cases": "new_cases",
    "Daily  deaths": "new_deaths",
}

# South Africa's national lockdown levels over the period in the file.
PHASES = [
    ("Before lockdown", "2020-03-05", "2020-03-26"),
    ("Level 5", "2020-03-27", "2020-04-30"),
    ("Level 4", "2020-05-01", "2020-05-31"),
    ("Level 3", "2020-06-01", "2020-06-12"),
]
PHASE_ORDER = [name for name, _, _ in PHASES]


def load_raw(path=RAW_PATH):
    """Read the JSON file as-is, keeping every value as text."""
    return pd.read_json(path, dtype=False)


def to_number(series):
    """Turn values like '1 139' or 1139 into integers.

    The file uses a space as the thousands separator once numbers pass 999,
    so those values arrive as strings while smaller ones are already ints.
    """
    cleaned = series.astype(str).str.replace(r"\s", "", regex=True)
    return pd.to_numeric(cleaned, errors="raise").astype("int64")


def clean(raw):
    """Rename columns, fix types and sort by date."""
    df = raw.rename(columns=COLUMN_NAMES)
    missing = set(COLUMN_NAMES.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    for col in df.columns.drop("date"):
        df[col] = to_number(df[col])

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # The first row reports 0 new cases even though the running total is 1.
    # There is no earlier day to difference against, so the first case is
    # counted as new on that day.
    first = df.index[0]
    if df.loc[first, "new_cases"] == 0 and df.loc[first, "total_cases"] > 0:
        df.loc[first, "new_cases"] = df.loc[first, "total_cases"]

    return df


def check_quality(df):
    """Run consistency checks and return a table of results.

    Each row is one rule, how many days break it and whether it passed.
    """
    later = df.iloc[1:]
    checks = {
        "No missing values": int(df.isna().sum().sum()),
        "No duplicate dates": int(df["date"].duplicated().sum()),
        "No gaps between dates": int((df["date"].diff().dt.days.iloc[1:] != 1).sum()),
        "No negative numbers": int((df.drop(columns="date") < 0).sum().sum()),
        "Totals never go down": int(
            (df[["total_cases", "total_deaths", "total_recovered"]].diff().iloc[1:] < 0).sum().sum()
        ),
        "Active = cases - deaths - recovered": int(
            (df["total_cases"] - df["total_deaths"] - df["total_recovered"] != df["active_cases"]).sum()
        ),
        "New cases = change in total cases": int(
            (df["total_cases"].diff().iloc[1:] != later["new_cases"]).sum()
        ),
        "New deaths = change in total deaths": int(
            (df["total_deaths"].diff().iloc[1:] != later["new_deaths"]).sum()
        ),
    }
    result = pd.DataFrame({"check": list(checks), "days_failing": list(checks.values())})
    result["passed"] = result["days_failing"] == 0
    return result


def assign_phase(dates):
    phase = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, start, end in PHASES:
        phase[(dates >= start) & (dates <= end)] = name
    return pd.Categorical(phase, categories=PHASE_ORDER, ordered=True)


def add_features(df):
    """Add the derived columns used throughout the analysis."""
    out = df.copy()
    out["day"] = np.arange(len(out))
    out["weekday"] = out["date"].dt.day_name()
    out["phase"] = assign_phase(out["date"])

    out["new_recovered"] = out["total_recovered"].diff().fillna(out["total_recovered"]).astype("int64")

    out["cases_7d"] = out["new_cases"].rolling(7).mean()
    out["deaths_7d"] = out["new_deaths"].rolling(7).mean()

    # Growth of the running total, averaged over a week to smooth reporting noise.
    log_total = np.log(out["total_cases"])
    out["growth_7d"] = (log_total - log_total.shift(7)) / 7
    out["doubling_days"] = np.log(2) / out["growth_7d"]

    out["cfr"] = out["total_deaths"] / out["total_cases"]
    resolved = out["total_deaths"] + out["total_recovered"]
    out["resolved_cfr"] = (out["total_deaths"] / resolved).where(resolved > 0)
    out["recovery_share"] = out["total_recovered"] / out["total_cases"]
    out["active_share"] = out["active_cases"] / out["total_cases"]
    return out


def load_clean(path=RAW_PATH):
    """Shortcut: raw file in, analysis-ready table out."""
    return add_features(clean(load_raw(path)))
