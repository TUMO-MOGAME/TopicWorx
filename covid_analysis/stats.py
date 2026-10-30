"""Statistical summaries and tests."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from .data import PHASES

LN2 = np.log(2)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def doubling_time(rate):
    """Days to double at a given daily log growth rate (inf if not growing)."""
    rate = np.asarray(rate, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(rate > 0, LN2 / rate, np.inf)


def phase_summary(df):
    """Daily cases and deaths described per lockdown phase."""
    grouped = df.groupby("phase", observed=True)
    summary = grouped.agg(
        start=("date", "min"),
        end=("date", "max"),
        days=("date", "size"),
        cases_added=("new_cases", "sum"),
        mean_daily_cases=("new_cases", "mean"),
        median_daily_cases=("new_cases", "median"),
        max_daily_cases=("new_cases", "max"),
        deaths_added=("new_deaths", "sum"),
        mean_daily_deaths=("new_deaths", "mean"),
        max_daily_deaths=("new_deaths", "max"),
    )
    numeric = summary.select_dtypes("number").columns
    summary[numeric] = summary[numeric].round(1)
    return summary


def phase_growth(df):
    """Fit log(total cases) = a + r * day separately for each phase.

    r is the average daily growth rate in that phase. The confidence interval
    uses the t distribution on the slope's standard error.
    """
    rows = []
    for phase, g in df.groupby("phase", observed=True):
        fit = stats.linregress(g["day"], np.log(g["total_cases"]))
        t_crit = stats.t.ppf(0.975, len(g) - 2)
        low, high = fit.slope - t_crit * fit.stderr, fit.slope + t_crit * fit.stderr
        rows.append({
            "phase": phase,
            "days": len(g),
            "daily_growth_pct": 100 * (np.exp(fit.slope) - 1),
            "doubling_days": float(doubling_time(fit.slope)),
            "doubling_low": float(doubling_time(high)),
            "doubling_high": float(doubling_time(low)),
            "r_squared": fit.rvalue ** 2,
        })
    return pd.DataFrame(rows).set_index("phase").round(2)


def segmented_regression(df, column="cases_7d", maxlags=7):
    """Piecewise log-linear trend with a change of slope at each phase start.

    Fitted to the 7-day average of new cases. Day-to-day values in a time
    series are not independent, so standard errors use the Newey-West (HAC)
    correction instead of the usual OLS formula.

    Returns the fitted model and a table with the growth rate in every phase,
    plus a test of whether each phase changed the slope from the one before.
    """
    data = df.dropna(subset=[column]).query(f"{column} > 0")
    X = pd.DataFrame({"const": 1.0}, index=data.index)
    starts = {}
    for i, (name, start, _) in enumerate(PHASES):
        offset = (pd.Timestamp(start) - df["date"].iloc[0]).days
        starts[name] = offset
        X[name] = data["day"] if i == 0 else np.clip(data["day"] - offset, 0, None)

    model = sm.OLS(np.log(data[column]), X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    rows = []
    names = list(starts)
    for i, name in enumerate(names):
        weights = np.zeros(len(model.params))
        for j in range(i + 1):
            weights[model.params.index.get_loc(names[j])] = 1
        slope = model.t_test(weights)
        rate = float(slope.effect[0])
        low, high = slope.conf_int()[0]
        rows.append({
            "phase": name,
            "daily_growth_pct": 100 * (np.exp(rate) - 1),
            "growth_low_pct": 100 * (np.exp(low) - 1),
            "growth_high_pct": 100 * (np.exp(high) - 1),
            "doubling_days": float(doubling_time(rate)),
            "change_vs_previous_p": float(model.pvalues[name]) if i > 0 else np.nan,
        })
    table = pd.DataFrame(rows).set_index("phase").round(3)
    return model, table


def weekday_effect(df, start="2020-04-01"):
    """Check whether some weekdays report more or fewer cases than expected.

    Each day's count is divided by a centred 7-day average, so a ratio of 1.0
    means a normal day. A Kruskal-Wallis test then asks whether the ratios
    differ by weekday. Early days are skipped because counts were too small.
    Ratios from the same week share an average, so they are not fully
    independent; treat the p-value as a guide rather than an exact figure.
    """
    w = df[df["date"] >= start].copy()
    w["ratio"] = w["new_cases"] / w["new_cases"].rolling(7, center=True).mean()
    w = w.dropna(subset=["ratio"])

    groups = [w.loc[w["weekday"] == day, "ratio"] for day in WEEKDAYS]
    result = stats.kruskal(*groups)

    table = w.groupby("weekday")["ratio"].agg(["mean", "median", "std", "count"]).reindex(WEEKDAYS)
    table["sem"] = table["std"] / np.sqrt(table["count"])
    return table.round(3), result


def fatality_rates(df):
    """Case fatality on the last day, with 95% Wilson intervals.

    Naive CFR divides deaths by all confirmed cases, which understates the
    risk while many cases are still open. Resolved CFR only counts cases with
    a known outcome, which tends to overstate it early on. The true figure
    usually sits between the two. The intervals only cover sampling error,
    not the reporting delays that drive the gap between the two measures.
    """
    last = df.iloc[-1]
    deaths, cases, recovered = int(last["total_deaths"]), int(last["total_cases"]), int(last["total_recovered"])
    rows = []
    for name, denom in [("Naive (deaths / cases)", cases),
                        ("Resolved (deaths / (deaths + recovered))", deaths + recovered)]:
        low, high = proportion_confint(deaths, denom, method="wilson")
        rows.append({"measure": name, "deaths": deaths, "denominator": denom,
                     "rate_pct": 100 * deaths / denom, "low_pct": 100 * low, "high_pct": 100 * high})
    return pd.DataFrame(rows).set_index("measure").round(2)


def lagged_cfr(df, lags=(0, 7, 14, 21)):
    """Deaths in the last week divided by cases from `lag` days earlier.

    Deaths trail infections, so comparing them with the cases of the same
    week understates fatality while numbers are rising.
    """
    rows = []
    for lag in lags:
        rows.append({
            "lag_days": lag,
            "rate_pct": 100 * df["deaths_7d"].iloc[-1] / df["cases_7d"].shift(lag).iloc[-1],
        })
    return pd.DataFrame(rows).set_index("lag_days").round(2)


def case_death_correlation(df, start="2020-04-15", max_lag=21):
    """Compare a naive correlation with a detrended, lagged one.

    Both raw series rise over time, so correlating them directly mostly
    measures the shared trend. Correlating day-on-day changes in their
    log 7-day averages removes that trend before looking for a lag.
    """
    pearson = stats.pearsonr(df["new_cases"], df["new_deaths"])
    sub = df.set_index("date").loc[start:]
    case_change = np.log(sub["cases_7d"]).diff()
    death_change = np.log(sub["deaths_7d"]).diff()
    rows = []
    for lag in range(max_lag + 1):
        shifted = case_change.shift(lag)
        pairs = int((shifted.notna() & death_change.notna()).sum())
        rows.append({"lag_days": lag, "correlation": shifted.corr(death_change), "pairs": pairs,
                     "noise_band": 1.96 / np.sqrt(pairs)})
    lagged = pd.DataFrame(rows).set_index("lag_days")
    return {"raw_r": pearson.statistic, "raw_p": pearson.pvalue, "lagged": lagged.round(3)}


def project_cases(df, fit_days=21, ahead=14):
    """Extend the recent log-linear trend in total cases.

    This is a simple 'if nothing changes' line, not a forecast model. It
    ignores changes in testing, policy and behaviour.
    """
    recent = df.tail(fit_days)
    X = sm.add_constant(recent["day"])
    model = sm.OLS(np.log(recent["total_cases"]), X).fit()

    future_days = np.arange(df["day"].iloc[-1] + 1, df["day"].iloc[-1] + 1 + ahead)
    pred = model.get_prediction(sm.add_constant(pd.Series(future_days, name="day"), has_constant="add"))
    frame = pred.summary_frame(alpha=0.05)
    out = pd.DataFrame({
        "date": df["date"].iloc[-1] + pd.to_timedelta(np.arange(1, ahead + 1), unit="D"),
        "projected": np.exp(frame["mean"]).values,
        "low": np.exp(frame["obs_ci_lower"]).values,
        "high": np.exp(frame["obs_ci_upper"]).values,
    })
    return model, out
