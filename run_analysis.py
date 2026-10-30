"""Run the whole analysis from the command line.

    python run_analysis.py

Writes the cleaned data, every chart and a plain-text summary into outputs/.
"""

import pandas as pd

from covid_analysis import plots, stats
from covid_analysis.data import PROJECT_ROOT, add_features, check_quality, clean, load_raw

OUTPUT = PROJECT_ROOT / "outputs"
FIGURES = OUTPUT / "figures"


def main():
    raw = load_raw()
    cleaned = clean(raw)
    quality = check_quality(cleaned)
    if not quality["passed"].all():
        print("Warning: some data quality checks failed")
    df = add_features(cleaned)

    OUTPUT.mkdir(exist_ok=True)
    df.to_csv(OUTPUT / "covid_clean.csv", index=False)

    growth = stats.phase_growth(df)
    _, segmented = stats.segmented_regression(df)
    weekday_table, kruskal = stats.weekday_effect(df)
    correlation = stats.case_death_correlation(df)
    _, projection = stats.project_cases(df)

    plots.apply_style()
    charts = {
        "01_daily_cases": plots.daily_cases(df),
        "02_cumulative_log": plots.cumulative_log(df, growth),
        "03_doubling_time": plots.doubling_trend(df),
        "04_daily_deaths": plots.daily_deaths(df),
        "05_outcomes": plots.outcome_breakdown(df),
        "06_recoveries_reporting": plots.recovered_reporting(df),
        "07_fatality_rate": plots.fatality_trend(df),
        "08_weekday_effect": plots.weekday_chart(weekday_table, kruskal.pvalue),
        "09_case_death_lag": plots.lag_correlation(correlation),
        "10_projection": plots.projection_chart(df, projection),
    }
    for name, fig in charts.items():
        plots.save(fig, name, FIGURES)

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        sections = [
            ("Data quality checks", quality.to_string(index=False)),
            ("Phase summary", stats.phase_summary(df).to_string()),
            ("Growth of total cases by phase", growth.to_string()),
            ("Segmented regression on 7-day average of new cases", segmented.to_string()),
            ("Weekday reporting ratio", f"{weekday_table.to_string()}\n"
                                        f"Kruskal-Wallis H = {kruskal.statistic:.2f}, p = {kruskal.pvalue:.3f}"),
            ("Case fatality on the last day", stats.fatality_rates(df).to_string()),
            ("Fatality against cases from earlier weeks", stats.lagged_cfr(df).to_string()),
            ("Cases vs deaths", f"Raw Pearson r = {correlation['raw_r']:.3f}\n"
                                f"Detrended correlations, with the range expected by chance at each lag:\n"
                                f"{correlation['lagged'].T.to_string()}"),
            ("Trend carried forward 14 days",
             projection.round({"projected": 0, "low": 0, "high": 0}).to_string(index=False)),
        ]
    text = "\n\n".join(f"{title}\n{'-' * len(title)}\n{body}" for title, body in sections)
    (OUTPUT / "summary.txt").write_text(text, encoding="utf-8")

    print(text)
    print(f"\nSaved {len(charts)} charts to {FIGURES.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
