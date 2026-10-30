"""Charts for the report. Every function returns a matplotlib Figure."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from .data import PHASES

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BAR_GREY = "#cfcec9"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
}


def apply_style():
    plt.rcParams.update(STYLE)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def _thousands(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _mark_phases(ax, label=True):
    """Thin vertical rules at each lockdown change, labelled along the top."""
    top = ax.get_ylim()[1]
    for name, start, _ in PHASES[1:]:
        x = pd.Timestamp(start)
        ax.axvline(x, color=INK_SOFT, linewidth=0.8, alpha=0.6, zorder=1)
        if label:
            ax.text(x, top, f" {name}", color=INK_SOFT, fontsize=8, va="top", ha="left")


def _subtitle(ax, text):
    ax.text(0, 1.01, text, transform=ax.transAxes, color=INK_SOFT, fontsize=9, va="bottom")


def daily_with_average(df, value, average, title, subtitle, colour=BLUE):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(df["date"], df[value], color=BAR_GREY, width=0.8, label="Reported that day")
    ax.plot(df["date"], df[average], color=colour, label="7-day average")
    ax.set_title(title)
    _subtitle(ax, subtitle)
    _date_axis(ax)
    _thousands(ax)
    ax.set_ylim(0, df[value].max() * 1.12)
    _mark_phases(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    return fig


def daily_cases(df):
    return daily_with_average(df, "new_cases", "cases_7d", "New confirmed cases per day",
                              "Grey bars are single days; the line smooths over a week")


def daily_deaths(df):
    return daily_with_average(df, "new_deaths", "deaths_7d", "New deaths per day",
                              "Grey bars are single days; the line smooths over a week", colour=ORANGE)


def cumulative_log(df, growth):
    """Total cases on a log scale with the fitted growth line for each phase."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df["date"], df["total_cases"], color=BLUE, label="Total confirmed cases")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for i, (phase, g) in enumerate(df.groupby("phase", observed=True)):
        slope, intercept = np.polyfit(g["day"], np.log(g["total_cases"]), 1)
        ax.plot(g["date"], np.exp(intercept + slope * g["day"]), color=INK, linewidth=1,
                linestyle=(0, (4, 3)), zorder=3)
        days = growth.loc[phase, "doubling_days"]
        mid = g.iloc[len(g) // 2]
        offset = (36, -30) if i == 0 else (0, -38)
        ax.annotate(f"doubles every\n{days:.1f} days" if days < 3 else f"doubles every\n{days:.0f} days",
                    xy=(mid["date"], mid["total_cases"]), xytext=offset, textcoords="offset points",
                    ha="center", fontsize=8, color=INK_SOFT)

    ax.set_title("Total confirmed cases (log scale)")
    _subtitle(ax, "A straight line on this scale means steady exponential growth; dashed lines are fitted per phase")
    _date_axis(ax)
    ax.set_ylim(0.8, df["total_cases"].max() * 3)
    _mark_phases(ax)
    return fig


def doubling_trend(df):
    data = df.dropna(subset=["doubling_days"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(data["date"], data["doubling_days"], color=BLUE)
    last = data.iloc[-1]
    ax.annotate(f"{last['doubling_days']:.1f} days", xy=(last["date"], last["doubling_days"]),
                xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_title("How long it takes total cases to double")
    _subtitle(ax, "Based on growth over the previous 7 days; higher is better")
    ax.set_ylabel("Days")
    _date_axis(ax)
    ax.set_ylim(0, data["doubling_days"].max() * 1.15)
    _mark_phases(ax)
    return fig


def outcome_breakdown(df):
    """Confirmed cases split into active, recovered and deaths."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    series = [("Active", df["active_cases"], BLUE),
              ("Deaths", df["total_deaths"], ORANGE),
              ("Recovered", df["total_recovered"], AQUA)]
    ax.stackplot(df["date"], [s for _, s, _ in series], colors=[c for _, _, c in series],
                 labels=[n for n, _, _ in series], edgecolor=SURFACE, linewidth=1)

    last = df.iloc[-1]
    x = last["date"] + pd.Timedelta(days=1)
    ax.text(x, last["active_cases"] / 2, f"Active {last['active_cases']:,}", va="center", fontsize=9, color=INK)
    ax.text(x, last["active_cases"] + last["total_deaths"] / 2, f"Deaths {last['total_deaths']:,}",
            va="center", fontsize=9, color=INK)
    ax.text(x, last["total_cases"] - last["total_recovered"] / 2, f"Recovered {last['total_recovered']:,}",
            va="center", fontsize=9, color=INK)

    ax.set_title("What happened to confirmed cases")
    _subtitle(ax, "Every confirmed case is either still active, recovered, or died")
    _date_axis(ax)
    _thousands(ax)
    ax.set_xlim(df["date"].iloc[0], df["date"].iloc[-1] + pd.Timedelta(days=14))
    ax.legend(loc="upper left")
    return fig


def recovered_reporting(df):
    """Shows that recoveries were updated in irregular jumps."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(df["date"], df["new_recovered"], color=AQUA, width=0.8)
    zero_days = (df["new_recovered"] == 0).sum()
    ax.set_title("Recoveries added per day")
    _subtitle(ax, f"{zero_days} of {len(df)} days added no recoveries at all, then large batches arrived at once")
    _date_axis(ax)
    _thousands(ax)
    _mark_phases(ax)
    return fig


def fatality_trend(df):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    data = df[df["total_deaths"] > 0]
    ax.plot(data["date"], 100 * data["cfr"], color=BLUE, label="Deaths / all confirmed cases")
    ax.plot(data["date"], 100 * data["resolved_cfr"], color=ORANGE,
            label="Deaths / cases with a known outcome")
    last = data.iloc[-1]
    for col in ["cfr", "resolved_cfr"]:
        ax.annotate(f"{100 * last[col]:.1f}%", xy=(last["date"], 100 * last[col]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_title("Case fatality rate, two ways")
    _subtitle(ax, "The real rate usually ends up between these two lines once all cases resolve")
    ax.set_ylabel("Percent")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    _date_axis(ax)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    return fig


def weekday_chart(table, p_value):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table))
    ax.axhline(1, color=INK_SOFT, linewidth=0.8)
    t_crit = stats.t.ppf(0.975, table["count"] - 1)
    ax.errorbar(x, table["mean"], yerr=t_crit * table["sem"], fmt="o", color=BLUE,
                ecolor=BLUE, elinewidth=1.5, capsize=0, markersize=8)
    ax.set_xticks(x, table.index)
    ax.set_title("Do some weekdays report more cases?")
    _subtitle(ax, f"Day's count / centred 7-day average, with 95% intervals. Kruskal-Wallis p = {p_value:.2f}")
    ax.set_ylabel("Ratio to weekly average")
    return fig


def lag_correlation(result):
    lagged = result["lagged"]
    fig, ax = plt.subplots(figsize=(8, 4))
    band = lagged["noise_band"]
    ax.fill_between(lagged.index, -band, band, color=GRID, alpha=0.8, linewidth=0, label="Range expected by chance")
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    ax.plot(lagged.index, lagged["correlation"], color=BLUE, marker="o", markersize=5)
    ax.set_title("Do changes in cases show up in deaths later?")
    _subtitle(ax, "Correlation between daily changes in cases and in deaths, cases shifted back by the lag")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper left")
    return fig


def projection_chart(df, projection):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    recent = df[df["date"] >= df["date"].iloc[-1] - pd.Timedelta(days=35)]
    ax.plot(recent["date"], recent["total_cases"], color=BLUE, label="Reported")
    ax.plot(projection["date"], projection["projected"], color=BLUE, linestyle=(0, (4, 3)),
            label="Trend carried forward")
    ax.fill_between(projection["date"], projection["low"], projection["high"], color=BLUE, alpha=0.15,
                    linewidth=0, label="95% prediction interval")
    end = projection.iloc[-1]
    ax.annotate(f"{end['projected']:,.0f}\n({end['low']:,.0f} to {end['high']:,.0f})",
                xy=(end["date"], end["projected"]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.set_title("Where the last three weeks' trend points")
    _subtitle(ax, "Not a forecast: assumes testing, policy and behaviour stay as they were")
    _date_axis(ax)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    _thousands(ax)
    ax.set_xlim(recent["date"].iloc[0], end["date"] + pd.Timedelta(days=9))
    ax.legend(loc="upper left")
    return fig


def save(fig, name, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
