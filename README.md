# COVID-19 daily figures: data analysis

Analysis of `covid.json`, 100 days of COVID-19 figures from 5 March to 12 June 2020. The project cleans the raw file, checks it for consistency, explores it with charts and runs a set of statistical tests on growth, reporting patterns and fatality.

The full write-up is the notebook **`notebooks/covid_analysis.ipynb`**. Two ready-to-read copies open in any browser without installing anything:

- **`outputs/covid_analysis_presentation.html`**: the report without code, for presenting
- **`outputs/covid_analysis.html`**: the same report with all code shown

---

## Key findings

| | |
|---|---|
| Data quality | Five of the six number columns stored values over 999 as text (`"1 139"`). Once fixed, every consistency check passes. Recoveries were reported in batches until mid-May. |
| Growth before lockdown | Total cases doubled about every **2.1 days**. |
| Growth after lockdown | Doubling time of total cases stretched to **12 to 15 days**. Growth in daily new cases fell from about 23% to 3% a day, a statistically significant change (p < 0.001, segmented regression with Newey-West errors). |
| Level 4 | Growth in daily cases rose from about 3% to 7% a day (p = 0.01). |
| Level 3 | 12 days added 29,244 cases, nearly as many as the 88 days before. Too short to show a significant change in growth. |
| Weekday effect | Small dip on Mondays and Tuesdays, not significant (Kruskal-Wallis p = 0.25). |
| Fatality among confirmed cases | Between **2.2%** (deaths / all cases) and **3.7%** (deaths / resolved cases) on 12 June. The gap comes from reporting delays, which matter far more than the narrow sampling intervals suggest. |
| Cases vs deaths | The raw correlation (r = 0.91) is driven by the shared trend. After removing it, most lags show nothing beyond chance; there is only a weak hint of a ~17-day lag. |
| Trend | If the last three weeks' growth held, cases would pass ~133,000 by 26 June. This is a "nothing changes" line, not a forecast. |

The file doesn't name the country. The dates and figures match South Africa's early outbreak, so the analysis uses South Africa's lockdown levels to split the timeline.

---

## Project layout

```
.
├── covid.json                  raw data (unchanged)
├── covid_analysis/             the analysis code
│   ├── data.py                 loading, cleaning, checks, new columns
│   ├── stats.py                statistical summaries and tests
│   ├── plots.py                charts
│   └── report.py               notebook styling: headline cards and tables
├── notebooks/
│   └── covid_analysis.ipynb    the report
├── tests/
│   └── test_data.py            automated tests
├── outputs/                    created by the code
│   ├── covid_clean.csv         cleaned data with derived columns
│   ├── summary.txt             every statistical result as plain text
│   ├── covid_analysis.html     the report as a web page, with code
│   ├── covid_analysis_presentation.html   the report without code
│   └── figures/                all charts as PNG
├── run_analysis.py             runs everything from the command line
├── requirements.txt
└── pytest.ini
```

---

## How to run it

You need **Python 3.10 or newer**. The steps are the same on Windows, macOS and Linux; only the activate command differs.

**1. Get the code, create a virtual environment and install the libraries**

```bash
git clone https://github.com/TUMO-MOGAME/TopicWorx.git
cd TopicWorx
python -m venv .venv
```

Activate it:

| System | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Command Prompt) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Then install:

```bash
pip install -r requirements.txt
```

On macOS and Linux, use `python3` instead of `python` if `python` isn't found.

**2. Run the whole analysis**

```bash
python run_analysis.py
```

This prints every result and writes the cleaned data, `summary.txt` and the ten charts into `outputs/`.

**3. Open the report notebook**

Register the environment as a Jupyter kernel once, so the notebook uses the libraries you just installed:

```bash
python -m ipykernel install --user --name covid-analysis --display-name "Python (covid-analysis)"
```

Then start Jupyter and open `notebooks/covid_analysis.ipynb`:

```bash
jupyter lab
```

In VS Code you can open the notebook directly and pick the `.venv` interpreter as the kernel instead.

To re-run the notebook and rebuild both HTML copies without opening it (this uses the kernel registered above):

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/covid_analysis.ipynb
python -m nbconvert --to html notebooks/covid_analysis.ipynb --output-dir outputs
python -m nbconvert --to html --no-input notebooks/covid_analysis.ipynb --output covid_analysis_presentation --output-dir outputs
```

**4. Run the tests**

```bash
python -m pytest
```

---

## External libraries

All of them are listed in `requirements.txt` and install with one `pip` command.

| Library | Used for |
|---|---|
| **pandas** | Loading the JSON, cleaning, rolling averages, grouping by phase and weekday |
| **numpy** | Logs, growth rates and other array maths |
| **scipy** | Linear regression per phase, Kruskal-Wallis test, Pearson correlation, t-distribution for confidence intervals |
| **statsmodels** | Segmented regression with Newey-West (HAC) standard errors, Wilson confidence intervals for fatality rates, prediction intervals for the projection |
| **matplotlib** | All charts |
| **jupyter**, **ipykernel** | Running the notebook report |
| **nbconvert** | Running the notebook from the command line and exporting it to HTML |
| **jinja2** | Needed by pandas to render the styled tables in the notebook |
| **pytest** | Automated tests |

`pathlib` and `sys` come with Python.

---

## How the code works

The work is split into three modules so that the notebook, the command-line script and the tests all run the same code.

### `covid_analysis/data.py`: getting the data ready

- **`load_raw()`** reads `covid.json` with `dtype=False`, so pandas doesn't guess types and every value stays as it was in the file.
- **`to_number()`** strips all whitespace from a column and converts it to integers. `"1 139"` becomes `1139`. It uses `errors="raise"`, so a value that isn't a number stops the program rather than quietly turning into a blank.
- **`clean()`**
  1. renames the columns to short names (`Daily  deaths` had a double space)
  2. parses dates
  3. converts the six number columns
  4. sorts by date and removes duplicate dates
  5. counts the first case as new on 5 March (the file had 0 there because there was no earlier day to subtract)
- **`check_quality()`** tests eight rules: no missing values, no duplicate or skipped dates, no negatives, totals that never fall, `active = confirmed - deaths - recovered`, and daily figures that match the change in totals. It returns a table showing how many days break each rule.
- **`add_features()`** adds the columns used for analysis:
  - lockdown `phase` and `weekday`
  - daily recoveries
  - 7-day averages of cases and deaths
  - weekly growth rate and doubling time
  - two versions of the case fatality rate
  - recovered and active shares

  It works on a copy, so the cleaned table is never changed by accident.
- **`load_clean()`** chains the above for convenience.

### `covid_analysis/stats.py`: the statistics

- **`phase_summary()`**: totals, means, medians and maxima of daily cases and deaths per lockdown phase.
- **`phase_growth()`**: for each phase, fits a straight line to log(total cases) over time. The slope is the daily growth rate, converted to a doubling time with ln(2) / slope, with 95% confidence intervals.
- **`segmented_regression()`**: one regression across all 100 days on log(7-day average of new cases), allowing the slope to change at each phase start. Daily values are correlated with their neighbours, so the standard errors use the Newey-West correction. It returns the growth rate in each phase and a p-value for whether each phase changed the trend.
- **`weekday_effect()`**: divides each day's cases by a centred 7-day average and uses a Kruskal-Wallis test to see if any weekday is consistently high or low.
- **`fatality_rates()`**: naive and resolved case fatality rates on the last day, with Wilson confidence intervals.
- **`lagged_cfr()`**: deaths in the final week divided by the cases from 0, 7, 14 and 21 days earlier, showing how much the fatality estimate depends on the delay between diagnosis and death.
- **`case_death_correlation()`**: compares the raw correlation between daily cases and deaths with the correlation after the trend is removed, across lags of 0 to 21 days.
- **`project_cases()`**: fits the last 21 days of total cases on a log scale and extends it 14 days with a 95% prediction interval. It's a "nothing changes" line, not a forecast.

### `covid_analysis/plots.py`: the charts

One function per chart, each returning a matplotlib figure, so the notebook can show it and `run_analysis.py` can save it. `apply_style()` sets one shared look: light grid, no top or right borders, and a three-colour palette that was checked to stay distinguishable for colour-blind readers. Lockdown changes are marked with thin vertical lines on every time chart, and key values are labelled on the chart itself so it reads without a legend lookup.

### `covid_analysis/report.py`: notebook presentation

- **`headline_cards()`** builds the row of summary cards at the top of the report from the data itself, so the numbers can never drift from the analysis.
- **`table()`** wraps pandas' `Styler`: a navy header row, zebra striping, thousands separators and per-column number formats, so every table in the report looks the same.
- **`flag_text_values()`** highlights raw values stored as text, which makes the formatting problem visible at a glance.
- **`pass_fail()`** shows the data quality checks as green Pass / red Fail labels.

The markdown cells use inline HTML styling (a title banner, numbered section badges, and coloured boxes for findings, methods and cautions). Inline styles are used because JupyterLab, VS Code and the HTML export all keep them, while `<style>` blocks get stripped.

### `run_analysis.py`

Runs the whole pipeline from the command line. It loads and cleans the data, warns if any quality check fails, saves the cleaned CSV, runs every statistical function, saves all ten charts and writes the results to `outputs/summary.txt`.

### `tests/test_data.py`

Eleven tests covering the number parsing, the cleaned shape and types, the first-day fix, daily figures adding up to the final totals, the quality checks (including one that deliberately breaks the data to make sure the check catches it), phase labels, the doubling-time maths and the direction of the main results.

---

## Limitations

- **No testing data.** More confirmed cases can mean more spread or more testing. This file can't tell them apart.
- **Country assumed** from the dates. Only the phase analysis depends on this.
- **Figures are counted by date of report**, which may trail the real test or death date.
- **National totals only**, which can hide very different regional trends.
- **Level 3 covers only 12 days**, so conclusions about it are tentative.

---

## Troubleshooting

- **`No such kernel named covid-analysis`**: run the `ipykernel install` command from step 3 first, or pick any Python kernel that has the requirements installed.
- **Windows: `filename or extension is too long` during `pip install`**: Jupyter installs some deeply nested files. Put the project in a short folder such as `C:\projects\TopicWorx`, or turn on Windows long-path support.
