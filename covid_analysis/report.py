"""Display helpers for the notebook: headline cards and styled tables."""

from IPython.display import HTML

INK = "#0b0b0b"
INK_SOFT = "#52514e"
NAVY = "#12213a"
LINE = "#e6e5e1"
ZEBRA = "#f8f8f6"
HOVER = "#eef4fc"
GOOD = "#1f7a3d"
BAD = "#b42318"
FLAG_BG = "#fdf0d5"
FLAG_INK = "#7a4b00"

TABLE_STYLES = [
    {"selector": "", "props": f"border-collapse: collapse; font-size: 13px; margin: 8px 0 16px; "
                              f"background: #ffffff; border: 1px solid {LINE};"},
    {"selector": "caption", "props": "caption-side: top; text-align: left; font-size: 14px; font-weight: 600; "
                                     "padding: 0 0 8px; color: inherit;"},
    {"selector": "thead th", "props": f"background: {NAVY}; color: #ffffff; font-weight: 600; "
                                      f"padding: 9px 14px; text-align: right; border: none;"},
    {"selector": "tbody tr", "props": "background: #ffffff;"},
    {"selector": "tbody tr:nth-child(even)", "props": f"background: {ZEBRA};"},
    {"selector": "tbody tr:hover", "props": f"background: {HOVER};"},
    {"selector": "tbody th", "props": f"background: #f4f4f2; color: {INK}; font-weight: 600; text-align: left; "
                                     f"padding: 8px 14px; border-bottom: 1px solid {LINE}; white-space: nowrap;"},
    {"selector": "td", "props": f"color: {INK}; padding: 8px 14px; text-align: right; "
                                f"border-bottom: 1px solid {LINE}; white-space: nowrap;"},
]


def _default_format(dtype, precision):
    if dtype.kind in "iu":
        return "{:,.0f}"
    if dtype.kind == "f":
        return "{:,.%df}" % precision
    return None


def table(df, formats=None, caption=None, hide_index=False, precision=2):
    """Return a pandas Styler with consistent formatting and house styling."""
    df = df.rename_axis(index=None)
    formats = formats or {}
    formatter = {col: formats.get(col, _default_format(df[col].dtype, precision)) for col in df.columns}
    formatter = {col: f for col, f in formatter.items() if f is not None}
    styler = df.style.format(formatter, na_rep="–").set_table_styles(TABLE_STYLES)
    if caption:
        styler = styler.set_caption(caption)
    if hide_index:
        styler = styler.hide(axis="index")
    return styler


def flag_text_values(df, caption=None):
    """Highlight raw values that are stored as text instead of numbers."""
    styler = df.style.set_table_styles(TABLE_STYLES)
    styler = styler.map(lambda v: f"background: {FLAG_BG}; color: {FLAG_INK}; font-weight: 700;"
                        if isinstance(v, str) and " " in v else "")
    if caption:
        styler = styler.set_caption(caption)
    return styler


def pass_fail(df, column="passed", caption=None):
    """Show a boolean column as coloured Pass / Fail labels."""
    shown = df.copy()
    shown[column] = shown[column].map({True: "✓ Pass", False: "✗ Fail"})
    styler = table(shown, caption=caption, hide_index=True)
    styler = styler.map(lambda v: f"color: {GOOD} !important; font-weight: 700;" if v == "✓ Pass"
                        else (f"color: {BAD} !important; font-weight: 700;" if v == "✗ Fail" else ""), subset=[column])
    return styler.set_properties(subset=[df.columns[0]], **{"text-align": "left !important"})


def _card(label, value, note, accent):
    return (
        f'<div style="flex: 1 1 130px; background: #ffffff; border: 1px solid {LINE}; '
        f'border-top: 4px solid {accent}; border-radius: 10px; padding: 14px 18px;">'
        f'<div style="font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: {INK_SOFT};">{label}</div>'
        f'<div style="font-size: 28px; font-weight: 700; color: {INK}; margin: 4px 0 2px;">{value}</div>'
        f'<div style="font-size: 12px; color: {INK_SOFT};">{note}</div>'
        f"</div>"
    )


def headline_cards(df, growth):
    """The main numbers at the end of the data, as a row of cards."""
    last = df.iloc[-1]
    first_phase, last_phase = growth.index[0], growth.index[-1]
    cards = [
        ("Confirmed cases", f"{last['total_cases']:,}", f"by {last['date']:%d %B %Y}", "#2a78d6"),
        ("Deaths", f"{last['total_deaths']:,}", f"{100 * last['cfr']:.1f}% of confirmed cases", "#eb6834"),
        ("Recovered", f"{last['total_recovered']:,}", f"{100 * last['recovery_share']:.0f}% of confirmed cases", "#1baf7a"),
        ("Still active", f"{last['active_cases']:,}", f"{100 * last['active_share']:.0f}% of confirmed cases", "#2a78d6"),
        ("Doubling time (days)",
         f"{growth.loc[first_phase, 'doubling_days']:.1f} → {growth.loc[last_phase, 'doubling_days']:.0f}",
         f"{first_phase.lower()} vs {last_phase}", "#6250d6"),
    ]
    body = "".join(_card(*c) for c in cards)
    return HTML(f'<div style="display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 4px;">{body}</div>')
