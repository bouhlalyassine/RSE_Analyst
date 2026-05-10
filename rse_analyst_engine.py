"""Deterministic RSE Analyst engine.

The Streamlit page uses this module before falling back to pandasai. The flow is
kept explicit and reusable: prepare data, classify the request, build analysis
tables, generate charts, and optionally assemble a PDF report.
"""

from __future__ import annotations

import io
import re
import textwrap
from html import escape
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_COLUMNS = {
    "tep": "Conso_TEP",
    "dh": "Conso_DH",
    "cout": "Conso_DH",
    "cost": "Conso_DH",
    "unite": "Conso_Unite",
    "consommation": "Conso_TEP",
}

DIMENSION_KEYWORDS = {
    "Site": ["site", "sites"],
    "BU": ["bu", "business unit", "business-unit"],
    "Societe": ["societe", "societes", "filiale", "filiales"],
    "Region": ["region", "regions"],
    "Activite": ["activite", "activites"],
    "Source_energie": ["source", "sources", "energie", "energies", "type"],
    "Campagne": ["campagne", "campagnes", "annee", "annees", "evolution", "tendance"],
}

STRUCTURED_KEYWORDS = [
    "analyse",
    "rapport",
    "pdf",
    "graphique",
    "visualisation",
    "courbe",
    "histogramme",
    "tableau",
    "kpi",
    "indicateur",
    "aggregation",
    "comparaison",
    "comparer",
    "evolution",
    "tendance",
    "ecart",
    "top",
    "bottom",
    "meilleur",
    "pire",
    "consommation",
    "energie",
    "tep",
    "dh",
    "cout",
    "site",
    "source",
    "campagne",
    "bu",
    "region",
    "activite",
]


@dataclass
class RSEAnalysisTable:
    name: str
    dataframe: pd.DataFrame
    description: str = ""


@dataclass
class RSEChart:
    name: str
    data: bytes
    description: str = ""
    mime_type: str = "image/png"
    file_name: str = "chart.png"


@dataclass
class RSEAnalysisResult:
    output_type: str
    summary: str
    tables: list[RSEAnalysisTable] = field(default_factory=list)
    charts: list[RSEChart] = field(default_factory=list)
    pdf_data: bytes | None = None
    pdf_filename: str = "rapport_rse_analyst.pdf"
    warnings: list[str] = field(default_factory=list)
    used_engine: str = "deterministic"


def should_use_structured_engine(query: str) -> bool:
    normalized = _normalize(query)
    return any(keyword in normalized for keyword in STRUCTURED_KEYWORDS)


def classify_rse_request(query: str) -> str:
    normalized = _normalize(query)
    if any(word in normalized for word in ["rapport", "pdf", "document", "presentation"]):
        return "pdf"
    if any(word in normalized for word in ["graphique", "visualisation", "courbe", "histogramme", "chart", "plot"]):
        return "chart"
    if any(word in normalized for word in ["tableau", "kpi", "aggregation", "comparaison", "ecart", "top", "bottom"]):
        return "table"
    return "summary"


def analyze_rse_data(df: pd.DataFrame, query: str, output_dir=None) -> RSEAnalysisResult:
    """Run the full structured analysis pipeline for one user query.

    Charts and PDF reports are produced entirely in memory (no files written
    to disk). The ``output_dir`` argument is accepted for backward
    compatibility but ignored.

    Only the artefacts explicitly requested by the user are returned for display:
    - "table" requests -> only the matching table(s)
    - "chart" requests -> only the chart, no tables
    - "pdf" requests   -> only the download link (no inline summary/tables/charts)
    The full set is still computed internally so the PDF report stays complete.
    """
    clean_df = prepare_rse_dataframe(df)
    clean_df, filter_warnings = apply_query_filters(clean_df, query)

    request_type = classify_rse_request(query)
    metric = infer_metric(query, clean_df)
    dimension = infer_dimension(query, default="Source_energie")

    full_tables = build_analysis_tables(clean_df, query=query, metric=metric, dimension=dimension)
    summary = build_executive_summary(clean_df, full_tables, metric)

    full_charts: list[RSEChart] = []
    if request_type in {"chart", "pdf"}:
        try:
            full_charts = build_relevant_charts(clean_df, query=query, metric=metric, dimension=dimension)
        except Exception as exc:
            filter_warnings.append(f"Generation des graphiques indisponible: {exc}")

    pdf_data: bytes | None = None
    pdf_filename = f"rapport_rse_analyst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    if request_type == "pdf":
        try:
            pdf_data = build_pdf_report(
                clean_df,
                query=query,
                summary=summary,
                tables=full_tables,
                charts=full_charts or build_relevant_charts(clean_df, query=query, metric=metric, dimension=dimension),
            )
        except Exception as exc:
            filter_warnings.append(f"Generation du rapport PDF indisponible: {exc}")

    # Trim the on-screen output to only what the user asked for.
    if request_type == "pdf":
        display_tables: list[RSEAnalysisTable] = []
        display_charts: list[RSEChart] = []
        display_summary = ""
    elif request_type == "chart":
        display_tables = []
        display_charts = full_charts
        display_summary = ""
    elif request_type == "table":
        display_tables = select_requested_tables(full_tables, query)
        display_charts = []
        display_summary = ""
    else:  # "summary"
        display_tables = select_requested_tables(full_tables, query)[:1]
        display_charts = []
        display_summary = summary

    return RSEAnalysisResult(
        output_type=request_type,
        summary=display_summary,
        tables=display_tables,
        charts=display_charts,
        pdf_data=pdf_data,
        pdf_filename=pdf_filename,
        warnings=filter_warnings,
    )


def select_requested_tables(tables: list[RSEAnalysisTable], query: str) -> list[RSEAnalysisTable]:
    """Return only the tables that match what the user explicitly asked for.

    Keywords drive the selection:
    - top / meilleur          -> "Top categories"
    - bottom / pire / moins   -> "Bottom categories"
    - ecart / comparaison     -> "Ecarts ..."
    - aggregation / repartition / par <dim> / no specific keyword -> "Aggregation par ..."
    """
    normalized = _normalize(query)
    keep_top = any(w in normalized for w in ["top", "meilleur", "meilleurs", "plus consommateur", "plus consommatrice"])
    keep_bottom = any(w in normalized for w in ["bottom", "pire", "pires", "moins consommateur", "moins consommatrice"])
    keep_gap = any(w in normalized for w in ["ecart", "ecarts", "comparaison", "comparer", " vs "])
    keep_aggregation = any(
        w in normalized
        for w in [
            "aggregation",
            "agregation",
            "repartition",
            "ventilation",
            "par site",
            "par bu",
            "par societe",
            "par source",
            "par activite",
            "par region",
            "par campagne",
        ]
    )

    selected: list[RSEAnalysisTable] = []
    for table in tables:
        name = table.name.lower()
        if name.startswith("top categories") and keep_top:
            selected.append(table)
        elif name.startswith("bottom categories") and keep_bottom:
            selected.append(table)
        elif name.startswith("ecarts") and keep_gap:
            selected.append(table)
        elif name.startswith("aggregation") and (
            keep_aggregation or not (keep_top or keep_bottom or keep_gap)
        ):
            selected.append(table)

    if not selected:
        for table in tables:
            if table.name.lower().startswith("aggregation"):
                selected.append(table)
                break

    return selected


def prepare_rse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the RSE energy dataframe before analysis."""
    required_columns = [
        "Campagne",
        "Site",
        "Societe",
        "BU",
        "Activite",
        "Region",
        "Source_energie",
        "Unite",
        "Conso_Unite",
        "Conso_TEP",
        "Conso_DH",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError("Colonnes manquantes pour RSE Analyst: " + ", ".join(missing))

    clean_df = df.copy()
    for column in ["Conso_Unite", "Conso_TEP", "Conso_DH"]:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce").fillna(0)
    for column in ["Campagne", "Site", "Societe", "BU", "Activite", "Region", "Source_energie", "Unite"]:
        clean_df[column] = clean_df[column].fillna("Non renseigne").astype(str).str.strip()

    clean_df["Campagne_Order"] = clean_df["Campagne"].map(_campaign_sort_key)
    clean_df.sort_values(["Campagne_Order", "Campagne"], inplace=True)
    return clean_df


def apply_query_filters(df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, list[str]]:
    """Apply explicit, auditable filters requested by the user."""
    normalized = _normalize(query)
    filtered = df.copy()
    warnings: list[str] = []

    energy_aliases = {
        "ELECTRICITE": ["electrique", "electricite", "electric"],
        "GASOIL": ["gasoil", "diesel"],
        "FUEL": ["fuel"],
        "ESSENCE": ["essence"],
        "AUTRES": ["autres", "butane", "propane", "bois"],
    }
    for source, keywords in energy_aliases.items():
        if any(keyword in normalized for keyword in keywords):
            source_filtered = filtered[filtered["Source_energie"].str.upper() == source]
            if source_filtered.empty:
                warnings.append(f"Aucune ligne trouvee pour la source d'energie {source}.")
            else:
                filtered = source_filtered
            break

    return filtered, warnings


def infer_metric(query: str, df: pd.DataFrame) -> str:
    normalized = _normalize(query)
    for keyword, column in METRIC_COLUMNS.items():
        if keyword in normalized and column in df.columns:
            return column
    return "Conso_TEP"


def infer_dimension(query: str, default: str = "Source_energie") -> str:
    normalized = _normalize(query)
    scores = {}
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        scores[dimension] = sum(1 for keyword in keywords if keyword in normalized)
    best_dimension = max(scores, key=scores.get)
    if scores[best_dimension] == 0:
        return default
    return best_dimension


def build_analysis_tables(
    df: pd.DataFrame,
    query: str = "",
    metric: str = "Conso_TEP",
    dimension: str = "Source_energie",
) -> list[RSEAnalysisTable]:
    """Create the standard reusable analysis tables requested for RSE outputs."""
    latest_campaign = _latest_campaign(df)
    previous_campaign = _previous_campaign(df)

    tables = [
        RSEAnalysisTable(
            f"Aggregation par {dimension}",
            build_aggregation_table(df, dimension=dimension, metric=metric),
            "Repartition des consommations par dimension analysee.",
        ),
        RSEAnalysisTable(
            "Top categories",
            build_top_bottom_table(df, dimension=dimension, metric=metric, top=True),
            "Categories les plus contributrices.",
        ),
        RSEAnalysisTable(
            "Bottom categories",
            build_top_bottom_table(df, dimension=dimension, metric=metric, top=False),
            "Categories les moins contributrices.",
        ),
    ]

    if latest_campaign and previous_campaign and dimension != "Campagne":
        tables.append(
            RSEAnalysisTable(
                f"Ecarts {previous_campaign} vs {latest_campaign}",
                build_gap_table(df, dimension=dimension, metric=metric),
                "Comparaison entre les deux dernieres campagnes.",
            )
        )

    return tables


def build_kpi_table(
    df: pd.DataFrame,
    latest_campaign: str | None = None,
    previous_campaign: str | None = None,
) -> pd.DataFrame:
    latest_df = df[df["Campagne"] == latest_campaign] if latest_campaign else df
    previous_df = df[df["Campagne"] == previous_campaign] if previous_campaign else pd.DataFrame(columns=df.columns)

    latest_tep = float(latest_df["Conso_TEP"].sum())
    previous_tep = float(previous_df["Conso_TEP"].sum()) if not previous_df.empty else np.nan
    latest_cost = float(latest_df["Conso_DH"].sum())
    previous_cost = float(previous_df["Conso_DH"].sum()) if not previous_df.empty else np.nan

    source_totals = latest_df.groupby("Source_energie", dropna=False)["Conso_TEP"].sum().sort_values(ascending=False)
    top_source = source_totals.index[0] if not source_totals.empty else "Non disponible"

    rows = [
        ("Campagne analysee", latest_campaign or "Toutes", ""),
        ("Consommation totale TEP", latest_tep, "TEP"),
        ("Cout total", latest_cost, "DH"),
        ("Nombre de sites", latest_df["Site"].nunique(), "sites"),
        ("Nombre de BU", latest_df["BU"].nunique(), "BU"),
        ("Source principale", top_source, ""),
    ]
    if previous_campaign:
        rows.extend(
            [
                ("Campagne precedente", previous_campaign, ""),
                ("Evolution TEP", latest_tep - previous_tep, "TEP"),
                ("Evolution TEP %", _safe_pct_change(latest_tep, previous_tep), "%"),
                ("Evolution cout %", _safe_pct_change(latest_cost, previous_cost), "%"),
            ]
        )

    table = pd.DataFrame(rows, columns=["Indicateur", "Valeur", "Unite"])
    return _format_numeric_columns(table)


def build_evolution_table(df: pd.DataFrame, metric: str = "Conso_TEP") -> pd.DataFrame:
    table = (
        df.groupby(["Campagne", "Campagne_Order"], dropna=False)
        .agg(Conso_TEP=("Conso_TEP", "sum"), Conso_DH=("Conso_DH", "sum"), Conso_Unite=("Conso_Unite", "sum"))
        .reset_index()
        .sort_values(["Campagne_Order", "Campagne"])
    )
    table[f"Variation_{metric}"] = table[metric].diff()
    table[f"Variation_{metric}_Pct"] = table[metric].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    table.drop(columns=["Campagne_Order"], inplace=True)
    return _format_numeric_columns(table)


def build_aggregation_table(df: pd.DataFrame, dimension: str, metric: str = "Conso_TEP", limit: int = 30) -> pd.DataFrame:
    dimension = dimension if dimension in df.columns else "Source_energie"
    total_metric = df[metric].sum()
    table = (
        df.groupby(dimension, dropna=False)
        .agg(Conso_TEP=("Conso_TEP", "sum"), Conso_DH=("Conso_DH", "sum"), Conso_Unite=("Conso_Unite", "sum"), Lignes=("Site", "size"))
        .reset_index()
        .sort_values(metric, ascending=False)
    )
    table[f"Part_{metric}_Pct"] = np.where(total_metric == 0, 0, table[metric] / total_metric * 100)
    return _format_numeric_columns(table.head(limit))


def build_top_bottom_table(
    df: pd.DataFrame,
    dimension: str,
    metric: str = "Conso_TEP",
    top: bool = True,
    limit: int = 10,
) -> pd.DataFrame:
    table = build_aggregation_table(df, dimension=dimension, metric=metric, limit=max(limit * 3, 30))
    table = table.sort_values(metric, ascending=not top).head(limit)
    return _format_numeric_columns(table)


def build_gap_table(df: pd.DataFrame, dimension: str, metric: str = "Conso_TEP", limit: int = 20) -> pd.DataFrame:
    dimension = dimension if dimension in df.columns else "Source_energie"
    latest = _latest_campaign(df)
    previous = _previous_campaign(df)
    if not latest or not previous:
        return pd.DataFrame(columns=[dimension, previous or "Precedent", latest or "Dernier", "Ecart", "Ecart_Pct"])

    pivot = (
        df[df["Campagne"].isin([previous, latest])]
        .pivot_table(index=dimension, columns="Campagne", values=metric, aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for campaign in [previous, latest]:
        if campaign not in pivot.columns:
            pivot[campaign] = 0
    pivot["Ecart"] = pivot[latest] - pivot[previous]
    pivot["Ecart_Pct"] = pivot.apply(lambda row: _safe_pct_change(row[latest], row[previous]), axis=1)
    pivot["Abs_Ecart"] = pivot["Ecart"].abs()
    pivot.sort_values("Abs_Ecart", ascending=False, inplace=True)
    pivot.drop(columns=["Abs_Ecart"], inplace=True)
    return _format_numeric_columns(pivot.head(limit))


def build_executive_summary(df: pd.DataFrame, tables: Iterable[RSEAnalysisTable], metric: str = "Conso_TEP") -> str:
    latest = _latest_campaign(df)
    previous = _previous_campaign(df)
    latest_df = df[df["Campagne"] == latest] if latest else df
    previous_df = df[df["Campagne"] == previous] if previous else pd.DataFrame(columns=df.columns)

    latest_total = float(latest_df[metric].sum())
    previous_total = float(previous_df[metric].sum()) if not previous_df.empty else np.nan
    change_pct = _safe_pct_change(latest_total, previous_total)

    source_table = build_aggregation_table(latest_df, "Source_energie", metric=metric, limit=3)
    site_table = build_aggregation_table(latest_df, "Site", metric=metric, limit=3)
    top_source = source_table.iloc[0]["Source_energie"] if not source_table.empty else "non disponible"
    top_site = site_table.iloc[0]["Site"] if not site_table.empty else "non disponible"

    trend_sentence = ""
    if previous:
        direction = "en hausse" if change_pct > 0 else "en baisse" if change_pct < 0 else "stable"
        trend_sentence = f" Par rapport a {previous}, l'indicateur est {direction} de {_format_number(change_pct)}%."

    return (
        f"Analyse RSE energie sur {df['Campagne'].nunique()} campagnes et {df['Site'].nunique()} sites. "
        f"La derniere campagne disponible est {latest}; le total {metric} atteint {_format_number(latest_total)}."
        f"{trend_sentence} La source principale est {top_source}, et le site le plus contributeur est {top_site}. "
        "Les tableaux fournis structurent les KPI, les evolutions, les aggregations et les principaux ecarts."
    )


def _png_bytes_from_fig(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes without touching the filesystem."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_relevant_charts(
    df: pd.DataFrame,
    query: str = "",
    metric: str = "Conso_TEP",
    dimension: str = "Source_energie",
) -> list[RSEChart]:
    """Generate charts in memory (PNG bytes) so nothing is persisted to disk."""
    charts: list[RSEChart] = []
    normalized = _normalize(query)
    chart_kind = "bar" if any(word in normalized for word in ["barre", "barres", "histogramme", "bar chart"]) else "line"
    wants_evolution = any(word in normalized for word in ["evolution", "tendance", "campagne", "annee"])
    wants_single_chart = classify_rse_request(query) == "chart"

    evolution = build_evolution_table(df, metric=metric)
    if not evolution.empty and (wants_evolution or dimension == "Campagne" or not wants_single_chart):
        if wants_single_chart:
            svg_text = _build_svg_evolution_chart(evolution, metric=metric, chart_kind=chart_kind)
            charts.append(
                RSEChart(
                    name="Evolution par campagne",
                    data=svg_text.encode("utf-8"),
                    description=f"Evolution de {metric}.",
                    mime_type="image/svg+xml",
                    file_name="evolution_campagne.svg",
                )
            )
            return charts

        plt = _load_matplotlib()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        if chart_kind == "bar":
            ax.bar(evolution["Campagne"], evolution[metric], color="#1f7a5f")
        else:
            ax.plot(evolution["Campagne"], evolution[metric], marker="o", linewidth=2.5, color="#1f7a5f")
        ax.set_title(f"Evolution {metric} par campagne")
        ax.set_xlabel("Campagne")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        png_bytes = _png_bytes_from_fig(fig)
        plt.close(fig)
        charts.append(
            RSEChart(
                name="Evolution par campagne",
                data=png_bytes,
                description=f"Courbe d'evolution de {metric}.",
                mime_type="image/png",
                file_name="evolution_campagne.png",
            )
        )

    plt = _load_matplotlib()
    chart_dimension = dimension if dimension != "Campagne" else "Source_energie"
    aggregation = build_aggregation_table(df, chart_dimension, metric=metric, limit=10)
    if not aggregation.empty:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        sorted_aggregation = aggregation.sort_values(metric, ascending=True)
        ax.barh(sorted_aggregation[chart_dimension], sorted_aggregation[metric], color="#2b6cb0")
        ax.set_title(f"Top {chart_dimension} par {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel(chart_dimension)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        png_bytes = _png_bytes_from_fig(fig)
        plt.close(fig)
        charts.append(
            RSEChart(
                name=f"Aggregation par {chart_dimension}",
                data=png_bytes,
                description=f"Classement des categories par {metric}.",
                mime_type="image/png",
                file_name=f"aggregation_{_safe_filename(chart_dimension)}.png",
            )
        )

    latest = _latest_campaign(df)
    latest_df = df[df["Campagne"] == latest] if latest else df
    source_table = build_aggregation_table(latest_df, "Source_energie", metric=metric, limit=8)
    if len(source_table) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5.2))
        ax.pie(source_table[metric], labels=source_table["Source_energie"], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Repartition des sources - {latest}")
        fig.tight_layout()
        png_bytes = _png_bytes_from_fig(fig)
        plt.close(fig)
        charts.append(
            RSEChart(
                name="Repartition des sources",
                data=png_bytes,
                description="Part des sources d'energie sur la derniere campagne.",
                mime_type="image/png",
                file_name="repartition_sources.png",
            )
        )

    return charts


def _build_svg_evolution_chart(evolution: pd.DataFrame, metric: str, chart_kind: str = "line") -> str:
    labels = [str(value) for value in evolution["Campagne"].tolist()]
    values = [float(value) if not pd.isna(value) else 0.0 for value in evolution[metric].tolist()]
    width = 920
    height = 460
    margin_left = 74
    margin_right = 34
    margin_top = 54
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 0
    y_max = max_value * 1.15 if max_value > 0 else 1

    def x_pos(index: int) -> float:
        if len(values) <= 1:
            return margin_left + plot_width / 2
        return margin_left + index * plot_width / (len(values) - 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_height - (value / y_max * plot_height)

    y_ticks = []
    for tick in range(5):
        value = y_max * tick / 4
        y = y_pos(value)
        y_ticks.append(
            f"<line x1='{margin_left}' y1='{y:.1f}' x2='{width - margin_right}' y2='{y:.1f}' stroke='#e5e7eb' />"
            f"<text x='{margin_left - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='#475569'>{_format_number(value)}</text>"
        )

    x_labels = []
    for index, label in enumerate(labels):
        x = x_pos(index)
        x_labels.append(
            f"<text x='{x:.1f}' y='{height - 34}' text-anchor='middle' font-size='12' fill='#334155'>{escape(label)}</text>"
        )

    chart_markup = []
    if chart_kind == "bar":
        bar_width = min(62, plot_width / max(len(values), 1) * 0.58)
        for index, value in enumerate(values):
            x = x_pos(index) - bar_width / 2
            y = y_pos(value)
            bar_height = margin_top + plot_height - y
            chart_markup.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' rx='3' fill='#1f7a5f' />"
                f"<text x='{x + bar_width / 2:.1f}' y='{max(y - 8, 18):.1f}' text-anchor='middle' font-size='12' fill='#173f2d'>{_format_number(value)}</text>"
            )
    else:
        points = " ".join(f"{x_pos(index):.1f},{y_pos(value):.1f}" for index, value in enumerate(values))
        chart_markup.append(f"<polyline points='{points}' fill='none' stroke='#1f7a5f' stroke-width='3' />")
        for index, value in enumerate(values):
            chart_markup.append(
                f"<circle cx='{x_pos(index):.1f}' cy='{y_pos(value):.1f}' r='5' fill='#1f7a5f' />"
                f"<text x='{x_pos(index):.1f}' y='{max(y_pos(value) - 10, 18):.1f}' text-anchor='middle' font-size='12' fill='#173f2d'>{_format_number(value)}</text>"
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{margin_left}" y="30" font-size="20" font-weight="700" fill="#173f2d">Evolution {escape(metric)} par campagne</text>
{''.join(y_ticks)}
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#94a3b8" />
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#94a3b8" />
{''.join(chart_markup)}
{''.join(x_labels)}
<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#64748b">Campagne</text>
<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-size="12" fill="#64748b">{escape(metric)}</text>
</svg>"""
    return svg


def build_pdf_report(
    df: pd.DataFrame,
    query: str,
    summary: str,
    tables: list[RSEAnalysisTable],
    charts: list[RSEChart],
) -> bytes:
    """Assemble a PDF report in memory and return its raw bytes (no file written)."""
    plt = _load_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.image import imread

    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.05, 0.95, "Rapport RSE Analyst", fontsize=22, fontweight="bold", color="#1f4f3a")
        ax.text(0.05, 0.90, f"Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')}", fontsize=10, color="#555555")
        ax.text(0.05, 0.84, "Demande utilisateur", fontsize=13, fontweight="bold")
        ax.text(0.05, 0.80, _wrap_for_pdf(query, width=95), fontsize=10, va="top")
        ax.text(0.05, 0.63, "Synthese executive", fontsize=13, fontweight="bold")
        ax.text(0.05, 0.59, _wrap_for_pdf(summary, width=95), fontsize=10, va="top", linespacing=1.35)
        ax.text(
            0.05,
            0.18,
            "Perimetre: donnees energie consolidees par campagne, site, BU, activite, region et source.",
            fontsize=9,
            color="#666666",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for table in tables[:6]:
            _add_table_page_to_pdf(pdf, plt, table)

        for chart in charts:
            # PDF embedding works only with raster images; SVG charts are skipped.
            if not chart.data or chart.mime_type == "image/svg+xml":
                continue
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(chart.name, fontsize=16, fontweight="bold", pad=18)
            image = imread(io.BytesIO(chart.data), format="png")
            ax.imshow(image)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _add_table_page_to_pdf(pdf, plt, table: RSEAnalysisTable) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(table.name, fontsize=15, fontweight="bold", pad=14)

    display_df = table.dataframe.head(18).copy().astype(str)
    display_df = display_df.apply(lambda column: column.str.slice(0, 45))
    mpl_table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    mpl_table.scale(1, 1.35)

    for (row, col), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dfeee7")
            cell.set_text_props(weight="bold", color="#173f2d")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f7f9f8")
        cell.set_edgecolor("#cfd8d3")

    if table.description:
        ax.text(0.01, 0.02, table.description, fontsize=8, color="#666666", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _normalize(value: str) -> str:
    normalized = str(value or "").lower()
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ù": "u",
        "û": "u",
        "ç": "c",
    }
    for accented, ascii_char in replacements.items():
        normalized = normalized.replace(accented, ascii_char)
    return normalized


def _campaign_sort_key(value: str) -> float:
    text = str(value)
    match = re.search(r"(\d{2})\s*/\s*(\d{2})", text)
    if match:
        start, end = match.groups()
        return float(f"20{end}")
    numeric = re.search(r"\d{4}", text)
    if numeric:
        return float(numeric.group(0))
    return 0.0


def _latest_campaign(df: pd.DataFrame) -> str | None:
    campaigns = df[["Campagne", "Campagne_Order"]].drop_duplicates().sort_values(["Campagne_Order", "Campagne"])
    if campaigns.empty:
        return None
    return str(campaigns.iloc[-1]["Campagne"])


def _previous_campaign(df: pd.DataFrame) -> str | None:
    campaigns = df[["Campagne", "Campagne_Order"]].drop_duplicates().sort_values(["Campagne_Order", "Campagne"])
    if len(campaigns) < 2:
        return None
    return str(campaigns.iloc[-2]["Campagne"])


def _safe_pct_change(current: float, previous: float) -> float:
    if pd.isna(previous) or previous == 0:
        return np.nan
    return (current - previous) / previous * 100


def _format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].round(2)
    return formatted


def _format_number(value) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (np.floating, float)):
        return f"{float(value):,.2f}".replace(",", " ")
    return str(value)


def _safe_filename(value: str) -> str:
    normalized = _normalize(value)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "chart"


def _wrap_for_pdf(value: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
