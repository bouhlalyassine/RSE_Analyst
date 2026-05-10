import base64
import io
import json
import os
from pathlib import Path
import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None

try:
    from pandasai.responses.response_parser import ResponseParser
except ImportError:
    class ResponseParser:
        def __init__(self, context=None) -> None:
            self.context = context

from rse_analyst_engine import (
    RSEAnalysisResult,
    analyze_rse_data,
    should_use_structured_engine,
)


current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
TITLE = "Taskforce IA"

config_APP = current_dir / "files" / "hash_APP.yaml"
css_file = current_dir / "main.css"

img_logo_name_ico = current_dir / "files" / "logo_name.png"
img_logo_ico = str(img_logo_name_ico) if img_logo_name_ico.exists() else None
lottie_warning = current_dir / "files" / "warning.json"
lottie_robot = current_dir / "files" / "AI_Robot.json"


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_img(local_img_path, width):
    local_img_path = str(local_img_path)
    img_format = os.path.splitext(local_img_path)[-1].replace(".", "").lower()
    bin_str = get_base64_of_bin_file(local_img_path)
    return f"""
        <div style='display:flex; justify-content:center; align-items:center;'>
            <img src='data:image/{img_format};base64,{bin_str}' width='{width}'>
        </div>
    """


def load_lottiefile(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lottieurl(url):
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return None
    return response.json()


def load_css():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_auth_config():
    with open(config_APP, encoding="utf-8") as file:
        return yaml.load(file, Loader=SafeLoader)


def build_authenticator(config):
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )


def require_authenticated_user():
    config = load_auth_config()
    authenticator = build_authenticator(config)
    load_css()

    name, authentication_status, username = authenticator.login("Login", "main")
    users = config["credentials"]["usernames"]

    if authentication_status is False:
        st.error("Username/password is incorrect")
        st.stop()

    if authentication_status is None:
        st.warning("Please enter your username and password")
        st.stop()

    if username not in users:
        st.warning("Vous n'avez pas acces a ce module")
        st.stop()

    return authenticator, name, username




RSE_ANALYST_GROQ_MODEL = "llama-3.3-70b-versatile"
RSE_ANALYST_TRANSLATION_MODEL = "llama-3.1-8b-instant"
RSE_ANALYST_MAX_TOKENS = 1200
RSE_ANALYST_TEMPERATURE = 0
THE_TTL = 84600

RSE_ANALYST_SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing energy consumption data.
The dataset contains energy usage fields across multiple sites.

Columns:
- Campagne: campaign period, typically running from October to September.
- Site: specific site.
- Societe: company that may correspond to a site or a group of sites.
- BU: Business Unit.
- Activite: site activity.
- Region: region where the site is located.
- Source_energie: type of energy source used.
- Unite: unit of measurement for the energy source.
- Conso_Unite: consumption value expressed in the source unit.
- Conso_TEP: consumption value expressed in TEP.
- Conso_DH: cost of consumption in Moroccan Dirhams.

Respond with precise, well-structured answers.

Here is the user question:
"""


class the_StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])

    def format_plot(self, result):
        st.image(result["value"])

    def format_other(self, result):
        st.write(result["value"])


@st.cache_resource(ttl=THE_TTL)
def load_conn(g_type):
    conn = st.connection("gsheets", type=g_type)
    now = datetime.now() + timedelta(hours=1)
    return conn, now.strftime("%d/%m/%Y")


def get_optional_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


def _col(df, ascii_name, accented_name=None):
    if ascii_name in df.columns:
        return ascii_name
    if accented_name and accented_name in df.columns:
        return accented_name
    return ascii_name


def _factor_value(factors_df, factor_name):
    match = factors_df.loc[factors_df["Facteur"] == factor_name, "Valeur"]
    if match.empty:
        raise KeyError(f"Facteur d'emission introuvable: {factor_name}")
    return float(match.iloc[0])


def reformule_site(df):
    return df


@st.cache_data
def MAIN_conso_energie_df(main_df, sbar_df, factors_df):
    df = pd.merge(main_df, sbar_df, on=["Site"], how="left", suffixes=("", "_y"))
    unit_col = _col(df, "Unite", "Unité")
    activity_col = _col(df, "Activite", "Activité")
    total_unit_col = _col(df, "TOTAL_Unite", "TOTAL_Unité")

    for column in ["Campagne", unit_col, "Type", "BU", "Site", activity_col]:
        df[column] = df[column].astype(str)

    df[total_unit_col] = df[total_unit_col].astype(float)
    df["TOTAL_DH"] = df["TOTAL_DH"].astype(float)

    tep_factors = {
        "ELECTRICITE": ("KWh", _factor_value(factors_df, "kwh_to_TEP")),
        "GASOIL": ("L", _factor_value(factors_df, "Gasoil_L_to_TEP")),
        "FUEL": ("kg", _factor_value(factors_df, "Fuel_kg_to_TEP")),
        "ESSENCE": ("L", _factor_value(factors_df, "Ess_to_TEP")),
        "GAZ BUTANE": ("kg", _factor_value(factors_df, "Butane_kg_to_TEP")),
        "GAZ VRAC PROPANE": ("kg", _factor_value(factors_df, "Propane_kg_to_TEP")),
        "BOIS DE CHAUDIERE": ("kg", _factor_value(factors_df, "BChaudi_kg_to_TEP")),
    }

    conditions = []
    values = []
    for energy_type, (unit, factor) in tep_factors.items():
        conditions.append((df["Type"] == energy_type) & (df[unit_col] == unit))
        values.append(df[total_unit_col] * factor)
    df["TOTAL_TEP"] = np.select(conditions, values, default=0)

    df["Site"].replace(
        {"LES DOMAINES AGRICOLES": "SIEGE LDA", "DOMSEEDS": "SIEGE LDA", "NADOR COTT CONCESSION": "SIEGE LDA"},
        inplace=True,
    )
    df["BU"].replace({"SIEGE": "AUTRES", "RGM": "AUTRES"}, inplace=True)
    df[activity_col].replace({"SIEGE": "AUTRES", "RGM": "AUTRES"}, inplace=True)
    df["Type"].replace(
        {"GAZ BUTANE": "AUTRES", "GAZ VRAC PROPANE": "AUTRES", "BOIS DE CHAUDIERE": "AUTRES"},
        inplace=True,
    )
    df.drop(df[df["Site"] == "CLUB HOUSE RGM"].index, inplace=True)

    campagne_mapping = {"18/19": 1, "19/20": 2, "20/21": 3, "21/22": 4, "22/23": 5, "23/24": 6, "24/25": 7}
    df["Campagne_Num"] = df["Campagne"].map(campagne_mapping)
    df = reformule_site(df)
    df.sort_values(["Campagne_Num"], ascending=True, inplace=True)
    return df


@st.cache_data
def Conso_energie_df_AI(main_df):
    df = main_df.copy()
    society_col = _col(df, "Societe", "Société")
    region_col = _col(df, "Region", "Région")
    activity_col = _col(df, "Activite", "Activité")
    unit_col = _col(df, "Unite", "Unité")
    total_unit_col = _col(df, "TOTAL_Unite", "TOTAL_Unité")

    df = df.reindex(
        columns=[
            "Campagne",
            "Site",
            society_col,
            "BU",
            activity_col,
            region_col,
            "Type",
            unit_col,
            total_unit_col,
            "TOTAL_TEP",
            "TOTAL_DH",
        ]
    )

    df.rename(
        columns={
            society_col: "Societe",
            activity_col: "Activite",
            region_col: "Region",
            unit_col: "Unite",
            total_unit_col: "TOTAL_Unite",
        },
        inplace=True,
    )

    for column in ["Campagne", "Unite", "Type", "BU", "Site", "Activite"]:
        df[column] = df[column].astype(str)
    for column in ["TOTAL_Unite", "TOTAL_DH", "TOTAL_TEP"]:
        df[column] = df[column].astype(float)

    df.rename(
        columns={
            "Type": "Source_energie",
            "TOTAL_TEP": "Conso_TEP",
            "TOTAL_DH": "Conso_DH",
            "TOTAL_Unite": "Conso_Unite",
        },
        inplace=True,
    )
    df = reformule_site(df)
    df["id"] = df["Campagne"].astype(str) + df["BU"].astype(str) + df["Site"].astype(str)
    return df


def load_rse_analyst_energy_data():
    conn = load_conn(GSheetsConnection)[0]
    annexe_table = conn.read(worksheet="ANX", usecols=list(range(15)), ttl=THE_TTL)
    sbar_table = annexe_table.iloc[:, 0:6].copy().dropna(how="all")
    factors_table = annexe_table.iloc[:, 6:15].copy().dropna(how="all")

    energy_db = conn.read(worksheet="ENR", usecols=list(range(9)), ttl=THE_TTL)
    energy_db = energy_db.dropna(how="all")

    main_df_conso = MAIN_conso_energie_df(energy_db, sbar_table, factors_table)
    main_energie_df = Conso_energie_df_AI(main_df_conso)
    return main_energie_df, sbar_table


def build_rse_analyst_llms(temp, max_tokens):
    llms = []
    groq_key = get_optional_secret("GROQ_API_KEY")
    if groq_key and ChatGroq is not None:
        llms.append(
            (
                f"Groq / {RSE_ANALYST_GROQ_MODEL}",
                ChatGroq(
                    groq_api_key=groq_key,
                    model_name=RSE_ANALYST_GROQ_MODEL,
                    temperature=temp,
                    max_tokens=max_tokens,
                ),
            )
        )

    mistral_key = get_optional_secret("MISTRAL_API_KEY")
    if mistral_key and ChatMistralAI is not None:
        llms.append(
            (
                "Mistral / mistral-large-latest",
                ChatMistralAI(
                    model="mistral-large-latest",
                    api_key=mistral_key,
                    temperature=temp,
                    max_tokens=max_tokens,
                ),
            )
        )
    return llms


def render_rse_analyst_answer(answer):
    if answer is None:
        return
    # Duck-typing on top of isinstance: when Streamlit hot-reloads this module,
    # the RSEAnalysisResult class is redefined in memory, so a stale instance
    # stored in st.session_state stops matching isinstance(). Falling back to
    # attribute checks keeps the structured renderer running and prevents the
    # raw dataclass introspection from leaking into the UI.
    looks_structured = (
        isinstance(answer, RSEAnalysisResult)
        or (
            hasattr(answer, "output_type")
            and hasattr(answer, "tables")
            and hasattr(answer, "charts")
            and hasattr(answer, "summary")
        )
    )
    if looks_structured:
        render_structured_rse_analyst_answer(answer)
        return
    st.write(answer)


def run_structured_rse_analyst(main_df, query, output_dir=None):
    """Run the structured analyst engine. ``output_dir`` is accepted for
    backward compatibility but ignored: charts and PDF are produced in memory."""
    return analyze_rse_data(main_df, query=query)


def should_run_structured_rse_analyst(query):
    return should_use_structured_engine(query)


def render_structured_rse_analyst_answer(result):
    # PDF report request: a short description, then the download button.
    pdf_data = getattr(result, "pdf_data", None)
    if pdf_data:
        st.markdown(
            "<div style='text-align:left; font-size:15px; font-weight:bold; margin-bottom:0.6rem;'>"
            "Rapport d'analyse genere par RSE Analyst : synthese, indicateurs, "
            "tableaux d'analyse et graphiques regroupes dans un document PDF "
            "pret a partager."
            "</div>",
            unsafe_allow_html=True,
        )

        pdf_filename = getattr(result, "pdf_filename", "rapport_rse_analyst.pdf")
        st.download_button(
            "Telecharger le rapport",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf",
            key=f"download_report_{pdf_filename}",
        )

        for warning in result.warnings:
            st.warning(warning)
        return

    # Summary text only when present (skipped for chart/table-only responses).
    if result.summary:
        st.write(result.summary)

    # Charts: render inline without a section header.
    if result.charts:
        chart_columns = st.columns(min(len(result.charts), 3))
        for index, chart in enumerate(result.charts):
            with chart_columns[index % len(chart_columns)]:
                if chart.mime_type == "image/svg+xml":
                    st.markdown(chart.data.decode("utf-8"), unsafe_allow_html=True)
                else:
                    st.image(chart.data, use_container_width=True)
                st.download_button(
                    f"Telecharger {chart.name}",
                    data=chart.data,
                    file_name=chart.file_name,
                    mime=chart.mime_type,
                    key=f"download_chart_{index}_{chart.file_name}",
                )

    # Tables: render directly without nested expanders or section headers.
    if result.tables:
        for index, table in enumerate(result.tables):
            if table.description:
                st.caption(table.description)
            st.dataframe(table.dataframe, hide_index=True, width="stretch")

            # Build an .xlsx file in-memory so the user gets a real Excel workbook,
            # not a CSV. openpyxl is already pinned in requirements.txt.
            excel_buffer = io.BytesIO()
            sheet_name = (table.name or "Tableau")[:31]  # Excel sheet limit
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                table.dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
            excel_buffer.seek(0)

            st.download_button(
                f"Telecharger {table.name} (Excel)",
                data=excel_buffer,
                file_name=f"{table.name.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_table_{index}_{table.name}",
            )

    for warning in result.warnings:
        st.warning(warning)
