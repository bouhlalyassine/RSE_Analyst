#  .venv/Scripts/Activate.ps1
#  python -m streamlit run app.py
#  git add .    # git commit -m "Màj"   # git push -u origin master

import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie

try:
    from pandasai import Agent
except ImportError:
    Agent = None

from Settings import require_authenticated_user, load_lottiefile, lottie_robot
from Settings import (
    RSE_ANALYST_MAX_TOKENS,
    RSE_ANALYST_SYSTEM_PROMPT,
    RSE_ANALYST_TEMPERATURE,
    build_rse_analyst_llms,
    load_rse_analyst_energy_data,
    reformule_site,
    render_rse_analyst_answer,
    run_structured_rse_analyst,
    should_run_structured_rse_analyst,
    the_StreamlitResponse,
)


st.set_page_config(layout="wide")

authenticator, name, username = require_authenticated_user()

with st.sidebar:
    st.markdown(
        "<h1 style='text-align:center; font-size:34px; font-weight:bold; padding:0rem 0px 0.15rem; margin-bottom:0.15rem'>RSE Analyst</h1>",
        unsafe_allow_html=True)
    
    
    st.markdown("<div class='thick-divider'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    get_answer_toggle = st.toggle("Activate Agent", key="Data_Agent_Toggle", width="stretch")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st_lottie(load_lottiefile(lottie_robot), speed=1, reverse=False, loop=True, quality="high", height=200)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

if "AI_Answer" not in st.session_state:
    st.session_state.AI_Answer = None
if "query" not in st.session_state:
    st.session_state.query = None

if get_answer_toggle:
    with st.spinner("Chargement des donnees Energie..."):
        try:
            main_energie_df, sbar_table = load_rse_analyst_energy_data()
        except Exception as exc:
            st.error("Impossible de charger les donnees Energie pour RSE Analyst.")
            st.exception(exc)
            st.stop()

    col_query, col_button = st.columns([80, 20], gap="small", vertical_alignment="center")
    with col_query:
        query = st.text_area(
            label=" ",
            placeholder="Posez vos questions ici...",
            label_visibility="collapsed",
            disabled=not get_answer_toggle,
            key="Data_Agent_Key",
        )
    with col_button:
        answer_button = st.button("Get Answer")

    with st.expander("Data Overview"):
        main_energie_df_show = main_energie_df.drop(columns=["id"])
        col_columns, col_tables = st.columns([23, 77], gap="medium", vertical_alignment="center")

        with col_columns:
            st.markdown(
                "<h1 style='text-align:center; font-size:20px; font-weight:bold; color:#000000; padding:0rem 0px 0rem'>"
                "BdD - Liste Colonnes</h1>",
                unsafe_allow_html=True,
            )
            df_styled = pd.DataFrame({"Colonnes": main_energie_df_show.columns})

            def style_columns(row):
                return [
                    "background-color:#b4f4b3" if row["Colonnes"] == "Campagne" else
                    "background-color:#bdc2fe" if row["Colonnes"] in ["BU", "Site", "Societe", "Region", "Activite"] else
                    "background-color:#eef0ad" if row["Colonnes"] in ["Source_energie", "Unite"] else
                    ""
                    for _ in row
                ]

            st.dataframe(df_styled.style.apply(style_columns, axis=1), hide_index=True, width="stretch", height=350)

        with col_tables:
            with st.container(border=True):
                col_campaigns, col_sites, col_sources = st.columns([20, 55, 25], gap="medium")

                with col_campaigns:
                    st.markdown(
                        "<h1 style='text-align:center; font-size:20px; font-weight:bold; color:#177915; padding:0rem 0px 0rem'>"
                        "Campagnes</h1>",
                        unsafe_allow_html=True,
                    )
                    table_campaigns = main_energie_df_show.drop(
                        columns=[
                            "Source_energie",
                            "BU",
                            "Societe",
                            "Site",
                            "Activite",
                            "Region",
                            "Unite",
                            "Conso_Unite",
                            "Conso_TEP",
                            "Conso_DH",
                        ]
                    )
                    table_campaigns.sort_values(["Campagne"], ascending=True, inplace=True)
                    table_campaigns = table_campaigns.drop_duplicates()
                    st.dataframe(table_campaigns.style.applymap(lambda _: "background-color:#b4f4b3"), width="stretch", hide_index=True)

                with col_sites:
                    st.markdown(
                        "<h1 style='text-align:center; font-size:20px; font-weight:bold; color:#192187; padding:0rem 0px 0rem'>Sites</h1>",
                        unsafe_allow_html=True,
                    )
                    sbar_table_show = sbar_table.copy()
                    sbar_table_show["Site"].replace(
                        {"LES DOMAINES AGRICOLES": "SIEGE LDA", "CLUB HOUSE RGM": "ROYAL GOLF MARRAKECH"},
                        inplace=True,
                    )
                    sbar_table_show = reformule_site(sbar_table_show).drop_duplicates().copy()
                    sbar_table_show.sort_values(["Site"], ascending=True, inplace=True)
                    st.dataframe(sbar_table_show.style.applymap(lambda _: "background-color:#bdc2fe"), width="stretch", hide_index=True, height=350)

                with col_sources:
                    st.markdown(
                        "<h1 style='text-align:center; font-size:20px; font-weight:bold; color:#a3a632; padding:0rem 0px 0rem'>"
                        "Sources Energie</h1>",
                        unsafe_allow_html=True,
                    )
                    table_type = main_energie_df_show.drop(
                        columns=[
                            "Campagne",
                            "BU",
                            "Societe",
                            "Site",
                            "Activite",
                            "Region",
                            "Conso_Unite",
                            "Conso_TEP",
                            "Conso_DH",
                        ]
                    )
                    table_type.sort_values(["Source_energie"], ascending=True, inplace=True)
                    table_type = table_type.drop_duplicates()
                    st.dataframe(table_type.style.applymap(lambda _: "background-color:#eef0ad"), width="stretch", hide_index=True)

    # Always-visible Reponse zone (rendered as a placeholder so the expander
    # stays in the layout even before any query has been submitted).
    response_zone = st.container()

    if len(query) < 7 and len(query) != 0:
        st.warning("Votre requete est trop courte")
    elif answer_button:
        with st.spinner("Reponse..."):
            if st.session_state.query != query or st.session_state.AI_Answer is None:
                st.session_state.RSE_Analyst_Used_Structured = False

                if should_run_structured_rse_analyst(query):
                    try:
                        # Charts and PDF reports are generated entirely in memory
                        # by the engine, so no exports directory is needed.
                        st.session_state.AI_Answer = run_structured_rse_analyst(
                            main_energie_df,
                            query,
                        )
                        st.session_state.RSE_Analyst_LLM = "Moteur pandas local"
                        st.session_state.RSE_Analyst_Used_Structured = True
                    except Exception as exc:
                        st.error("RSE Analyst n'a pas pu produire l'analyse structuree.")
                        st.exception(exc)
                        st.stop()
                else:
                    if Agent is None:
                        try:
                            st.session_state.AI_Answer = run_structured_rse_analyst(
                                main_energie_df,
                                query,
                            )
                            st.session_state.RSE_Analyst_LLM = "Moteur pandas local"
                            st.session_state.RSE_Analyst_Used_Structured = True
                        except Exception as exc:
                            st.error("RSE Analyst n'a pas pu produire l'analyse structuree.")
                            st.exception(exc)
                            st.stop()
                    else:
                        llm_candidates = build_rse_analyst_llms(RSE_ANALYST_TEMPERATURE, RSE_ANALYST_MAX_TOKENS)
                        if not llm_candidates:
                            st.error("Aucune cle API disponible pour RSE Analyst. Ajoutez GROQ_API_KEY ou MISTRAL_API_KEY dans les secrets Streamlit.")
                            st.stop()

                        last_error = None
                        for llm_name, llm in llm_candidates:
                            try:
                                st.session_state.agent = Agent(
                                    [main_energie_df],
                                    config={
                                        "llm": llm,
                                        "save_logs": False,
                                        "save_charts": True,
                                        "enable_cache": False,
                                        "max_retries": 3,
                                        "response_parser": the_StreamlitResponse,
                                    },
                                    memory_size=10,
                                )
                                st.session_state.AI_Answer = st.session_state.agent.chat(RSE_ANALYST_SYSTEM_PROMPT + query)
                                st.session_state.RSE_Analyst_LLM = llm_name
                                last_error = None
                                break
                            except Exception as exc:
                                last_error = exc

                        if last_error is not None:
                            st.error("RSE Analyst n'a pas pu generer de reponse avec les modeles disponibles.")
                            st.exception(last_error)
                            st.stop()

        st.session_state.query = query

    # Render the Reponse expander every run (placeholder before the first query,
    # full content after). This keeps the response zone visually anchored.
    with response_zone:
        with st.expander("Reponse", expanded=True):
            if st.session_state.AI_Answer is None:
                st.info(
                    "Posez une question puis cliquez sur **Get Answer** : "
                    "la reponse de l'agent (graphique, tableau ou rapport) "
                    "s'affichera dans cette zone."
                )
            else:
                render_rse_analyst_answer(st.session_state.AI_Answer)
else:
    st.markdown(
        "<h1 style='text-align:center; font-size:30px; font-weight:bold'>RSE Analyst</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='thick-divider'></div>", unsafe_allow_html=True)
    st.markdown("""<style>div[data-testid="stMarkdownContainer"] {text-align: left}</style>""", unsafe_allow_html=True)
    st.write(
        """
        <div style="background-color:#ddeee3; color:#167232; padding:15px; border-radius:5px; margin-bottom:15px">
            <span style="font-size:18px;"><u><b>Presentation :</b></u></span>
            <ul style="list-style-type:none; margin:0;">
                <li><span style="font-size:17px;"><b>RSE Analyst est un agent IA specialise dans l'analyse de donnees Energie, capable d'effectuer des calculs complexes et de generer des graphiques.</b></span></li>
                <li><span style="font-size:17px;"><i>Modele Groq : llama-3.3-70b-versatile</i></span></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        <div style="background-color:#dce7f0; color:#014380; padding:15px; border-radius:5px; margin-bottom:15px">
            <span style="font-size:18px;"><u><b>Tester RSE Analyst :</b></u></span>
            <ul style="list-style-type:none; margin:0;">
                <li><span style="font-size:17px;">1 - Activer le toggle button "Activate Agent"</span></li>
                <li><span style="font-size:17px;">2 - Ouvrir "Data Overview" pour visualiser les donnees traitees</span></li>
                <li><span style="font-size:17px;">3 - Poser une question en conservant l'orthographe des parametres affiches</span></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        <div style="background-color:#fedfd7; color:#9c0000; padding:15px; border-radius:5px; margin-bottom:15px">
            <span style="font-size:18px;"><u><b>Remarque :</b></u> Uniquement la BdD Energie est fournie a l'IA dans cette version beta.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    _, col = st.columns([33, 67], vertical_alignment="center")
    with col:
        authenticator.logout("Logout")
