# app/main.py

# Scout Pulse: AI Player Recruitment & Replacement Platform - main
# Streamlit application

import os
import re
import sys

PROJECT_ROOT = os.path.abspath (os.path.join (os.path.dirname (__file__),".."))
APP_DIR = os.path.dirname (os.path.abspath (__file__))
for _path in (PROJECT_ROOT,APP_DIR):
    if _path not in sys.path:
        sys.path.insert (0,_path)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS,inject_theme,sentiment_color
from src.data_loader import load_player_data
from src.recommender import get_player_recommendations
from src.sentiment_engine import analyze_player_sentiments,get_sentiment_breakdown

st.set_page_config (
    page_title = "Scout Pulse AI",
    page_icon = "⚽",
    layout = "wide",
    initial_sidebar_state = "collapsed",
)
inject_theme ()

COMPARE_B_COLOR = "#4FA8E0"
CATEGORIES = ["Pace","Shooting","Passing","Dribbling","Defending","Physical"]

def style_fig (fig,height = None):

    # Applies the Scout Pulse theme to any Plotly figure

    fig.update_layout (
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = dict(family = "Barlow, sans-serif",color = COLORS ["text_secondary"],size = 12),
        margin = dict(l = 10,r = 10,t = 40,b = 10),
        legend = dict(bgcolor = "rgba(0,0,0,0)",font = dict (color = COLORS ["text_secondary"])),
    )
    fig.update_xaxes (gridcolor = COLORS ["border"],zerolinecolor = COLORS ["border"],color = COLORS ["text_secondary"])
    fig.update_yaxes (gridcolor = COLORS ["border"],zerolinecolor = COLORS ["border"],color = COLORS ["text_secondary"])
    if height:
        fig.update_layout (height = height)
    return fig

def render_highlighted_report (text,lexicon_words):

    # Wraps every word VADER's lexicon actually scored with a colored
    # span - green for positive lexicon weight, red for negative - so
    # the scout report becomes a visibly annotated view of exactly what
    # drove its sentiment score. Uses VADER's real per-word values, not
    # an invented explanation

    def replace (match):
        word = match.group (0)
        score = lexicon_words.get (word)
        if score is None:
            return word
        color = COLORS ["sentiment_positive"] if score > 0 else COLORS ["sentiment_negative"]
        return f'<span style="color:{color}; font-weight:700;">{word}</span>'

    return re.sub (r"[A-Za-z']+",replace,text)

def gk_caveat_if_needed (position):

    # This dataset has exactly one goalkeeper. The recommender's
    # position-aware ranking falls back to cross-group matches when no
    # same-group players exist - which, for a keeper, means the "best
    # match" is whichever outfield player happens to be least dissimilar
    # on outfield attributes. Tested directly: Neuer's top match comes
    # back as a CDM. This caveat is shown, not hidden.

    if position == "GK":
        st.caption (
            "⚠️ This dataset includes only one goalkeeper. Goalkeeper-specific "
            "skills (reflexes, distribution, positioning) aren't captured by "
            "these six outfield-oriented attributes and there's no other "
            "keeper to match against - treat any recommendation here as "
            "illustrative only, not a tactically meaningful match.")

@st.cache_data
def load_and_process_data ():

    raw_df = load_player_data ("data/players_2026.csv")
    return analyze_player_sentiments (raw_df)

df = load_and_process_data ()

st.title ("⚽ Scout Pulse AI: Player Recruitment & Replacement Platform")
st.caption ("NLP scouting report sentiment analysis, position aware cosine similarity replacement engine and squad builder")

st.sidebar.header ("🕹️ Navigation Panel")
app_mode = st.sidebar.selectbox (
    "Select View",
    [
        "🚀 Executive Command Center",
        "♻️ AI Replacement & Scout Engine",
        "⚔️ Player Head To Head Comparison",
        "🛡️ Custom Squad Builder & Chemistry",
        "🎯 Scouting Filter & Target Finder",
        "📊 Tactical Radar Cluster & Archetypes",
        "🔥 Hype vs Reality",
        "💰 Market Value & Wage Estimator",
        "📈 Sentiment & Market Matrix",
    ],
    index = 0,)

st.divider ()

if app_mode == "🚀 Executive Command Center":
    col_left,col_mid,col_right = st.columns ([1.0,1.2,1.1])

    with col_left:
        st.subheader ("👤 Select Player Profile")
        selected_player = st.selectbox ("Target Footballer",options = sorted (df ["Player"].unique ()))
        p_data = df [df ["Player"] == selected_player].iloc [0]

        st.markdown (
            f"""
            <div class="player-card">
                <p><b>Club:</b> {p_data ['Current_Club']}</p>
                <p><b>League:</b> {p_data ['League']}</p>
                <p><b>Position / Age:</b> {p_data ['Position']} | {p_data ['Age']} years</p>
                <p><b>Overall Rating:</b> <span style="color:{COLORS ['accent']}; font-weight:bold;">{p_data ['Overall']}</span></p>
            </div>
            """,
            unsafe_allow_html = True,)

        st.subheader ("📝 Scouting Report Summary")
        st.info (f'"{p_data ["Scout_Report"]}"')

        breakdown = get_sentiment_breakdown (p_data ["Scout_Report"])
        with st.expander ("🔍 Why this sentiment score ?? (sentence + keyword breakdown)"):
            st.markdown (
                f'<p style="line-height:1.8;">{render_highlighted_report (p_data ["Scout_Report"],breakdown ["lexicon_words"])}</p>',
                unsafe_allow_html = True,)

            st.markdown ("**Sentence-level scores:**")
            for s in breakdown ["sentences"]:
                st.markdown (
                    f'<p style="margin:0.3rem 0; color:{sentiment_color (s ["compound"])};">[{s ["compound"]:+.2f}] {s ["text"]}</p>',
                    unsafe_allow_html = True,)

            st.caption (
                "Note: general-purpose sentiment lexicons occasionally misjudge "
                "sports-specific language - e.g. \"dominating\" scores negative "
                "in VADER's lexicon, likely from contexts like \"dominating and "
                "controlling.\" Shown transparently rather than hidden."
            )

        scout_txt = (
            f"OFFICIAL SCOUTING DOSSIER\n"
            f"Player: {p_data ['Player']}\n"
            f"Club: {p_data ['Current_Club']} ({p_data ['League']})\n"
            f"Overall: {p_data ['Overall']}\n"
            f"Estimated Market Value: \u20ac{p_data ['Est_Market_Value_M']}M (illustrative estimate)\n"
            f"Scout Report: {p_data ['Scout_Report']}\n"
            f"Sentiment Score: {p_data ['Sentiment_Compound']:.2f} ({p_data ['Sentiment_Label']})"
        )
        st.download_button (
            label = "📄 Export Scouting Dossier (.txt)",
            data = scout_txt,
            file_name = f"{p_data ['Player']}_Scout_Report.txt",
            mime = "text/plain",
            use_container_width = True,)

    with col_mid:
        st.subheader ("📊 Live Performance & NLP Metrics")
        m1,m2,m3 = st.columns (3)
        m1.markdown (
            f'<div class="metric-card"><div class="metric-title">Overall Score</div><div class="metric-value">{p_data ["Overall"]}</div></div>',
            unsafe_allow_html = True,)

        m2.markdown (
            f'<div class="metric-card"><div class="metric-title">Compound Sentiment</div>'
            f'<div class="metric-value" style="color:{sentiment_color (p_data ["Sentiment_Compound"])};">{p_data ["Sentiment_Compound"]:.2f}</div></div>',
            unsafe_allow_html = True,)

        m3.markdown (
            f'<div class="metric-card"><div class="metric-title">Scout Sentiment</div>'
            f'<div class="metric-value" style="font-size: 0.95rem; color:{sentiment_color (p_data ["Sentiment_Compound"])};">{p_data ["Sentiment_Label"]}</div></div>',
            unsafe_allow_html = True,)

        st.markdown ("<div style='margin-bottom: 10px;'></div>",unsafe_allow_html = True)

        st.subheader ("🕸️ Tactical Attribute Radar")
        values = [p_data [cat] for cat in CATEGORIES]
        fig_radar = go.Figure ()
        fig_radar.add_trace (go.Scatterpolar (
            r = values,theta = CATEGORIES,fill = "toself",
            line_color = COLORS ["accent"],fillcolor = "rgba(242, 184, 7, 0.25)",
        ))
        fig_radar.update_layout (showlegend = False)
        st.plotly_chart (style_fig (fig_radar,height = 340),use_container_width = True)

    with col_right:
        st.subheader ("♻️ Instant AI Replacement Match")
        gk_caveat_if_needed (p_data ["Position"])
        top_rec = get_player_recommendations (selected_player,df,top_n = 1).iloc [0]

        st.success (f"Best Match: **{top_rec ['Player']}** ({top_rec ['Match_Similarity_%']} % Match)")
        st.markdown (
            f"""
            <div class="player-card">
                <p><b>Club:</b> {top_rec ['Current_Club']} ({top_rec ['League']})</p>
                <p><b>Position / Age:</b> {top_rec ['Position']} | {top_rec ['Age']} years</p>
                <p style="font-size: 0.85rem;"><b>Scout Note:</b> {top_rec ['Scout_Report']}</p>
            </div>
            """,
            unsafe_allow_html = True,)

        st.subheader ("📊 Sentiment Polarity Split")
        pol_df = pd.DataFrame ({
            "Polarity": ["Positive","Neutral","Negative"],
            "Score": [p_data ["Sentiment_Positive"],p_data ["Sentiment_Neutral"],p_data ["Sentiment_Negative"]],
        })
        fig_pol = px.bar (
            pol_df,x = "Polarity",y = "Score",color = "Polarity",
            color_discrete_map = {
                "Positive": COLORS ["sentiment_positive"],
                "Neutral": COLORS ["sentiment_neutral"],
                "Negative": COLORS ["sentiment_negative"],
            },
        )
        fig_pol.update_layout (showlegend = False)
        st.plotly_chart (style_fig (fig_pol,height = 260),use_container_width = True)

elif app_mode == "♻️ AI Replacement & Scout Engine":
    st.subheader ("♻️ Tactical Replacement Engine (Position Aware Cosine Similarity)")
    target_player = st.selectbox ("Select Player to Replace",options = sorted (df ["Player"].unique ()))
    target_position = df [df ["Player"] == target_player].iloc [0]["Position"]
    gk_caveat_if_needed (target_position)

    num_recs = st.slider ("Number of Candidates",1,5,3)

    recs_df = get_player_recommendations (target_player,df,top_n = num_recs)
    st.success (f"Top {num_recs} tactical matches for **{target_player}**:")

    for i,(_,row) in enumerate(recs_df.iterrows ()):
        with st.expander (
            f"Candidate {i + 1}: {row ['Player']} ({row ['Current_Club']}) \u2014 Match Score: {row ['Match_Similarity_%']} %",
            expanded = True,
        ):
            c1,c2,c3,c4 = st.columns (4)
            c1.write (f"**Position:** {row ['Position']}")
            c2.write (f"**Age:** {row ['Age']}")
            c3.write (f"**League:** {row ['League']}")
            c4.write (f"**Overall Rating:** {row ['Overall']}")
            st.write (f"**Scout Report:** {row ['Scout_Report']}")

elif app_mode == "⚔️ Player Head To Head Comparison":
    st.subheader ("⚔️ Head To Head Player Comparison")

    col1,col2 = st.columns (2)
    with col1:
        player1 = st.selectbox ("Player A",options = sorted (df ["Player"].unique ()),index = 0)
    with col2:
        player2 = st.selectbox ("Player B",options = sorted (df ["Player"].unique ()),index = 1)

    p1 = df [df ["Player"] == player1].iloc [0]
    p2 = df [df ["Player"] == player2].iloc [0]

    m1,m2,m3,m4 = st.columns (4)
    m1.markdown (f'<div class="metric-card"><div class="metric-title">{player1} Rating</div><div class="metric-value">{p1 ["Overall"]}</div></div>',unsafe_allow_html = True)
    m2.markdown (f'<div class="metric-card"><div class="metric-title">{player2} Rating</div><div class="metric-value" style="color:{COMPARE_B_COLOR};">{p2 ["Overall"]}</div></div>',unsafe_allow_html = True)
    m3.markdown (f'<div class="metric-card"><div class="metric-title">{player1} Sentiment</div><div class="metric-value" style="color:{sentiment_color (p1 ["Sentiment_Compound"])};">{p1 ["Sentiment_Compound"]:.2f}</div></div>',unsafe_allow_html = True)
    m4.markdown (f'<div class="metric-card"><div class="metric-title">{player2} Sentiment</div><div class="metric-value" style="color:{sentiment_color (p2 ["Sentiment_Compound"])};">{p2 ["Sentiment_Compound"]:.2f}</div></div>',unsafe_allow_html = True)

    st.markdown ("<div style='margin-bottom: 15px;'></div>",unsafe_allow_html = True)

    col_radar,col_diff = st.columns ([1.1,0.9])

    with col_radar:
        st.subheader ("🕸️ Tactical Attribute Overlay")
        fig_comp = go.Figure ()
        fig_comp.add_trace (go.Scatterpolar (r = [p1 [c] for c in CATEGORIES],theta = CATEGORIES,fill = "toself",name = player1,line_color = COLORS ["accent"]))
        fig_comp.add_trace (go.Scatterpolar (r = [p2 [c] for c in CATEGORIES],theta = CATEGORIES,fill = "toself",name = player2,line_color = COMPARE_B_COLOR))
        fig_comp.update_layout (legend = dict(orientation = "h",y = -0.1,x = 0.2))
        st.plotly_chart (style_fig (fig_comp,height = 420),use_container_width = True)

    with col_diff:
        st.subheader ("📊 Attribute Differential")
        diff_values = [p1 [c] - p2 [c] for c in CATEGORIES]
        diff_df = pd.DataFrame ({"Attribute": CATEGORIES,"Advantage": diff_values})
        diff_df ["Winner"] = np.where (diff_df ["Advantage"] >= 0,player1,player2)

        fig_diff = px.bar (
            diff_df,x = "Advantage",y = "Attribute",orientation = "h",color = "Winner",
            color_discrete_map = {player1: COLORS ["accent"],player2: COMPARE_B_COLOR},
            title = f"Metric Gap ({player1} vs {player2})",
        )
        st.plotly_chart (style_fig (fig_diff,height = 420),use_container_width = True)

    st.subheader ("📋 Statistical Breakdown")
    comp_matrix = pd.DataFrame ({
        "Attribute": CATEGORIES + ["Overall Rating","Compound Sentiment","Age"],
        player1: [p1 [c] for c in CATEGORIES] + [p1 ["Overall"],p1 ["Sentiment_Compound"],p1 ["Age"]],
        player2: [p2 [c] for c in CATEGORIES] + [p2 ["Overall"],p2 ["Sentiment_Compound"],p2 ["Age"]],
    })
    st.dataframe (comp_matrix,use_container_width = True)

elif app_mode == "🛡️ Custom Squad Builder & Chemistry":
    st.subheader ("🛡️ Squad Builder & Chemistry Estimate")
    st.caption("Chemistry is an illustrative composite of league diversity and average scout sentiment - not a trained model")

    squad_players = st.multiselect (
        "Select Players for Your Squad",
        options = sorted (df ["Player"].unique ()),
        default = sorted (df ["Player"].unique ())[:11],)

    if squad_players:
        sq_df = df [df ["Player"].isin (squad_players)]
        avg_rating = sq_df ["Overall"].mean ()
        avg_sentiment = sq_df ["Sentiment_Compound"].mean ()
        leagues = sq_df ["League"].nunique ()
        chemistry_score = min (100,int ((10-leagues)*5 + avg_sentiment*30 + 50))

        m1,m2,m3 = st.columns (3)
        m1.markdown (f'<div class="metric-card"><div class="metric-title">Squad Average Rating</div><div class="metric-value">{avg_rating:.1f}</div></div>',unsafe_allow_html = True)
        m2.markdown (f'<div class="metric-card"><div class="metric-title">Chemistry Estimate</div><div class="metric-value">{chemistry_score} / 100</div></div>',unsafe_allow_html = True)
        m3.markdown (f'<div class="metric-card"><div class="metric-title">Average Sentiment</div><div class="metric-value" style="color:{sentiment_color (avg_sentiment)};">{avg_sentiment:.2f}</div></div>',unsafe_allow_html = True)

        st.markdown ("<div style='margin-bottom: 10px;'></div>",unsafe_allow_html = True)
        st.dataframe (
            sq_df [["Player","Position","Current_Club","League","Overall","Sentiment_Label"]],
            use_container_width = True,)

elif app_mode == "🎯 Scouting Filter & Target Finder":
    st.subheader ("🎯 Multi Criteria Scouting Filter")

    col_f1,col_f2,col_f3 = st.columns (3)
    with col_f1:
        min_overall = st.slider ("Minimum Overall Rating",60,95,75)
        selected_positions = st.multiselect ("Positions",options = sorted (df ["Position"].unique ()),default = list(df ["Position"].unique ()))
    with col_f2:
        max_age = st.slider ("Maximum Age",18,45,35)
        selected_leagues = st.multiselect ("Leagues",options = sorted (df ["League"].unique ()),default = list (df ["League"].unique ()))
    with col_f3:
        min_sentiment = st.slider ("Minimum Sentiment Score",-1.0,1.0,0.0)
        selected_sentiment_label = st.multiselect ("Sentiment Category",options = sorted (df ["Sentiment_Label"].unique ()),default = list(df ["Sentiment_Label"].unique ()))

    filtered_df = df [
        (df ["Overall"] >= min_overall)
        & (df ["Age"] <= max_age)
        & (df ["Position"].isin (selected_positions))
        & (df ["League"].isin (selected_leagues))
        & (df ["Sentiment_Compound"] >= min_sentiment)
        & (df ["Sentiment_Label"].isin (selected_sentiment_label))
    ]

    st.success (f"Found **{len (filtered_df)}** players matching your scouting criteria")

    st.dataframe (
        filtered_df [["Player","Age","Position","Current_Club","League","Overall","Sentiment_Compound","Sentiment_Label","Scout_Report"]],
        use_container_width = True,
        height = 380,)

    csv_data = filtered_df.to_csv (index = False).encode ("utf-8")
    st.download_button (
        label = "📥 Export Scouting Target Pipeline (.csv)",
        data = csv_data,
        file_name = "Scouting_Targets_Pipeline_2026.csv",
        mime = "text/csv",
        use_container_width = True,)

elif app_mode == "📊 Tactical Radar Cluster & Archetypes":
    st.subheader ("📊 Tactical Archetype Cluster Map")
    st.caption ("Players plotted by attacking output vs defensive contribution, grouped by position - reveals natural tactical archetypes across the dataset")

    cluster_df = df.copy ()
    cluster_df ["Attacking_Index"] = (cluster_df ["Shooting"] + cluster_df ["Dribbling"] + cluster_df ["Pace"])/3
    cluster_df ["Defensive_Index"] = (cluster_df ["Defending"] + cluster_df ["Physical"])/2

    fig_cluster = px.scatter (
        cluster_df,x = "Defensive_Index",y = "Attacking_Index",
        color = "Position",size = "Overall",hover_name = "Player",
        hover_data = ["Current_Club","Age"],
    )
    fig_cluster.add_hline (y = cluster_df ["Attacking_Index"].mean (),line_dash = "dash",line_color = COLORS ["text_secondary"])
    fig_cluster.add_vline (x = cluster_df ["Defensive_Index"].mean (),line_dash = "dash",line_color = COLORS ["text_secondary"])
    st.plotly_chart (style_fig (fig_cluster,height = 560),use_container_width = True)

    st.caption ("Quadrants relative to this squad's averages: top-right = complete two-way players, top-left = pure attackers, bottom-right = defensive anchors, bottom-left = below-average on both axes here")

elif app_mode == "🔥 Hype vs Reality":
    st.subheader ("🔥 Hype vs Reality: Sentiment vs Tactical Rating")
    st.caption ("Sentiment (scaled 0-100 to match Overall) plotted against tactical Overall rating - a simple, real cross-check between scouting narrative and the numbers, not a prediction of anything")

    hype_df = df.copy ()
    hype_df ["Sentiment_Scaled"] = ((hype_df ["Sentiment_Compound"] + 1) / 2) * 100

    fig_hype = px.scatter (
        hype_df,x = "Overall",y = "Sentiment_Scaled",color = "Position",
        size = "Age",hover_name = "Player",hover_data = ["Current_Club","Sentiment_Label"],
        labels = {"Sentiment_Scaled": "Sentiment (scaled to 0-100)"},
    )
    fig_hype.add_hline (y = hype_df ["Sentiment_Scaled"].mean (),line_dash = "dash",line_color = COLORS ["text_secondary"])
    fig_hype.add_vline (x = hype_df ["Overall"].mean (),line_dash = "dash",line_color = COLORS ["text_secondary"])
    st.plotly_chart (style_fig (fig_hype,height = 520),use_container_width = True)

    hype_df ["Gap"] = hype_df ["Sentiment_Scaled"] - hype_df ["Overall"]
    col_a,col_b = st.columns (2)
    with col_a:
        st.markdown ("**📈 Sentiment running ahead of rating**")
        st.dataframe (
            hype_df.sort_values ("Gap",ascending = False)[["Player","Overall","Sentiment_Label"]].head (5),
            use_container_width = True,hide_index = True,)

    with col_b:
        st.markdown ("**📉 Rating running ahead of sentiment**")
        st.dataframe (
            hype_df.sort_values ("Gap")[["Player","Overall","Sentiment_Label"]].head (5),
            use_container_width = True,hide_index = True,)

elif app_mode == "💰 Market Value & Wage Estimator":
    st.subheader ("💰 Market Value & Wage Estimator")
    st.caption ("Illustrative estimates from a disclosed formula (age curve \u00d7 overall rating \u00d7 position) \u2014 not a trained prediction model or real Transfermarkt data")

    col_v1,col_v2 = st.columns (2)
    with col_v1:
        val_player = st.selectbox ("Select Player",options = sorted (df ["Player"].unique ()))
        v_data = df [df ["Player"] == val_player].iloc [0]

        st.markdown (
            f"""
            <div class="player-card">
                <h3 style="color:{COLORS ['accent']}; margin:0;">{v_data ['Player']}</h3>
                <p><b>Estimated Market Value:</b> <span style="color:{COLORS ['accent']}; font-size:1.3rem; font-weight:bold;">\u20ac{v_data ['Est_Market_Value_M']}M</span></p>
                <p><b>Estimated Weekly Wage:</b> \u20ac{v_data ['Est_Weekly_Wage_K']}K / week</p>
                <p><b>Club:</b> {v_data ['Current_Club']} ({v_data ['League']})</p>
            </div>
            """,
            unsafe_allow_html = True,)

        fig_val_bar = px.bar (
            df.sort_values ("Est_Market_Value_M",ascending = False).head (10),
            x = "Est_Market_Value_M",y = "Player",orientation = "h",
            color = "Est_Market_Value_M",
            color_continuous_scale = [COLORS ["surface_alt"],COLORS ["accent"]],
            title = "Top 10 by Estimated Value (\u20acM)",
        )
        st.plotly_chart (style_fig (fig_val_bar,height = 320),use_container_width = True)

    with col_v2:
        st.subheader ("📈 Value vs Age Curve")
        fig_val_scat = px.scatter (
            df,x = "Age",y = "Est_Market_Value_M",size = "Overall",color = "Position",
            hover_name = "Player",hover_data = ["Current_Club","Est_Weekly_Wage_K"],
        )
        st.plotly_chart (style_fig (fig_val_scat,height = 450),use_container_width = True)

elif app_mode == "📈 Sentiment & Market Matrix":
    st.subheader ("📈 Scout Sentiment & Market Matrix")
    c1,c2 = st.columns (2)
    with c1:
        fig_sent = px.bar (
            df.sort_values ("Sentiment_Compound",ascending = False),
            x = "Sentiment_Compound",y = "Player",orientation = "h",
            color = "Sentiment_Compound",
            color_continuous_scale = [COLORS ["sentiment_negative"],COLORS ["sentiment_neutral"],COLORS ["sentiment_positive"]],
        )
        st.plotly_chart (style_fig (fig_sent,height = 480),use_container_width = True)

    with c2:
        fig_scat = px.scatter (
            df,x = "Overall",y = "Sentiment_Compound",size = "Age",color = "Position",hover_name = "Player",)
        st.plotly_chart (style_fig (fig_scat,height = 480),use_container_width = True)