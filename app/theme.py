# app/theme.py

# Design system for Scout Pulse: AI Player Recruitment & Replacement
# Engine.

# Palette concept: a stadium under floodlights, not a trading floor or
# an ops room. Deep pitch-green-black background, kept structural
# rather than paired with a bright green accent (avoiding the generic
# "near-black plus acid accent" AI-dashboard look). The brand accent is
# a broadcast-graphics gold - "star player" gold, distinct from the
# green/red used for sentiment so an important number and a sentiment
# reading never look the same.

# Typography: Oswald (an authentic condensed sports-broadcast typeface)
# for headings and large numbers, Barlow for body text and labels

import streamlit as st

COLORS = {
    "bg": "#0D1410",
    "surface": "#16211B",
    "surface_alt": "#1E2C24",
    "border": "#2A3D33",
    "text_primary": "#EDF3EF",
    "text_secondary": "#7E9686",
    "accent": "#F2B807",               # broadcast gold - brand, star ratings, highlights
    "sentiment_positive": "#34C77B",
    "sentiment_negative": "#D93B56",
    "sentiment_neutral": "#7E9686",
}

def inject_theme ():

    # Injects fonts and CSS overrides. Call once, at the top of app/main.py

    st.markdown (
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

        :root {{
            --bg: {COLORS ["bg"]};
            --surface: {COLORS ["surface"]};
            --surface-alt: {COLORS ["surface_alt"]};
            --border: {COLORS ["border"]};
            --text-primary: {COLORS ["text_primary"]};
            --text-secondary: {COLORS ["text_secondary"]};
            --accent: {COLORS ["accent"]};
            --sentiment-positive: {COLORS ["sentiment_positive"]};
            --sentiment-negative: {COLORS ["sentiment_negative"]};
            --sentiment-neutral: {COLORS ["sentiment_neutral"]};
        }}

        .stApp {{
            background-color: var(--bg);
            font-family: 'Barlow', sans-serif;
            color: var(--text-primary);
        }}

        [data-testid="stHeader"] {{
            background-color: var(--bg);
        }}

        h1, h2, h3, h4 {{
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: 0.01em;
            text-transform: uppercase;
        }}

        [data-testid="stSidebar"] {{
            background-color: var(--surface);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text-primary);
        }}

        .metric-card {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: 6px;
            padding: 0.8rem 0.7rem;
            text-align: center;
        }}
        .metric-title {{
            font-family: 'Barlow', sans-serif;
            color: var(--text-secondary);
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-value {{
            font-family: 'Oswald', sans-serif;
            color: var(--accent);
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }}

        .player-card {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }}
        .player-card p {{
            margin: 0.15rem 0;
            font-family: 'Barlow', sans-serif;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 4px;
        }}
        </style>
        """,
        unsafe_allow_html = True,)

def sentiment_color (compound_score):

    # Returns the themed color for a VADER compound sentiment score

    if compound_score >= 0.05:
        return COLORS ["sentiment_positive"]
    if compound_score <= -0.05:
        return COLORS ["sentiment_negative"]
    return COLORS ["sentiment_neutral"]