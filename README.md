# Scout Pulse AI: Player Recruitment & Replacement Platform

> NLP-driven scouting sentiment analysis combined with a position-aware
> tactical replacement engine — built for the Elite Tech Intern Data Analytics
> Internship (Sentiment Analysis Track).

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-2DB88A)

## Overview

Scout Pulse AI analyzes real scouting-report text with VADER sentiment
analysis and recommends tactical replacements using position-aware
cosine similarity across six tactical attributes — with every score
inspectable down to the sentence and keyword level, not a black box.

## Live demo

**[Open the live app](https://scout-pulse-ai-player-recruitment-platform.streamlit.app/)**

## Engineering notes

- **Sentiment scoring is transparent, not a black box.** Every player
  profile includes a sentence-level breakdown and keyword highlighting
  showing exactly which words in the scout report drove the score,
  using VADER's own published lexicon values.
- **A real limitation, disclosed rather than hidden:** general-purpose
  sentiment lexicons can misjudge sports-specific language — VADER
  scores "dominating" as negative, likely from contexts like
  "dominating and controlling." Surfaced directly in the UI.
- **The replacement engine is position-aware.** An earlier version
  matched purely on tactical-attribute similarity, which could suggest
  a defender as a forward's "replacement." Same-position-group matches
  now rank first.
- **The dataset's single goalkeeper is a known, disclosed edge case.**
  With no other keeper to compare against, any goalkeeper "match" is
  flagged in the UI as illustrative only, not tactically meaningful.
- **Market value/wage figures are a disclosed formula** (age curve ×
  rating × position) — explicitly labeled as an illustrative estimate,
  not a trained model or real transfer-market data.
- **Tactical attribute ratings (Pace, Shooting, etc.) are scouting-style
  estimates**, the same category of number as FIFA/EA FC ratings —
  clearly disclosed as such, not presented as precise measured stats.

## Features

- [x] VADER sentiment analysis on real scouting-report text
- [x] Sentence-level sentiment breakdown with keyword highlighting
- [x] Position-aware cosine similarity replacement engine
- [x] Head-to-head player comparison
- [x] Custom squad builder with chemistry estimate
- [x] Multi-criteria scouting filter with CSV export
- [x] Tactical archetype cluster map
- [x] Hype vs Reality (sentiment vs rating) analysis
- [x] Disclosed market value / wage estimator
- [x] Automated test suite (pytest)

## Tech stack

| Layer | Tools |
|---|---|
| Sentiment analysis | VADER (vaderSentiment) |
| Recommendation engine | scikit-learn (cosine similarity) |
| Visualization | Plotly |
| Web app | Streamlit |
| Testing | pytest |

## Project structure

Scout_Pulse_AI_Player_Recruitment_Platform/
├── app/
│ ├── main.py # Streamlit entry point
│ └── theme.py # Design system: colors, fonts, CSS
├── src/
│ ├── data_loader.py # Loads data, computes Overall + valuation estimate
│ ├── recommender.py # Position-aware cosine similarity engine
│ └── sentiment_engine.py # VADER scoring + sentence/keyword breakdown
├── data/
│ └── players_2026.csv
├── tests/
│ └── test_sentiment_engine.py
├── requirements.txt
└── README.md


## Getting started

```bash
git clone https://github.com/inanbiswas10/Scout_Pulse_AI_Player_Recruitment_Platform.git
cd Scout_Pulse_AI_Player_Recruitment_Platform
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
```

## Dataset

Player names, clubs and leagues reflect real September 2026 transfer
information. Tactical attribute ratings (Pace, Shooting, Passing,
Dribbling, Defending, Physical) are scouting-style estimates, not
precise measured statistics — disclosed here rather than implied.

## Author

Built by Inan Biswas as part of the Elite Tech Intern Data Analytics Internship (July 2026 - September 2026)