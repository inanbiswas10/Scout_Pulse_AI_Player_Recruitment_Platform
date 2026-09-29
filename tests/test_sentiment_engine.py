# tests/test_sentiment_engine.py

# Automated checks for the sentiment engine and the full data pipeline.
# Run with: pytest tests/test_sentiment_engine.py -v

import os
import sys
import pandas as pd
import pytest

sys.path.insert (0,os.path.join (os.path.dirname (__file__),"..","src"))

from data_loader import load_player_data
from sentiment_engine import analyze_player_sentiments,get_sentiment_breakdown
from recommender import get_player_recommendations

DATA_PATH = os.path.join (os.path.dirname (__file__),"..","data","players_2026.csv")

@pytest.fixture (scope = "module")
def df ():
    return analyze_player_sentiments (load_player_data (DATA_PATH))

class TestDataLoader:
    def test_expected_row_count (self,df):
        assert len (df) == 20

    def test_overall_rating_in_valid_range (self,df):
        assert df ["Overall"].between (0,99).all ()

    def test_market_value_never_negative (self,df):
        assert (df ["Est_Market_Value_M"] >= 0).all ()

    def test_no_duplicate_players (self,df):
        assert df ["Player"].duplicated ().sum () == 0

class TestSentimentEngine:

    def test_compound_score_in_valid_range (self,df):

        # VADER's compound score is always bounded -1 to 1 by definition

        assert df ["Sentiment_Compound"].between (-1,1).all ()

    def test_every_player_has_a_label (self,df):

        assert df ["Sentiment_Label"].notna ().all ()

    def test_positive_neutral_negative_sum_to_one (self,df):

        # VADER's pos/neu/neg proportions are defined to sum to 1.0

        totals = df ["Sentiment_Positive"] + df ["Sentiment_Neutral"] + df ["Sentiment_Negative"]
        assert (totals.round (2) == 1.0).all ()

    def test_breakdown_sentences_sum_reasonably (self,df):

        sample_text = df.iloc [0]["Scout_Report"]
        breakdown = get_sentiment_breakdown (sample_text)
        assert len (breakdown ["sentences"]) >= 1
        assert all (-1 <= s ["compound"] <= 1 for s in breakdown ["sentences"])

    def test_breakdown_lexicon_words_are_real_words_from_text (self,df):

        sample_text = df.iloc [0]["Scout_Report"]
        breakdown = get_sentiment_breakdown (sample_text)
        for word in breakdown ["lexicon_words"]:
            assert word in sample_text

class TestRecommender:

    def test_recommendations_exclude_target_player (self,df):
        recs = get_player_recommendations ("Lionel Messi",df,top_n = 3)
        assert "Lionel Messi" not in recs ["Player"].values

    def test_returns_requested_count (self,df):

        recs = get_player_recommendations ("Kylian Mbappe",df,top_n = 3)
        assert len (recs) == 3

    def test_match_similarity_bounded (self,df):

        recs = get_player_recommendations ("Erling Haaland",df,top_n = 5)
        assert recs ["Match_Similarity_%"].between (0,100).all ()

    def test_unknown_player_raises_error (self,df):

        with pytest.raises(ValueError):

            get_player_recommendations ("Not A Real Player",df,top_n = 3)