# src/sentiment_engine.py

# Parses 'Scout_Report' text using VADER Sentiment Analysis. Appends
# Compound, Positive, Neutral, and Negative scores, along with
# categorical Sentiment Labels for the full dataset. Also exposes
# get_sentiment_breakdown() for a single scout report - sentence-level
# scores plus the exact words VADER's own lexicon assigned sentiment
# weight to, so the score is inspectable rather than a black box.

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer ()

def analyze_player_sentiments (df):

    # Scores every player's Scout_Report and appends sentiment columns.

    df = df.copy ()

    compounds,positives,neutrals,negatives,labels = [],[],[],[],[]

    for text in df ["Scout_Report"]:
        scores = _analyzer.polarity_scores (str (text))
        comp = scores ["compound"]

        compounds.append (comp)
        positives.append (scores ["pos"])
        neutrals.append (scores ["neu"])
        negatives.append (scores ["neg"])

        if comp >= 0.45:
            labels.append ("Highly Praised / Elite Consensus")
        elif comp >= 0.05:
            labels.append ("Favorable Scout Assessment")
        elif comp <= -0.05:
            labels.append ("Concerns / Critical Observations")
        else:
            labels.append ("Neutral / Mixed Scouting Notes")

    df ["Sentiment_Compound"] = compounds
    df ["Sentiment_Positive"] = positives
    df ["Sentiment_Neutral"] = neutrals
    df ["Sentiment_Negative"] = negatives
    df ["Sentiment_Label"] = labels

    return df

def get_sentiment_breakdown (text: str) -> dict:

    # Sentence-level sentiment scores, plus every word VADER's own
    # lexicon assigns a sentiment value to (roughly -4 to +4). Both are
    # real outputs of VADER's own scoring process, not an invented
    # explanation layered on top.

    text = str (text)
    sentences = re.split (r"(?<=[.!?])\s+",text.strip ())

    sentence_scores = []
    for sentence in sentences:
        if not sentence:
            continue
        scores = _analyzer.polarity_scores (sentence)
        sentence_scores.append ({"text": sentence,"compound": scores ["compound"]})

    words = re.findall (r"[A-Za-z']+",text)
    lexicon_words = {}
    for word in words:
        lex_score = _analyzer.lexicon.get (word.lower ())
        if lex_score is not None:
            lexicon_words [word] = lex_score

    return {"sentences": sentence_scores,"lexicon_words": lexicon_words}

if __name__ == "__main__":

    from data_loader import load_player_data

    df = load_player_data ("data/players_2026.csv")
    df_sent = analyze_player_sentiments (df)
    print ("VADER Sentiment Engine Verified !!")
    print (df_sent [["Player","Sentiment_Compound","Sentiment_Label"]].head ())

    print ("\nSentence level breakdown example (Messi):")
    sample_text = df_sent [df_sent ["Player"] == "Lionel Messi"].iloc [0]["Scout_Report"]
    breakdown = get_sentiment_breakdown (sample_text)
    for s in breakdown ["sentences"]:
        print (f"  [{s ['compound']:+.2f}] {s ['text']}")
    print (f"  Lexicon words found: {breakdown ['lexicon_words']}")