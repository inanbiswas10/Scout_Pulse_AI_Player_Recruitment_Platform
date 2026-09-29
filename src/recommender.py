# src/recommender.py

# Computes cosine similarity across tactical attributes to find
# replacement candidates for a target player. Ranks same-position-group
# players above cross-group ones, rather than ignoring position
# entirely - a pure attribute-similarity match could otherwise suggest
# a center-back as Messi's "replacement." Falls back gracefully to
# cross-group candidates when too few same-group players exist (the
# dataset has only one goalkeeper, for example).

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = ["Pace","Shooting","Passing","Dribbling","Defending","Physical"]

POSITION_GROUPS = {
    "ST": "Attack","RW": "Attack","LW": "Attack","CAM": "Attack",
    "CM": "Midfield","CDM": "Midfield",
    "CB": "Defense","RB": "Defense","LB": "Defense",
    "GK": "Goalkeeper",
}

def get_player_recommendations (target_player: str,df: pd.DataFrame,top_n: int = 3) -> pd.DataFrame:

    # Finds the top N tactical replacements for a target player

    if target_player not in df ["Player"].values:
        raise ValueError (f"Player '{target_player}' not found in dataset !!")

    scaler = MinMaxScaler ()
    scaled_features = scaler.fit_transform (df [FEATURE_COLS])
    sim_matrix = cosine_similarity (scaled_features)

    target_idx = df [df ["Player"] == target_player].index [0]
    target_group = POSITION_GROUPS.get (df.loc [target_idx,"Position"],"Other")

    candidates = []
    for idx,score in enumerate(sim_matrix [target_idx]):
        if idx == target_idx:
            continue
        same_group = POSITION_GROUPS.get (df.loc [idx,"Position"],"Other") == target_group
        candidates.append ((idx,score,same_group))

    # Same position group ranks first, then by similarity score within
    # that. Not a hard filter - with only 20 players and just one
    # goalkeeper, a strict same-position filter could return nothing
    # for some players. This degrades gracefully instead.

    candidates.sort (key = lambda c: (not c [2],-c [1]))

    top_candidates = candidates [:top_n]
    recommended_indices = [c [0] for c in top_candidates]
    match_percentages = [round (c [1] * 100,1) for c in top_candidates]

    recs_df = df.iloc [recommended_indices].copy ()
    recs_df ["Match_Similarity_%"] = match_percentages
    return recs_df

if __name__ == "__main__":

    from data_loader import load_player_data
    
    df = load_player_data ("data/players_2026.csv")
    recs = get_player_recommendations ("Lionel Messi",df,top_n = 3)
    print ()
    print ("\nCosine Similarity Engine Verified (position-aware) !!")
    print ()
    print (recs [["Player","Position","Current_Club","Match_Similarity_%"]])