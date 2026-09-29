# src/data_loader.py

# Loads and cleans the 2026 football player dataset. Computes a
# composite Overall rating from tactical attributes, and a transparent,
# disclosed market value / wage estimate - NOT a machine learning
# prediction, just a documented formula based on age, rating and
# position. Labeled as an illustrative estimate everywhere it appears,
# not presented as real Transfermarkt-sourced valuations.

import os
import pandas as pd

NUM_COLS = ["Pace","Shooting","Passing","Dribbling","Defending","Physical"]

# Broad, disclosed position value multipliers reflecting real transfer
# market patterns (attackers typically command premiums over defenders
# and goalkeepers) - not derived from any per-player real data, purely
# illustrative

POSITION_VALUE_MULTIPLIER = {
    "ST": 1.15,"RW": 1.15,"LW": 1.15,"CAM": 1.10,
    "CM": 1.00,"RB": 0.95,"LB": 0.95,"CDM": 0.95,
    "CB": 0.90,"GK": 0.75,
}

def load_player_data (file_path: str = "data/players_2026.csv") -> pd.DataFrame:

    # Loads the dataset, computes Overall rating and illustrative valuation estimates.
    
    if not os.path.exists (file_path):
        raise FileNotFoundError (f"Dataset not found at path: {file_path} !!")

    df = pd.read_csv (file_path)
    df.columns = df.columns.str.strip ()

    for col in NUM_COLS:
        df [col] = pd.to_numeric (df[col],errors = "coerce")

    df ["Overall"] = (
        df ["Pace"] * 0.15
        + df ["Shooting"] * 0.20
        + df ["Passing"] * 0.20
        + df ["Dribbling"] * 0.20
        + df ["Defending"] * 0.10
        + df ["Physical"] * 0.15
    ).round ().astype (int)

    df = _estimate_valuation (df)

    return df

def _estimate_valuation (df: pd.DataFrame) -> pd.DataFrame:

    # Adds Est_Market_Value_M and Est_Weekly_Wage_K - a transparent,
    # disclosed formula based on age curve + Overall rating + position,
    # NOT a trained model and NOT real Transfermarkt data. This exists
    # to be an honest, clearly-labeled illustrative estimate, not a
    # precision claim.

    df = df.copy ()

    def age_factor (age):
        if 24 <= age <= 27:
            return 1.0
        if age < 24:
            return 0.75 + (age-18) * 0.05
        return max (0.15,1.0-(age-27) * 0.06)

    market_values = []
    weekly_wages = []

    for _,row in df.iterrows ():

        pos_mult = POSITION_VALUE_MULTIPLIER.get (row ["Position"],1.0)
        base_value = max (0,(row ["Overall"]-60)) ** 2 * 0.05
        value = round (base_value*age_factor (row ["Age"]) * pos_mult,1)
        value = max (0.5,value)
        wage = round (value*3.2 + row ["Overall"] * 0.8,0)

        market_values.append (value)
        weekly_wages.append (wage)

    df ["Est_Market_Value_M"] = market_values
    df ["Est_Weekly_Wage_K"] = weekly_wages

    return df


if __name__ == "__main__":

    test_df = load_player_data ("data/players_2026.csv")
    print ()
    print (f"\nLoaded {len (test_df)} players successfully !!")
    print ()
    print (test_df [["Player","Overall","Est_Market_Value_M","Est_Weekly_Wage_K"]].head ())