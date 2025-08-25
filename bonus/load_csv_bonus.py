import pandas as pd

def load(path: str) -> pd.core.frame.DataFrame:
    df = None
    try:
        df = pd.read_csv(path, index_col='Index')
        print(f"Loading dataset of dimensions {df.shape}")
        print(len(df.index))
        for i in range(len(df.index)):
            float(df['Arithmancy'].iloc[i])
            float(df['Astronomy'].iloc[i])
            float(df['Herbology'].iloc[i])
            float(df['Defense Against the Dark Arts'].iloc[i])
            float(df['Divination'].iloc[i])
            float(df['Muggle Studies'].iloc[i])
            float(df['Ancient Runes'].iloc[i])
            float(df['History of Magic'].iloc[i])
            float(df['Transfiguration'].iloc[i])
            float(df['Potions'].iloc[i])
            float(df['Care of Magical Creatures'].iloc[i])
            float(df['Charms'].iloc[i])
            float(df['Flying'].iloc[i])

    except Exception as e:
        raise e
        
    return df
