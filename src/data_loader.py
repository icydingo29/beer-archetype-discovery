import pandas as pd

FEATURES = ['ABV', 'Body', 'Bitter', 'Sweet', 'Sour', 'Hoppy', 'Malty']

def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    features = FEATURES
    data = df[features].dropna()

    data_norm = (data - data.min()) / (data.max() - data.min())
    df = df.dropna(subset=features).reset_index(drop=True)
    
    return data_norm.values, df, features

if __name__ == "__main__":
    pass