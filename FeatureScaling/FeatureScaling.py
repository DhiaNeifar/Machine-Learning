import pandas as pd


def featureScaling(df) -> pd.DataFrame:
    return (df - df.mean()) / df.std()

