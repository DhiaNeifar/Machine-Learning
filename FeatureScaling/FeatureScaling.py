import pandas as pd


def featureScaling(df):
    mu = df.mean()
    sigma = df.std()
    return (df - mu) / sigma, mu, sigma

