import pandas
import pandas as pd


def map_feature(df, n=6) -> pandas.DataFrame:
    cols = df.columns.tolist()
    col_1 = df[cols[0]]
    col_2 = df[cols[1]]
    df['bias'] = [1 for _ in range(len(df))]
    new_cols = ['bias'] + cols
    new_df = df.reindex(columns=new_cols)

    for i in range(2, n + 1):
        for j in range(i + 1):
            new_df[f'({i - j}, {j})'] = col_1.pow(i - j) * col_2.pow(j)
    return new_df


def map_feature_row(fl1, fl2, theta, n=6) -> float:
    comp, ind = 0, 0

    for i in range(n + 1):
        for j in range(i + 1):
            print(f'theta {ind}, x1 {i - j}, x2 {j}')
            comp += pow(fl1, i - j) * pow(fl2, j) * theta[ind]
            ind += 1
    return comp


if __name__ == '__main__':

    col_names = ['test_1', 'test_2', 'admission']
    data = pd.read_csv('ex2data2.txt', names=col_names, header=None, delimiter=',')
    x = data[['test_1', 'test_2']]
    X = map_feature(x, 6)
    print(X.head())
