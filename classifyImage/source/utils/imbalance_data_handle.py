import pandas as pd
def balance_data(csv=None,image_per_epoch=200):
    if isinstance(csv,str):
        csv = pd.read_csv(csv)
    labels = set(csv.label)
    dfs = []
    for label in labels:
        df = csv[csv.label==label].sample(n=image_per_epoch,replace=True)
        dfs.append(df)
    df = pd.concat(dfs,axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df
