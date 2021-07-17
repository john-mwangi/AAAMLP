import pandas as pd
from pandas.core.algorithms import mode
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("../inputs/cat_train.csv")

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, "kfold"] = fold

    df.to_csv("../inputs/cat_train_folds.csv", index=False)
