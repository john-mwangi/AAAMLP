# import os
# import gc
# import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers, utils
# from tensorflow.keras import optimizers
# from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model


def create_model(data, catcols):
    """
    This function returns a compiled tensorflow model for entity embeddings
    :param data: a pandas dataframe
    :param catcols: a list of categorical columns
    :return: compiled tf.keras model
    """
    inputs = []
    outputs = []

    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_values/2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(
            input_dim=num_unique_values + 1,
            output_dim=embed_dim,
            name=c
            )(inp)

        out = layers.SpatialDropout1D(rate=0.3)(out)
        out = layers.Reshape(target_shape=(embed_dim,))(out)

        inputs.append(inp)
        outputs.append(out)

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=300, activation="relu")(x)
    x = layers.Dropout(rate=300)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=300, activation="relu")(x)
    x = layers.Dropout(rate=300)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(units=2, activation="softmax")(x)

    model = Model(inputs=x, outputs=y)

    model.compile(optimizer="adam", loss="binary_crossentropy")

    return(model)


def run(fold):
    df = pd.read_csv("../inputs/cat_train_folds.csv")

    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = create_model(data=df, catcols=features)

    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]

    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    ytrain_cat = utils.to_categorical(y=ytrain)
    yvalid_cat = utils.to_categorical(y=yvalid)

    model.fit(
        x=xtrain,
        y=ytrain_cat,
        batch_size=1024,
        validation_data=(xvalid, yvalid_cat),
        epochs=3,
        verbose=1
        )

    valid_preds = model.predict(xvalid)[:, 1]

    print(metrics.roc_auc_score(y_true=yvalid, y_score=valid_preds))

    K.clear_session()


if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
