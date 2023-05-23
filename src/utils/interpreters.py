import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import utils.models as utm
import utils.helpers as uth


def save_linear_model_features(model, vectorizer, homer_saved, save_file):
    # get weights
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            weights = layer.state_dict()['weight']
    # from vectorizer, get the feature names
    if vectorizer=="homer":
        df = pd.read_csv(homer_saved, nrows=0, index_col=0)
        features = np.array(df.columns)
        weights = weights.detach().cpu().numpy().flatten()
        assert len(features) == len(weights)
        feat_df = pd.DataFrame({"features": features, "weights": weights})
        feat_df.sort_values("weights", ascending=False).to_csv(save_file, index=False)
    else:
        raise ValueError(f"Model interpretation not available for linear model with {vectorizer} vectorizer")
    return
