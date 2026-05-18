import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wittgenstein as lw
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

def prettify_raw_params(filename):
    params_csv = pd.read_csv(f"{filename}.csv")

    numeric_cols = [ "accuracy_mean", "accuracy_std", "f1_mean", "f1_std" ]

    for col in numeric_cols:
        params_csv[col] = params_csv[col].round(4)

    params_csv["Dokladnosc"] = (
            params_csv["accuracy_mean"].map(lambda x: f"{x:.4f}")
            + " ± "
            + params_csv["accuracy_std"].map(lambda x: f"{x:.4f}")
    )

    params_csv["F1-score"] = (
            params_csv["f1_mean"].map(lambda x: f"{x:.4f}")
            + " ± "
            + params_csv["f1_std"].map(lambda x: f"{x:.4f}")
    )

    params_csv = params_csv[ [ "max_rules", "max_rule_conds", "prune_size", "Dokladnosc", "F1-score" ] ]

    params_csv.to_csv(f"pretty_{filename}.csv", index=False)


prettify_raw_params("raw_parameters")