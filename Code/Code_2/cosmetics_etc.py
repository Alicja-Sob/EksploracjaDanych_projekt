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


def count_and_graph_attributes_from_txt(txt_filename):
    def extract_attrs(rule):
        parts = rule.split(" AND ")
        return [p.split("=")[0].strip() for p in parts if "=" in p]

    with open(txt_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rules = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("-----"):
            continue
        if line.startswith("Rule"):
            rules.append(line.split(":", 1)[1].strip())

    counter = Counter()
    for rule in rules:
        counter.update(extract_attrs(rule))

    total = sum(counter.values())

    df = pd.DataFrame(
        [{"attribute": k, "count": v, "percentage": v / total * 100}
         for k, v in counter.items()]
    ).sort_values("percentage", ascending=False)

    print(df)

    plt.figure()
    sns.barplot(data=df, x="percentage", y="attribute", palette="mako")
    plt.title("Wystąpienia atrybutów w wygenerowanych regułach")
    plt.xlabel("Procentowa ilość wystąpień atrybutów")
    plt.ylabel("Atrybut")
    plt.tight_layout()
    os.makedirs("../../Reports/Report_2/4_model", exist_ok=True)
    plt.savefig("../../Reports/Report_2/4_model/attribute_importance.png", dpi=300)
    plt.show()

    return df

count_and_graph_attributes_from_txt("../../Dataset/rules_1.txt")
#prettify_raw_params("raw_parameters")