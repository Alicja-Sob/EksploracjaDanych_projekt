import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wittgenstein as lw
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


def merge_legendary(row):
    first = row["Legendary_1"]
    second = row["Legendary_2"]

    if first and second: return "Both"
    elif first: return "Only_first"
    elif second: return "Only_second"
    else: return "Neither"


def prepare_the_data(pokemon_csv, combat_csv, file_name="perpped_dataset"):
    columns_to_keep_1 = ["#", "Type 1", "Type 2", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation", "Legendary"]
    pokemons = pokemon_csv[columns_to_keep_1].copy()

    pokemon_1 = pokemons.rename(columns={
        "#": "ID_1", "Type 1": "Type1_1", "Type 2": "Type2_1", "HP": "HP_1",
        "Attack": "Attack_1", "Defense": "Defense_1", "Sp. Atk": "SpAtk_1",
        "Sp. Def": "SpDef_1", "Speed": "Speed_1",
        "Generation": "Generation_1", "Legendary": "Legendary_1"
    })

    pokemon_2 = pokemons.rename(columns={
        "#": "ID_2", "Type 1": "Type1_2", "Type 2": "Type2_2", "HP": "HP_2",
        "Attack": "Attack_2", "Defense": "Defense_2", "Sp. Atk": "SpAtk_2",
        "Sp. Def": "SpDef_2", "Speed": "Speed_2",
        "Generation": "Generation_2", "Legendary": "Legendary_2"
    })

    fights = combat_csv.merge(pokemon_1, left_on="First_pokemon", right_on="ID_1", how="left")
    fights = fights.merge(pokemon_2, left_on="Second_pokemon", right_on="ID_2", how="left")

    statistic_columns = ["HP", "Attack", "Defense", "SpAtk", "SpDef", "Speed", "Generation"]
    for stat in statistic_columns:
        fights[f"{stat}_diff"] = fights[f"{stat}_1"] - fights[f"{stat}_2"]

    fights["Type1_same"] = (fights["Type1_1"] == fights["Type1_2"])
    fights["Type2_same"] = (fights["Type2_1"] == fights["Type2_2"])
    fights["First_has_type2"] = (fights["Type2_1"] != "None")
    fights["Second_has_type2"] = (fights["Type2_2"] != "None")

    fights["Legendary_rel"] = fights.apply(merge_legendary, axis=1)

    final_columns = [
        "HP_diff", "Attack_diff", "Defense_diff", "SpAtk_diff", "SpDef_diff", "Speed_diff",
        "Generation_diff", "Legendary_rel", "Type1_same", "Type2_same",
        "First_has_type2", "Second_has_type2"
    ]

    fights["Did_1st_win"] = (fights["Winner"] == fights["First_pokemon"])
    final_dataset = fights[final_columns + ["Did_1st_win"]].copy()

    folder = "../../Dataset/Main_prepared_data"
    os.makedirs(folder, exist_ok=True)
    final_dataset.to_csv(f"{folder}/{file_name}.csv", index=False)

    return final_dataset


def get_a_model(max_rule_no, max_rule_length, prune_size):
    return lw.RIPPER(
        random_state=0,
        max_rules=max_rule_no,
        max_rule_conds=max_rule_length,
        prune_size=prune_size
    )


def print_ruleset(ruleset):
    print("\n----- FOLD -----")
    for i, rule in enumerate(ruleset):
        rule_str = str(rule)
        rule_str = rule_str.replace("^", " AND ").replace("[", "").replace("]", "")
        print(f"Rule {i + 1}: {rule_str}")


import re
from collections import Counter

def extract_rule_features(ruleset, counter):
    for rule in ruleset:
        rule_str = str(rule)

        # remove brackets
        rule_str = rule_str.replace("[", "").replace("]", "")

        # split conditions
        conditions = rule_str.split("^")

        for cond in conditions:
            # feature is before '=' or comparison operator
            match = re.match(r"([^=<>!]+)", cond.strip())
            if match:
                feature = match.group(1).strip()
                counter[feature] += 1


def plot_feature_importance(counter, n_folds):
    avg = {k: v / n_folds for k, v in counter.items()}
    df = pd.DataFrame(avg.items(), columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="importance", y="feature")
    plt.title("Average feature importance from RIPPER rules")
    plt.tight_layout()
    plt.show()


def evaluate_model(attributes_X, target_Y, model):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    accuracies, f1s = [], []

    global_counter = Counter()
    n_folds = 0

    for train_index, test_index in kf.split(attributes_X):
        print(f"doing a fold {n_folds}--------------------------------------")
        X_train, X_test = attributes_X.iloc[train_index], attributes_X.iloc[test_index]
        y_train, y_test = target_Y.iloc[train_index], target_Y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, predictions))
        f1s.append(f1_score(y_test, predictions))

        extract_rule_features(model.ruleset_, global_counter)
        n_folds += 1

    plot_feature_importance(global_counter, n_folds)

    return np.mean(accuracies), np.std(accuracies), np.mean(f1s), np.std(f1s)


def train_final_model(X, y, best_parameters):
    model = get_a_model(
        best_parameters["max_rules"],
        best_parameters["max_rule_conds"],
        best_parameters["prune_size"]
    )
    model.fit(X, y)
    return model


def get_X_y(dataset):
    dataset_copy = dataset.copy()
    y = dataset_copy["Did_1st_win"]
    x = dataset_copy.drop(columns=["Did_1st_win"])
    return x, y


def macierz_pomylek(X_test, y_test, model):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))

    cm_df = pd.DataFrame(
        cm,
        index=['Wygra P2 (0)', 'Wygra P1 (1)'],
        columns=['Wygra P2 (0)', 'Wygra P1 (1)']
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    pos_label = labels[1]

    tn, fp, fn, tp = cm.ravel()

    if labels[0] == pos_label:
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"Positive class: {pos_label}")
    print("-" * 65)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    plt.figure()
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz pomyłek - reguły decyzyjne")
    plt.tight_layout()
    os.makedirs("../../Reports/Report_2/4_model", exist_ok=True)
    plt.savefig("../../Reports/Report_2/4_model/classification_results.png", dpi=300)
    plt.show()

    return cm_df




def main():
    combats_csv = pd.read_csv('../../Dataset/combats.csv')
    pokemon_csv = pd.read_csv('../../Dataset/pokemon.csv').fillna("None")

    preped_data_1 = prepare_the_data(pokemon_csv, combats_csv, "main_training_dataset")

    x, y = get_X_y(preped_data_1)

    best_params = {"max_rules": 50, "max_rule_conds": 2, "prune_size": 0.1}

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    model = train_final_model(X_train, y_train, best_params)

    macierz_pomylek(X_test, y_test, model)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy = {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1-score = {f1:.4f} ({f1 * 100:.2f}%)")


if __name__ == "__main__":
    main()