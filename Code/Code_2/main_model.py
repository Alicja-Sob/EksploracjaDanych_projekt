import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wittgenstein as lw
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

def merge_legendary(row):
    first = row["Legendary_1"]
    second = row["Legendary_2"]

    if first and second: return "Both"
    elif first: return "Only_first"
    elif second: return "Only_second"
    else: return "Neither"


def prepare_the_data(pokemon_csv, combat_csv, file_name = "perpped_dataset"):
    # prepares the data for main model and saves it to a csv file
    columns_to_keep_1 = ["#", "Type 1", "Type 2", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation", "Legendary"]
    pokemons = pokemon_csv[columns_to_keep_1].copy()

    pokemon_1 = pokemons.rename(columns={ "#": "ID_1",  "Type 1": "Type1_1", "Type 2": "Type2_1", "HP": "HP_1",
        "Attack": "Attack_1", "Defense": "Defense_1", "Sp. Atk": "SpAtk_1", "Sp. Def": "SpDef_1", "Speed": "Speed_1",
        "Generation": "Generation_1", "Legendary": "Legendary_1"})

    pokemon_2 = pokemons.rename(columns={ "#": "ID_2",  "Type 1": "Type1_2", "Type 2": "Type2_2", "HP": "HP_2",
        "Attack": "Attack_2", "Defense": "Defense_2", "Sp. Atk": "SpAtk_2", "Sp. Def": "SpDef_2", "Speed": "Speed_2",
        "Generation": "Generation_2", "Legendary": "Legendary_2"})

    fights = combat_csv.merge(pokemon_1, left_on="First_pokemon", right_on="ID_1", how="left")
    fights = fights.merge(pokemon_2, left_on="Second_pokemon", right_on="ID_2", how="left" )

    statistic_columns = ["HP", "Attack", "Defense", "SpAtk", "SpDef", "Speed", "Generation"]
    for stat in statistic_columns:
        fights[f"{stat}_diff"] = fights[f"{stat}_1"] - fights[f"{stat}_2"]

    fights["Type1_same"] = (fights["Type1_1"] == fights["Type1_2"])
    fights["Type2_same"] = (fights["Type2_1"] == fights["Type2_2"])
    fights["First_has_type2"] = (fights["Type2_1"] != "None")
    fights["Second_has_type2"] = (fights["Type2_2"] != "None")

    fights["Legendary_rel"] = fights.apply(merge_legendary, axis=1)

    final_columns = [ "HP_diff", "Attack_diff", "Defense_diff", "SpAtk_diff", "SpDef_diff", "Speed_diff",
        "Generation_diff", "Legendary_rel", "Type1_same", "Type2_same", "First_has_type2", "Second_has_type2"]

    fights["Did_1st_win"] = (fights["Winner"] == fights["First_pokemon"])
    final_dataset = fights[final_columns + ["Did_1st_win"]].copy()

    folder = "../../Dataset/Main_prepared_data"
    os.makedirs(folder, exist_ok=True)
    final_dataset.to_csv(f"{folder}/{file_name}.csv", index=False)

    return final_dataset


def get_a_model(max_rule_no, max_rule_length, prune_size):
    return lw.RIPPER(random_state=0, max_rules = max_rule_no, max_rule_conds = max_rule_length, prune_size = prune_size)


def print_ruleset(ruleset):
    print("\n----- FOLD -----")

    for i, rule in enumerate(ruleset):
        rule_str = str(rule)

        rule_str = rule_str.replace("^", " AND ")
        rule_str = rule_str.replace("[", "").replace("]", "")

        print(f"Rule {i + 1}: {rule_str}")


def evaluate_model(attributes_X, target_Y, model):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    accuracies, f1s = [], []

    for train_index, test_index in kf.split(attributes_X):
        X_train, X_test = attributes_X.iloc[train_index], attributes_X.iloc[test_index]
        y_train, y_test = target_Y.iloc[train_index], target_Y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)


        accuracies.append(accuracy_score(y_test, predictions))
        f1s.append(f1_score(y_test, predictions))

        print_ruleset(model.ruleset_)


    return np.mean(accuracies), np.std(accuracies), np.mean(f1s), np.std(f1s)


def find_best_parameters(attributes_X, target_Y):
    results = []
    for rules in [10,20,30,40,50]:
        for c_length in [2,3,4,5]:
            for prune in [0.1, 0.33, 0.75]:
                print(f"Checking: {rules}, {c_length}, {prune}")
                model = get_a_model(max_rule_no = rules, max_rule_length = c_length, prune_size = prune)
                acc_avg, acc_s, f1_avg, f1_s = evaluate_model(attributes_X, target_Y, model)

                results.append({
                    "max_rules": rules,
                    "max_rule_conds": c_length,
                    "prune_size": prune,
                    "accuracy_mean": acc_avg,
                    "accuracy_std": acc_s,
                    "f1_mean": f1_avg,
                    "f1_std": f1_s
                })

    return pd.DataFrame(results)


def train_final_model(X, y, best_parameters):
    model = get_a_model(best_parameters["max_rules"], best_parameters["max_rule_conds"], best_parameters["prune_size"])

    model.fit(X, y)
    return model

def get_X_y(dataset):
    dataset_copy = dataset.copy()
    y = dataset_copy["Did_1st_win"]
    x = dataset_copy.drop(columns=["Did_1st_win"])

    return x, y


def get_best_parameters(raw_parameters):
    best_row = raw_parameters.sort_values("f1_mean", ascending=False).iloc[0]

    best_params = {
        "max_rules": int(best_row["max_rules"]),
        "max_rule_conds": int(best_row["max_rule_conds"]),
        "prune_size": float(best_row["prune_size"])
    }

    print("\nBest parameters:", best_params)
    return best_params



def main():
    combats_csv = pd.read_csv('../../Dataset/combats.csv')
    pokemon_csv = pd.read_csv('../../Dataset/pokemon.csv').fillna("None")
    tests_csv = pd.read_csv('../../Dataset/tests.csv')

    preped_data_1 = prepare_the_data(pokemon_csv, combats_csv, "main_training_dataset")  # data for training - pokemons merged by combats file

    x, y = get_X_y(preped_data_1)
    #raw_parameters = find_best_parameters(x, y)
    #raw_parameters.to_csv("raw_parameters.csv", index=False)

    #print(raw_parameters)
    #best_params = get_best_parameters(raw_parameters)
    best_params = { "max_rules": 50, "max_rule_conds": 2, "prune_size": 0.1 }

    model = train_final_model(x, y, best_params)

    evaluate_model(x, y, model)


if __name__=="__main__":
    main()