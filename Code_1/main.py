import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

LABELS = {
    "Sp. Atk": "Special Attack points",
    "Sp. Def": "Special Defence points"
}

def barplots(csv_file, attribute):
    plt.figure()
    if csv_file[attribute].nunique() == 2:
        sns.countplot(x=attribute, data=csv_file, hue=attribute,palette={False: "#4d00ff", True: "#ff5900"}, legend=False)
    else:
        sns.countplot(x=attribute, data=csv_file, color="#4d00ff")

    if "Type" in attribute:
        plt.xticks(rotation=65)
    plt.title(f"Dystrybucja atrybutu {attribute}")
    plt.ylabel("Liczba przykładów")

    plt.tight_layout()
 #   plt.show()
    folder = "../Report_1/1_distribution"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{attribute.replace(" ", "_")}_dist.png")


def histograms(csv_file, attribute):
    plt.figure()

    sns.histplot(x=attribute, data=csv_file, color="#4d00ff")

    plt.xlabel(LABELS.get(attribute, attribute))
    plt.title(f"Histogram atrybutu {attribute}")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    #plt.show()
    folder = "../Report_1/1_distribution"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{attribute.replace(" ", "_").replace(".","")}_hist.png")


def distribution_generation(csv_file):
    # Not really reusable but I need an easy way to comment it out in main, to not generate the same graphs over and over again
    for attribute in ['Type 1', 'Type 2', 'Generation', 'Legendary']:
        barplots(csv_file,attribute)

    for attribute in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
        histograms(csv_file, attribute)


def main():
    combats_csv = pd.read_csv('../Dataset/combats.csv')
    pokemon_csv = pd.read_csv('../Dataset/pokemon.csv')
    tests_csv = pd.read_csv('../Dataset/tests.csv')

    distribution_generation(pokemon_csv)

if __name__=="__main__":
    main()