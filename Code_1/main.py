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
    plt.show()
    folder = "../Report_1/1_distribution"
    os.makedirs(folder, exist_ok=True)
   # plt.savefig(f"{folder}/{attribute.replace(" ", "_")}_dist.png")


def histograms(csv_file, attribute):
    plt.figure()

    sns.histplot(x=attribute, data=csv_file, color="#4d00ff")

    plt.xlabel(LABELS.get(attribute, attribute))
    plt.title(f"Histogram atrybutu {attribute}")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    plt.show()
    folder = "../Report_1/1_distribution"
    os.makedirs(folder, exist_ok=True)
  #  plt.savefig(f"{folder}/{attribute.replace(" ", "_").replace(".","")}_hist.png")


def box_plot(csv_file, attribute):
    plt.figure()

    sns.boxplot(y=attribute, data=csv_file, boxprops=dict(facecolor="white"), medianprops=dict(color="#ff5900"))

    plt.xlabel(LABELS.get(attribute, attribute))
    plt.title(f"Wykres pudełkowy atrybutu {attribute}")
    plt.ylabel("Wartość atrybutu")
    plt.tight_layout()
   # plt.show()
    folder = "../Report_1/3_dataQualityCharts"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{attribute.replace(" ", "_").replace(".", "")}_box.png")


def distribution_generation(csv_file):
    # Not really reusable but I need an easy way to comment it out in main, to not generate the same graphs over and over again
    for attribute in ['Type 1', 'Type 2', 'Generation', 'Legendary']:
        barplots(csv_file,attribute)

    for attribute in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
        histograms(csv_file, attribute)


def attribute_data(csv_file, attribute, save=False, folder=None):
    data = csv_file[attribute].dropna()

    n = len(data)

    median = np.median(data)
    averge = np.mean(data)
    std_dev = np.std(data, ddof=1)  # sample standard deviation

    # Nonparametric skew
    if std_dev != 0:
        skew_np = (averge - median) / std_dev
    else:
        skew_np = 0  # avoid division by zero

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers_count = len(outliers)

    # percentage of outliers
    outliers_pct = (outliers_count / n) * 100

    print(f"======= {attribute} =======")
    print(f"Mediana: {median:.2f}\n"
          f"Odchylenie standardowe: {std_dev:.2f}\n"
          f"Skośność nieparametryczna: {skew_np:.4f}\n"
          f"Przedział wartości (Q1–Q3): [{q1:.2f}; {q3:.2f}]\n"
          f"Liczba punktów oddalonych: {outliers_count}\n"
          f"Procent punktów oddalonych: {outliers_pct:.2f}%\n"
          f"Średnia: {averge:.2f}\n")


def box_plots_generation(csv_file):
    for attribute in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
        box_plot(csv_file, attribute)
        attribute_data(csv_file, attribute)


def main():
    combats_csv = pd.read_csv('../Dataset/combats.csv')
    pokemon_csv = pd.read_csv('../Dataset/pokemon.csv').fillna("None")
    tests_csv = pd.read_csv('../Dataset/tests.csv')

#    distribution_generation(pokemon_csv)
    box_plots_generation(pokemon_csv)

if __name__=="__main__":
    main()