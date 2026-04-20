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
   # plt.show()
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
  #  plt.show()
    folder = "../Report_1/1_distribution"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{attribute.replace(" ", "_").replace(".","")}_hist.png")


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


def attribute_data(csv_file, attribute):
    data = csv_file[attribute].dropna()
    n = len(data)

    median = np.median(data)
    averge = np.mean(data)
    std_dev = np.std(data, ddof=1)

    if std_dev != 0:
        skew_np = (averge - median) / std_dev
    else:
        skew_np = 0

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers_count = len(outliers)
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

def attack_type_vs_generation(csv_file):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    df = csv_file.copy()

    # reshape types
    types_long = pd.melt(
        df,
        id_vars=["Generation"],
        value_vars=["Type 1", "Type 2"],
        value_name="Type"
    )

    types_long = types_long.dropna()
    types_long = types_long[types_long["Type"] != "None"]

    # contingency table
    pivot = pd.crosstab(types_long["Generation"], types_long["Type"])

    # ✔ normalize per TYPE (column-wise)
    pivot_percent = pivot.div(pivot.sum(axis=0), axis=1) * 100

    # optional: sort types by generation of peak occurrence
    pivot_percent = pivot_percent.loc[:, pivot_percent.idxmax().sort_values().index]

    # flip generations (Gen 1 at bottom)
    pivot_percent = pivot_percent.sort_index(ascending=False)

    # plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_percent, cmap="mako_r", vmin=0,
    annot=True,
    fmt=".0f")

    plt.title("Rozkład typów Pokémonów w generacjach (% udział typu)")
    plt.xlabel("Typ")
    plt.ylabel("Generacja")

    plt.tight_layout()

    folder = "../Report_1/4_other"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/type_distribution_by_generation.png")

    plt.close()

def type2_missing_vs_generation(csv_file):
    df = csv_file.copy()
    df["Type2_missing"] = df["Type 2"].isna() | (df["Type 2"] == "None")

    pivot = pd.crosstab(df["Generation"], df["Type2_missing"])
    pivot.columns = ["Wykonuje drugi atak", "Brak drugiego ataku"]
    pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    ax = pivot_percent.plot(kind="bar", stacked=True, figsize=(10,6),color=["#4d00ff", "#ff5900"])

    plt.title("Występowanie drugiego typu w poszczególnych generacjach")
    plt.xticks(rotation=0)
    plt.ylabel("Procent pokemonów")
    plt.ylim(0, 100)
    plt.legend(loc="upper right")

    plt.tight_layout()
    #plt.show()
    folder = "../Report_1/4_other"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/missing_t2_per_gen.png")

def category_analysis(csv_file):
    attack_type_vs_generation(csv_file)
    type2_missing_vs_generation(csv_file)


def main():
    combats_csv = pd.read_csv('../Dataset/combats.csv')
    pokemon_csv = pd.read_csv('../Dataset/pokemon.csv').fillna("None")
    tests_csv = pd.read_csv('../Dataset/tests.csv')

#    distribution_generation(pokemon_csv)
 #   box_plots_generation(pokemon_csv)
    category_analysis(pokemon_csv)

if __name__=="__main__":
    main()