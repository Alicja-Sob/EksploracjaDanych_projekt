import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = '../Report_1/2_correlations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

pokemon = pd.read_csv('../Dataset/pokemon.csv')
combats = pd.read_csv('../Dataset/combats.csv')

first_counts = combats['First_pokemon'].value_counts()
second_counts = combats['Second_pokemon'].value_counts()
total_fights = first_counts.add(second_counts, fill_value=0)

winners = combats['Winner'].value_counts()

win_results = pd.DataFrame({
    'TotalFights': total_fights,
    'Wins': winners
}).fillna(0)

# Win Rate
win_results['WinRate'] = win_results['Wins'] / win_results['TotalFights']

df = pokemon.merge(win_results, left_on='#', right_index=True, how='left')

df['Legendary'] = df['Legendary'].astype(int)

stats_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']

corr_matrix_stats = df[stats_cols].corr()

#heatmap bez WinRate
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_stats,
            annot=True,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            fmt=".2f",
            linewidths=0.5)
plt.title('Macierz korelacji statystyk Pokemonów')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/korelacja_statystyk.png')
print(f"Wykres korelacji statystyk został zapisany do {OUTPUT_DIR}/korelacja_statystyk.png")

# korelacjaz WinRate
all_cols = stats_cols + ['WinRate']
corr_matrix_all = df[all_cols].corr()
win_rate_corr = corr_matrix_all['WinRate'].drop('WinRate').sort_values(ascending=False)

#heatmap + WinRate
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_all,
            annot=True,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            fmt=".2f",
            linewidths=0.5)
plt.title('Macierz korelacji statystyk Pokemonów (z Win Rate)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/korelacja_heatmap.png')
print(f"Heatmap korelacji (z WinRate) został zapisany do {OUTPUT_DIR}/korelacja_heatmap.png")

# korelacja z WinRate wykres
plt.figure(figsize=(10, 6))
sns.barplot(x=win_rate_corr.values, y=win_rate_corr.index, hue=win_rate_corr.index, palette='viridis', legend=False)
plt.title('Korelacja statystyk z szansą na wygraną (Win Rate)')
plt.xlabel('Współczynnik korelacji (Pearson)')
plt.ylabel('Statystyka')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/korelacja_winrate.png')
print(f"Wykres korelacji z WinRate został zapisany do {OUTPUT_DIR}/korelacja_winrate.png")

print("\nKorelacja statystyk z szansą na wygraną (WinRate):")
print(win_rate_corr)
print("\nMacierz korelacji statystyk (zaokrąglona do 2 miejsc):")
print(corr_matrix_all.round(2).to_string())

