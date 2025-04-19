import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import preprocessing file
from preprocessing import load_data

# Create folder for visualizations
VISUAL_DIR = "data_visualizations"
os.makedirs(VISUAL_DIR, exist_ok=True)

# Load the processed data
X_train, X_test, y_train, y_test, feature_names = load_data()
df = X_train.copy()
df["Views"] = y_train

# --- Univariate Visualizations ---
univariate_cols = [
    "Danceability", "Energy", "Valence", "Loudness", "Speechiness",
    "Instrumentalness", "Acousticness", "Tempo", "Duration_ms", "Stream", "Views", "Channel"
]

for col in univariate_cols:
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=40)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"univariate_{col.lower()}.png"))
    plt.close()

    # Boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"boxplot_{col.lower()}.png"))
    plt.close()

# --- Multivariate Visualizations ---
multivariate_pairs = [
    ("Danceability", "Views"),
    ("Energy", "Views"),
    ("Valence", "Loudness"),
    ("Tempo", "Energy"),
    ("Loudness", "Energy"),
    ("Views", "Stream"),
    ("Speechiness", "Acousticness"),
    ("Instrumentalness", "Acousticness"),
    ("Valence", "Views"),
    ("Duration_ms", "Views"),
    ("Danceability", "Valence"),
    ("Speechiness", "Instrumentalness"),
    ("Channel", "Views"),
    ("Title", "Views")
]

for x, y in multivariate_pairs:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"multivariate_{x.lower()}_vs_{y.lower()}.png"))
    plt.close()

# --- Pairplot (Selected Features) ---
sns.pairplot(df[["Views", "Energy", "Loudness", "Danceability", "Valence"]], corner=True)
plt.savefig(os.path.join(VISUAL_DIR, "pairplot_subset.png"))
plt.close()

print(f"All visualizations saved to the '{VISUAL_DIR}' folder.")


# --- Histogram of views ---
plt.figure(figsize=(10, 6))

# Use regular binning
num_bins = 50
plt.hist(df['Views'], bins=num_bins, alpha=0.7, linewidth=0.5)

# Add labels and title
plt.title('Distribution of YouTube video Views', fontsize=16)
plt.xlabel('Views', fontsize=12)
plt.ylabel('Videos', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "views_distribution.png"))
plt.close()