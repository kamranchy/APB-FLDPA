import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def make_dirs():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def load_and_balance_data(path):
    df = pd.read_csv(path)
    for col in ["gender", "smoking_history"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    df.dropna(inplace=True)

    df_majority = df[df.diabetes == 0]
    df_minority = resample(df[df.diabetes == 1], n_samples=len(df_majority), random_state=42)
    return pd.concat([df_majority, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)


def split_clients(df, n_clients=5):
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    return [df.iloc[idx[int(len(df) * i / n_clients): int(len(df) * (i + 1) / n_clients)]] for i in range(n_clients)]


def save_bar_comparison(comp_df, dpi=300):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for idx, metric in enumerate(["Accuracy", "Precision", "Recall", "AUC"]):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(["Centralized", "Standard FL", "APB-FLDPA"], comp_df[metric])
        ax.set_ylabel(metric + " (%)", fontweight="bold")
        ax.set_title(f"{metric} Comparison", fontweight="bold")
        ax.set_ylim([0, 100])
        ax.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, comp_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/comparison_bars.png", dpi=dpi, bbox_inches="tight")
    plt.close()
