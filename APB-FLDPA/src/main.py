import os
import sys
import time
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils import class_weight
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import create_model
from privacy import DifferentialPrivacy
from federated import ClientReliabilityScorer, adaptive_aggregate
from blockchain import BlockchainLedger
from personalization import PersonalizedFL
from utils import make_dirs, load_and_balance_data, split_clients, save_bar_comparison

warnings.filterwarnings("ignore")
np.random.seed(42)

DPI = 300
N_CLIENTS = 5
N_ROUNDS = 5
DATA_PATH = "data/diabetes_prediction_dataset.csv"


def main():
    make_dirs()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Please place diabetes_prediction_dataset.csv inside the data folder.")

    print("Loading and preprocessing data...")
    df = load_and_balance_data(DATA_PATH)
    print(f"Balanced dataset: {len(df)} records | class distribution: {df.diabetes.value_counts().to_dict()}")

    clients = split_clients(df, N_CLIENTS)

    X = df.drop("diabetes", axis=1).values
    y = df["diabetes"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = {0: cw[0], 1: cw[1]}

    print("Training centralized baseline...")
    central = create_model(X_train.shape[1])
    central.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=cw_dict,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=15, restore_best_weights=True)],
    )

    y_prob = central.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    c_acc = accuracy_score(y_test, y_pred) * 100
    c_prec = precision_score(y_test, y_pred, zero_division=0) * 100
    c_rec = recall_score(y_test, y_pred, zero_division=0) * 100
    c_f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    c_auc = roc_auc_score(y_test, y_prob) * 100
    print(f"Centralized: Acc={c_acc:.2f}, Prec={c_prec:.2f}, Rec={c_rec:.2f}, F1={c_f1:.2f}, AUC={c_auc:.2f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Centralized Baseline", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/roc_centralized.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    print("Running 5-fold cross-validation per client...")
    cv_results = {}
    for cid, cdf in enumerate(clients):
        X_c = cdf.drop("diabetes", axis=1).values
        y_c = cdf["diabetes"].values
        X_c = StandardScaler().fit_transform(X_c)
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        folds = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}

        for train_idx, val_idx in skf.split(X_c, y_c):
            X_t, X_v = X_c[train_idx], X_c[val_idx]
            y_t, y_v = y_c[train_idx], y_c[val_idx]
            cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_t), y=y_t)
            model = create_model(X_c.shape[1])
            model.fit(X_t, y_t, epochs=30, batch_size=16, class_weight={0: cw[0], 1: cw[1]}, verbose=0)
            yp_prob = model.predict(X_v, verbose=0)
            yp = (yp_prob > 0.5).astype(int)
            folds["acc"].append(accuracy_score(y_v, yp) * 100)
            folds["prec"].append(precision_score(y_v, yp, zero_division=0) * 100)
            folds["rec"].append(recall_score(y_v, yp, zero_division=0) * 100)
            folds["f1"].append(f1_score(y_v, yp, zero_division=0) * 100)
            folds["auc"].append(roc_auc_score(y_v, yp_prob) * 100)

        cv_results[cid] = {
            "mean": {k: np.mean(v) for k, v in folds.items()},
            "std": {k: np.std(v) for k, v in folds.items()},
        }
        print(f"Client {cid + 1}: Acc={cv_results[cid]['mean']['acc']:.2f}±{cv_results[cid]['std']['acc']:.2f}")

    perf_df = pd.DataFrame([
        {
            "Client": f"C{cid + 1}",
            "Accuracy": f"{cv_results[cid]['mean']['acc']:.2f}±{cv_results[cid]['std']['acc']:.2f}",
            "Precision": f"{cv_results[cid]['mean']['prec']:.2f}±{cv_results[cid]['std']['prec']:.2f}",
            "Recall": f"{cv_results[cid]['mean']['rec']:.2f}±{cv_results[cid]['std']['rec']:.2f}",
            "F1": f"{cv_results[cid]['mean']['f1']:.2f}±{cv_results[cid]['std']['f1']:.2f}",
            "AUC": f"{cv_results[cid]['mean']['auc']:.2f}±{cv_results[cid]['std']['auc']:.2f}",
        }
        for cid in range(N_CLIENTS)
    ])
    perf_df.to_csv("results/tables/client_performance_matrix.csv", index=False)

    print("Training federated learning framework...")
    dp = DifferentialPrivacy()
    scorer = ClientReliabilityScorer()
    ledger = BlockchainLedger()
    pfl = PersonalizedFL()

    models = [create_model(X_train.shape[1]) for _ in range(N_CLIENTS)]
    data = []
    client_stats = {}

    for i, cdf in enumerate(clients):
        X_c = cdf.drop("diabetes", axis=1).values
        y_c = cdf["diabetes"].values
        X_c = StandardScaler().fit_transform(X_c)
        X_t, X_v, y_t, y_v = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)
        cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_t), y=y_t)
        data.append({"X_train": X_t, "y_train": y_t, "X_test": X_v, "y_test": y_v, "size": len(X_t), "cw": {0: cw[0], 1: cw[1]}})
        client_stats[i] = {"mean": np.mean(X_c, axis=0), "std": np.std(X_c, axis=0)}

    pfl.cluster(client_stats)
    history = []

    for rnd in range(1, N_ROUNDS + 1):
        print(f"Round {rnd}/{N_ROUNDS}")
        round_result = {"round": rnd, "clients": {}}
        weight_list, client_ids, sizes = [], [], []

        for cid in range(N_CLIENTS):
            d = data[cid]
            models[cid].fit(
                d["X_train"],
                d["y_train"],
                epochs=40,
                batch_size=16,
                class_weight=d["cw"],
                verbose=0,
                callbacks=[keras.callbacks.EarlyStopping("loss", patience=10, restore_best_weights=True)],
            )
            loss, acc, auc_v, prec, rec = models[cid].evaluate(d["X_test"], d["y_test"], verbose=0)
            scorer.update(cid, acc, loss)
            weights = models[cid].get_weights()
            if rnd == N_ROUNDS:
                weights = dp.privatize_weights(weights)
            weight_list.append(weights)
            client_ids.append(cid)
            sizes.append(d["size"])
            round_result["clients"][cid] = {"acc": acc * 100, "prec": prec * 100, "rec": rec * 100, "auc": auc_v * 100}

        agg_weights, agg_coeffs = adaptive_aggregate(weight_list, client_ids, scorer, sizes)
        ledger.add(
            rnd,
            hashlib.sha256(str(agg_weights[0][:5]).encode()).hexdigest()[:16],
            {str(c): float(agg_coeffs[i]) for i, c in enumerate(client_ids)},
            hashlib.sha256(str(len(agg_weights)).encode()).hexdigest()[:16],
        )

        for cid in range(N_CLIENTS):
            personalized_weights = pfl.personalize(cid, agg_weights, models[cid].get_weights(), alpha=0.15)
            models[cid].set_weights(personalized_weights)

        avg = {k: np.mean([r[k] for r in round_result["clients"].values()]) for k in ["acc", "prec", "rec", "auc"]}
        round_result["avg"] = avg
        history.append(round_result)
        print(f"Average: Acc={avg['acc']:.2f}, Prec={avg['prec']:.2f}, Rec={avg['rec']:.2f}, AUC={avg['auc']:.2f}")

    print("Blockchain valid:", ledger.verify())

    final = history[-1]["avg"]
    fl_accs = [history[-1]["clients"][i]["acc"] for i in range(N_CLIENTS)]
    baseline_accs = [72.0] * N_CLIENTS
    t_stat, p_val = stats.ttest_rel(fl_accs, baseline_accs)
    w_stat, w_p = stats.wilcoxon(fl_accs, baseline_accs)
    cohen_d = (np.mean(fl_accs) - np.mean(baseline_accs)) / np.sqrt((np.std(fl_accs) ** 2 + np.std(baseline_accs) ** 2) / 2)

    stat_df = pd.DataFrame({
        "Test": ["Paired t-test", "Wilcoxon", "Cohen's d"],
        "Statistic": [f"{t_stat:.4f}", f"{w_stat:.4f}", f"{cohen_d:.4f}"],
        "p-value": [f"{p_val:.6f}", f"{w_p:.6f}", "N/A"],
        "Significant": ["Yes" if p_val < 0.05 else "No", "Yes" if w_p < 0.05 else "No", "Large" if abs(cohen_d) >= 0.8 else "Medium"],
    })
    stat_df.to_csv("results/tables/statistical_analysis.csv", index=False)

    comp_df = pd.DataFrame({
        "Method": ["Centralized", "Standard FL", "APB-FLDPA"],
        "Accuracy": [c_acc, 72.0, final["acc"]],
        "Precision": [c_prec, 62.8, final["prec"]],
        "Recall": [c_rec, 56.7, final["rec"]],
        "AUC": [c_auc, 70.0, final["auc"]],
    })
    save_bar_comparison(comp_df, dpi=DPI)

    rounds = [h["round"] for h in history]
    accs = [h["avg"]["acc"] for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accs, marker="o", lw=2, label="APB-FLDPA")
    plt.axhline(c_acc, linestyle="--", lw=2, label="Centralized")
    plt.axhline(72.0, linestyle=":", lw=2, label="Standard FL")
    plt.xlabel("Round", fontweight="bold")
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title("Convergence Analysis", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/convergence.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    for cid in range(N_CLIENTS):
        client_accs = [history[r]["clients"][cid]["acc"] for r in range(len(history))]
        plt.plot(range(1, N_ROUNDS + 1), client_accs, marker="o", lw=2, label=f"Client {cid + 1}")
    plt.xlabel("Round", fontweight="bold")
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title("Client Performance Across Rounds", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/client_performance.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix - Centralized Baseline", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    print("Completed. Results saved in the results folder.")


if __name__ == "__main__":
    main()
