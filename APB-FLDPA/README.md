# APB-FLDPA: Adaptive Personalized Blockchain-Federated Learning with Differential Privacy and Attention

This repository contains the implementation of **APB-FLDPA**, an enhanced federated learning framework for diabetes prediction. The framework integrates feature attention, adaptive personalized federated learning, differential privacy, blockchain-based audit logging, client reliability scoring, statistical analysis, and publication-ready visualizations.

## Key Features

- Centralized deep learning baseline
- 5-client federated learning setup
- 5-fold cross-validation per client
- Feature attention layer
- Adaptive reliability-based aggregation
- Differential privacy noise injection
- Blockchain-style training ledger
- Personalized federated learning
- Statistical significance testing
- 300 DPI figures for publication

## Repository Structure

```text
APB-FLDPA/
├── data/
│   └── diabetes_prediction_dataset.csv
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── main.py
│   ├── model.py
│   ├── federated.py
│   ├── privacy.py
│   ├── blockchain.py
│   ├── personalization.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Place the dataset file inside the `data/` directory:

```text
data/diabetes_prediction_dataset.csv
```

The dataset should contain a binary target column named:

```text
diabetes
```

Categorical columns such as `gender` and `smoking_history` are automatically label-encoded.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/APB-FLDPA.git
cd APB-FLDPA
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Outputs

After running the code, results will be saved automatically:

```text
results/figures/roc_centralized.png
results/figures/comparison_bars.png
results/figures/convergence.png
results/figures/client_performance.png
results/figures/confusion_matrix.png
results/tables/client_performance_matrix.csv
results/tables/statistical_analysis.csv
```

## Framework Overview

APB-FLDPA includes the following components:

1. **Feature Attention Module**  
   Learns informative feature interactions using query-key-value attention.

2. **Adaptive Personalized Federated Learning**  
   Aggregates client models using reliability scores and personalizes global updates for each client.

3. **Differential Privacy**  
   Clips and perturbs model weights to improve privacy protection.

4. **Blockchain Ledger**  
   Records each federated round with cryptographic hashes for auditability.

5. **Statistical Evaluation**  
   Uses paired t-test, Wilcoxon signed-rank test, and Cohen’s d effect size.

## Citation

If you use this code, please cite the related article:



## License

This project is released for academic and research purposes.
