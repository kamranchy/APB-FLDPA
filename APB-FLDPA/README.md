# APB-FLDPA: Adaptive Personalized Blockchain-Federated Learning with Differential Privacy and Attention

This repository contains the implementation of **APB-FLDPA**, an enhanced federated learning framework for diabetes prediction. The framework integrates feature attention, adaptive personalized federated learning, differential privacy, blockchain-based audit logging, client reliability scoring, statistical analysis, and publication-ready visualizations.

## Abstract
Developing robust medical artificial intelligence (AI) requires collaboration across multiple institutions, but strict data protection regulations such as HIPAA and GDPR prevent centralized patient data sharing. Existing federated learning (FL) methods often exhibit 15–30% performance degradation in real-world clinical settings due to data heterogeneity, security threats, and privacy constraints.
We present APB-FLDPA, a privacy-preserving federated learning framework for secure multi-hospital disease prediction. APB-FLDPA integrates five key innovations: (i) adaptive Byzantine-resilient aggregation using dynamic client trust scoring, (ii) self-attention for automated clinical feature importance, (iii) selective differential privacy applied at the final aggregation stage, (iv) cluster-aware personalization to handle cross-institutional heterogeneity, and (v) a lightweight blockchain module to ensure model integrity.
Evaluated across five institutions using large-scale Diabetes (183,000 patients) and Thyroid (6,840 patients) datasets, APB-FLDPA achieved 90.8% accuracy for diabetes and 83.8% accuracy for thyroid disease, with minimal performance loss (<0.2%) compared to centralized learning. Statistical tests confirmed significant improvements, and selective differential privacy outperformed conventional methods by 5.6% in accuracy.
These results demonstrate that APB-FLDPA provides a scalable, high-performance, and privacy-compliant solution for real-world federated medical AI.

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

## Architecture Pipeline
![Architecture](https://github.com/kamranchy/APB-FLDPA/blob/2a26b5340fe89df9ecf949ea7ac13896949ca199/APB-FLDPA/APB-FLDPA%20Architecture.png)




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
