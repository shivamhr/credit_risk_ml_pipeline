# Credit Risk ML Pipeline

## Project Overview
This project focuses on building an end-to-end **credit risk classification pipeline** using machine learning. The goal is to identify potentially risky loan applicants and demonstrate how model decisions change based on business-driven risk thresholds.

The project is intentionally designed to reflect **real-world credit risk thinking**, not just model accuracy. Emphasis is placed on **probability-based decisions, recall for risky customers, and trade-offs between financial risk and business opportunity**.

---

## Dataset
The project uses the **German Credit Dataset**, which contains customer-level information such as:
- Age, sex, job type
- Housing status
- Saving and checking account information
- Credit amount and duration
- Loan purpose

The target variable is **credit risk**:
- `0` → Good customer
- `1` → Risky customer

Raw data is cleaned and processed before modelling.

---

## Approach
The workflow follows a structured, reproducible pipeline:

1. **Exploratory Data Analysis (EDA)**  
   - Understanding feature distributions
   - Identifying categorical vs numerical variables
   - Checking class imbalance

2. **Data Preprocessing**  
   - Encoding categorical variables
   - Splitting data into train and test sets

3. **Baseline Model**  
   - Logistic Regression chosen for interpretability
   - Model trained to output probability scores rather than just class labels

4. **Evaluation & Threshold Tuning**  
   - Model evaluated using ROC-AUC, confusion matrix, and recall
   - Default threshold (0.5) compared against lower thresholds
   - Threshold tuning used to increase recall for risky customers

---

## Key Results
- Baseline recall for risky customers at 0.5 threshold: **~0.37**
- Recall after lowering threshold to 0.3: **~0.77**

Lowering the decision threshold significantly reduced **false negatives** (missed risky customers) at the cost of increased **false positives** (rejected good customers).

---

## Business Interpretation
In credit risk, **missing a bad customer is more costly than rejecting a good one**.

This project demonstrates:
- Why accuracy alone is misleading for imbalanced credit data
- Why probability-based decisions are preferred over hard classifications
- How threshold tuning reflects real-world risk appetite and lending strategy

The trade-off between false positives and false negatives is explicitly analysed and justified from a business perspective.

---

## Project Structure
```
credit_risk_ml_pipeline/
│
├── data/
│   ├── raw/                 # Original dataset (optional / ignored)
│   └── processed/           # Cleaned dataset used for modelling
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory data analysis
│   └── 02_Baseline_Model.ipynb  # Modelling, evaluation, threshold tuning
│
├── src/                     # Placeholder for future modular code
├── model/                   # Placeholder for saved model artifacts
├── app/                     # Placeholder for potential deployment
└── README.md
```

The structure is intentionally designed to allow future expansion into modular code or deployment, while keeping the current work notebook-driven for clarity.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## Next Steps
Potential future extensions include:
- SQL-based credit risk analysis
- Cost-sensitive evaluation metrics
- Feature importance analysis
- Simple API or dashboard for model inference

---

## Author
**Shivam Sharma**  
Master’s in Computer Science  
Background in Data Analysis and Risk-focused Machine Learning

