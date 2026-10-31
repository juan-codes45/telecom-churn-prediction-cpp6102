Telecom Customer Churn Prediction
Predicting which telecom customers are likely to leave, using pandas, Matplotlib and scikit-learn.
Built for CPP 6102: Computational Thinking & Programming 4 Data Science (Class Activity 3).
Overview
Customer churn is costly because retaining a customer is usually cheaper than acquiring a new one.
This project analyses the public IBM Telco Customer Churn dataset (7,043 customers, 21 variables), identifies who churns and why and builds a model that ranks customers by churn risk so a retention team can prioritise its budget.
 Dataset
- Source: Project data file [`data/Customer_Churn.csv`](data/Customer_Churn.csv).
- Target: `Churn` (Yes/No). 1,869 churned (26.5%) and 5,174 retained (73.5%).
- Cleaning: `TotalCharges` is stored as text because of 11 blank values (customers with zero tenure). It is converted to numeric and imputed inside the pipeline.

Project structure (files named after the assignment tasks)

 Task 1: Load the Dataset | `/task1_load_dataset.py` |
 Task 2(i) Search | task2i_search.py` |
 Task 2(ii) Sort | task2ii_sort.py` |
 Task 2(iii) Iterate | `task2iii_iterate.py` |
 Task 2(iv) Sequence | `task2iv_sequence.py` |
 Task 3: Data Science Analysis | `task3_data_science_analysis.py` |
 Tasks 4-6: written answers | `docs/` |
 Shared settings | `/config.py` |

Methodology
1. Search: finds high service-volume customers (top quartile of a `service_load` proxy) and risk segments using boolean masks and `DataFrame.query()`.
2. Sort:orders customers by tenure and charges and builds a watch-list of new, high-bill, month-to-month customers.
3. Iterate: engineers nine features, including tenure bands, autopay flag, fibre-without-support flag, average monthly spend, price creep and a new-customer high-bill flag.
4. Sequence: a scikit-learn `Pipeline` (median imputation, scaling, one-hot encoding, model) with a stratified 80/20 split and 5-fold cross-validation. Logistic regression and random forest are compared. Preprocessing is fitted inside each fold, so there is no data leakage.
5. Visualise: six Matplotlib figures (churn by contract, tenure band, charges, service load, ROC curves and feature coefficients).

Key results
- Churn is 42.7% on month-to-month contracts versus 2.8% on two-year contracts.
- Customers in their first year churn at 47.4%.
- Fibre customers without security or tech support churn at 55.0%.
- Both models reach a test AUC of about 0.84 (logistic regression 0.842, random forest 0.840).
- The riskiest 10% of customers contain 75.7% actual churners, a 2.85x lift over the 26.5% base rate.
Limitations
- The dataset has no call-log column, so "service call volume" is approximated by the number of subscribed services. The finding that heavy-bundle customers churn less applies to this proxy only.
- The data is a single snapshot, so results show associations, not causation.
- No ablation was run to measure how much the engineered features add.
How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
Figures and `results.json` are written to `outputs/`.
Tech stack
Python, pandas, NumPy, Matplotlib, scikit-learn
 Author
Juan Collince 
