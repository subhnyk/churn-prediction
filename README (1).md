# Bank Customer Churn Prediction

An end-to-end machine learning project for predicting whether a bank customer is likely to churn. The project includes exploratory data analysis, feature engineering, model comparison, XGBoost hyperparameter tuning, model evaluation, and an interactive Streamlit application for churn-risk prediction.

## Features

- Data cleaning and leakage checks
- Exploratory data analysis with saved visualizations
- Feature engineering for customer balance and age behavior
- Logistic Regression, Random Forest, and XGBoost comparison
- Class-imbalance handling
- XGBoost tuning with `GridSearchCV`
- Evaluation using precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curve
- Feature-importance analysis
- Adjustable churn decision-threshold analysis
- Streamlit interface for individual customer predictions and model insights

## Project Structure

```text
.
├── app.py
├── churn_prediction.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
└── reports/
    └── .gitkeep
```

After running the notebook, the project will also contain:

```text
models/churn_model.joblib
reports/01_target_distribution.png
reports/02_churn_by_category.png
reports/03_numeric_distributions.png
reports/04_correlation.png
reports/05_model_comparison.png
reports/06_evaluation.png
reports/07_feature_importance.png
```

## Dataset

The notebook and Streamlit app expect the dataset at:

```text
data/Customer-Churn-Records.csv
```

Add the CSV file to the `data` directory before running the project. The expected data contains bank customer attributes such as credit score, geography, gender, age, tenure, balance, number of products, activity status, salary, satisfaction score, card type, points earned, and the `Exited` churn target.

> The dataset itself is not included in this repository package because it was not part of the uploaded project files.

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Create and activate a virtual environment.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

1. Place `Customer-Churn-Records.csv` inside `data/`.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Open `churn_prediction.ipynb` and run all cells.

The notebook trains and tunes the models, saves plots under `reports/`, and writes the final trained model to:

```text
models/churn_model.joblib
```

## Run the Streamlit App

After running the notebook at least once:

```bash
streamlit run app.py
```

The app provides two tabs:

- **Predict** — enter customer information and receive a churn probability with low, medium, or high risk classification.
- **Data & Model Insights** — view dataset metrics, churn-rate breakdowns, and saved model-evaluation graphics.

## Modeling Approach

The notebook removes identifier columns and excludes `Complain` after identifying it as a potential leakage feature. It creates three additional features:

- `HasBalance`
- `BalanceSalaryRatio`
- `AgeGroup`

Categorical variables are encoded, and the data is split into stratified 80/20 train and test sets. Logistic Regression, Random Forest, and XGBoost are compared while accounting for class imbalance. XGBoost is then tuned using 5-fold cross-validation with ROC-AUC as the optimization metric.

## Results

The notebook reports a tuned XGBoost model with approximately **0.88 ROC-AUC** and about **77% recall** for churners. The analysis identifies age, number of products, geography, activity status, gender, and account balance as important churn-related factors.

These values come from the current notebook analysis and may change if the dataset, split, preprocessing, or model parameters are changed.

## Business Use

The model can help prioritize customers for retention campaigns. The notebook particularly highlights higher-risk customer groups and recommends ranking customers by predicted churn probability so retention teams can focus on the highest-risk cases first.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Joblib, Jupyter Notebook, and Streamlit.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
