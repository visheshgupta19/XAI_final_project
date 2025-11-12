# XAI_final_project  Credit Risk Explainability: End-to-End Analysis

# [Watch the Video on YouTube](https://youtu.be/Jb5HeNL_hYQ)

## Overview  
This project develops a credit risk prediction and explainability system using the HELOC dataset.  
It combines machine learning, interpretability, and business optimization to assess borrower risk while maintaining transparency and fairness.  

The final deliverable includes an interactive Streamlit web application for real-time applicant scoring and model explainability.

## Approach

### 1. Data Preparation
- Cleaned dataset by replacing special missing codes (`-9, -8, -7`) with `NaN`.  
- Imputed missing values using median strategy.  
- Encoded target variable: RiskPerformance → (Good = 1, Bad = 0).

### 2. Model Training
- Built and compared **XGBoost** and **LightGBM** classifiers.  

### 3. Global Explainability
- Used Permutation Importance and SHAP summary plots to understand model-wide drivers.  
- Determine key features and if the results align with real world intuition.

### 4. Local Explainability
- Generated SHAP waterfall plots to explain individual predictions.  

### 5. Behavioral Sensitivity
- PDP and ICE plots visualize how changes in top features influence predictions.  

### 6. Threshold Optimization
- Conducted Threshold Sensitivity Analysis to find the optimal cutoff balancing recall and precision.  

### 7. What-If / Counterfactual Analysis
- Simulated how feature improvements affect approval probability.  
- The Counterfactual Path Plot shows how coordinated improvements across multiple behaviors push borderline applicants toward approval.

---

## Streamlit App – HELOC Credit Decision System  

An interactive Streamlit dashboard was built to operationalize the model and make results interpretable for end-users.

### Key Features
#### New Applicant Check
- Users can manually enter, upload, or simulate applicant profiles.  
- The system predicts approval/denial, displays approval probability, and highlights top helping and hurting factors via SHAP values.  
- Provides personalized improvement recommendations, showing how specific credit behaviors could increase approval odds.

#### Dashboard Overview
- Displays business-level metrics (approval rate, precision, recall, profit).  
- Interactive threshold slider connects model performance to profitability.  
- Visualizes confusion matrix, profit vs. threshold curve, and prediction distribution.

#### Individual Assessment
- Allows detailed inspection of any applicant in the dataset.  
- Includes SHAP-based bar plots explaining *why* each decision was made.  
- Provides a full feature profile vs. dataset averages.

---

## Highlights
- **Explainable AI:** Every prediction is transparent and traceable.  
- **Applicant Guidance:** Generates concrete recommendations for credit improvement.  
- **Business Control:** Decision thresholds can be tuned for desired risk-return balance.  
- **Interactive Visualization:** Real-time, user-friendly interface for decision-makers and applicants alike.

## How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/heloc-explainability.git
cd heloc-explainability
```

---

### 2. Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

---

### 3. Prepare Model and Data
Make sure you have the following files in your project folder:
- `xgb_model.pkl` → trained XGBoost model  
- `X_test.csv` → test feature dataset  
- `predictions.csv` → test predictions (with columns `y_test`, `y_prob`)  

If not, train the model first using your notebook or Python script:
```python
import pickle
pickle.dump(xgb, open("xgb_model.pkl", "wb"))
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame({"y_test": y_test, "y_prob": y_prob}).to_csv("predictions.csv", index=False)
```

---

### 4. Run the Streamlit App
Once everything is ready:
```bash
streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

---

