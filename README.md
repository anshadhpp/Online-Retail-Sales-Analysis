# 🛒 E-Commerce High-Value Transaction Classifier

A machine learning pipeline and exploratory analytics project built on real-world transnational online retail logs (541,909 transactions)[cite: 1]. This project cleans transactional anomalies, engineers temporal and spend features, and trains an end-to-end tuned **XGBoost Classifier** to predict whether a given transaction line item will result in a **High-Value Order (> £15.00)**[cite: 1].

---

## 📌 Project Overview

- **Dataset**: Online retail transaction logs across 8 original features (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`)[cite: 1].
- **Core Objective**: Predict high-value revenue drivers at the transaction level using categorical, numerical, and temporal signals.
- **Modeling Framework**: Scikit-Learn `ColumnTransformer` + `Pipeline` integrating `XGBClassifier` with `RandomizedSearchCV` hyperparameter optimization[cite: 1].

---

## 📊 Key Findings from Exploratory Data Analysis (EDA)

1. **Volume Clustering**: Transaction quantities are heavily concentrated in single/double-unit orders, with noticeable spikes at **6, 12, and 24 units** reflecting standard wholesale pack distributions[cite: 1].
2. **Catalog Price Elasticity**: >75% of unit prices fall below **£4.00**, classifying the merchant primarily as a high-volume giftware and novelty goods supplier[cite: 1].
3. **Temporal Peak**: Order volume surges between **12:00 PM and 3:00 PM** across business days[cite: 1].
4. **Target Threshold**: Binarizing transactions at **£15.00** creates a natural separation between routine retail line items (67.2%) and commercial/bulk orders (32.8%)[cite: 1].

---

## ⚙️ Model Architecture & Performance

The pipeline applies `StandardScaler` to continuous/temporal metrics (`UnitPrice`, `Hour`, `Month`) and `OneHotEncoder(handle_unknown='ignore')` to geographic features (`Country`), preventing data leakage across cross-validation splits[cite: 1].

| Metric | Baseline XGBoost | Tuned XGBoost (`RandomizedSearchCV`) |
| :--- | :---: | :---: |
| **Accuracy** | 76.93% | **77.12%** |
| **F1-Score (Macro / Class 1)** | 0.5545 | **0.5812** |
| **ROC / Logloss Optimization** | Default (`max_depth=5`) | Tuned (`max_depth=9`, `subsample=0.6`, `colsample_bytree=0.6`) |

---

## 🚀 Quickstart & Installation

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/your-username/online-retail-value-classifier.git
cd online-retail-value-classifier

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
