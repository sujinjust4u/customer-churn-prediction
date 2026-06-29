# 📉 Cloud-Native Customer Churn Prediction

A full-stack, cloud-ready machine learning application that predicts customer churn for telecom companies using an XGBoost classifier. Built with a FastAPI backend, HTML/CSS frontend, and fully containerized via Docker for seamless deployment.

---

## 🚀 Overview

Customer churn is one of the costliest problems in subscription-based industries. This project provides an end-to-end predictive pipeline — from raw data preprocessing to a deployable REST API with a live web interface — enabling businesses to proactively identify at-risk customers and act before they leave.

The model is trained on the **IBM Telco Customer Churn dataset** and achieves strong classification performance via gradient boosting, with support for batch and single-record inference.

---

## 🛠️ Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| ML Model     | XGBoost (`binary:logistic`)         |
| Backend      | Python 3.12, FastAPI, Uvicorn       |
| Frontend     | HTML, CSS (served via StaticFiles)  |
| Preprocessing| scikit-learn, pandas, NumPy         |
| Containerization | Docker                          |
| Dataset      | IBM Telco Customer Churn (Kaggle)   |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── backend/
│   ├── app.py               # FastAPI application entry point
│   └── requirements.txt     # Python dependencies
├── content/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Training dataset
├── frontend/
│   └── index.html           # Web UI for churn prediction
├── Dockerfile               # Container build config
└── test.py                  # Standalone model training & evaluation script
```

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Converts `TotalCharges` to numeric, drops rows with nulls
   - Drops non-predictive columns (`customerID`)
   - Label-encodes all categorical features

2. **Model Training**
   - XGBoost classifier (`binary:logistic`) with `logloss` eval metric
   - 80/20 stratified train-test split

3. **Inference API**
   - FastAPI exposes a prediction endpoint
   - Accepts customer feature data, returns churn probability/class

4. **Frontend**
   - Simple HTML interface served as static files via FastAPI

---

## 📊 Model Performance

Evaluated on a held-out test set (20%, stratified):

| Metric              | Value             |
|---------------------|-------------------|
| Model               | XGBoost Classifier|
| Objective           | binary:logistic   |
| Eval Metric         | logloss           |
| Train/Test Split    | 80% / 20%         |

> Run `test.py` locally to view accuracy, classification report, and confusion matrix on your dataset split.

---

## 🐳 Running with Docker

### Build the image

```bash
docker build -t churn-prediction .
```

### Run the container

```bash
docker run -p 8080:8080 churn-prediction
```

The API will be live at `http://localhost:8080`.

---

## 🧪 Running Locally (without Docker)

### 1. Clone the repo

```bash
git clone https://github.com/sujinjust4u/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Open the UI

Navigate to `http://localhost:8080` in your browser.

---

## 🧬 Standalone Training Script

To train and evaluate the model independently:

```bash
python test.py
```

This outputs accuracy, a full classification report, and a confusion matrix to stdout.

---

## 📦 Dataset

**IBM Telco Customer Churn** — available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the CSV at `content/WA_Fn-UseC_-Telco-Customer-Churn.csv` before running the training script.

**Features include:** tenure, contract type, internet service, monthly charges, total charges, payment method, and more (19 input features → binary churn label).

---

## 🔮 Future Improvements

- [ ] Integrate GCP Vertex AI for managed model hosting
- [ ] Add BigQuery for large-scale batch inference logging
- [ ] SHAP-based feature importance visualization in the UI
- [ ] CI/CD pipeline via GitHub Actions
- [ ] Model versioning and experiment tracking (MLflow / W&B)

---

## 👤 Author

**Sujin S P**  
B.Tech CSE (AI & ML) — SRM Institute of Science and Technology, Trichy  
📎 [GitHub](https://github.com/sujinjust4u)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
