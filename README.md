# 📘 Product Feature Usage Intelligence Dashboard

### *End-to-End Business Analyst Project (Python • Streamlit • RFM • KMeans • Plotly)*

---

# 📌 **1. Project Overview**

This project simulates a **real SaaS product analytics workflow**, where we analyze how users interact with different product features over time.

The system provides:

* Feature usage metrics
* User engagement analytics
* RFM-based behavioral scoring
* KMeans usage segmentation
* A polished multi-page Streamlit dashboard

This project is designed for **Business Analyst / Product Analyst / Data Analyst** roles and is fully runnable locally.

---

# 📌 **2. Problem Statement & Business Context**

Modern digital products generate large volumes of event-level data (clicks, searches, feature interactions).
However, companies often struggle with:

* Understanding which features are actually used
* Identifying “power users” vs “at-risk” users
* Measuring adoption and engagement
* Making data-driven product decisions

This project solves that by simulating:

* Feature-level adoption
* Usage trends
* RFM-based engagement scoring
* Usage-based user segmentation
* Monitoring active vs inactive users

Product Managers, Growth Teams, and Analysts use dashboards like this to:

* Prioritize the roadmap
* Improve user retention
* Identify adoption gaps
* Personalize engagement or marketing campaigns

---

# 📌 **3. Tech Stack**

**Programming & Analytics**

* Python 3.10+
* Pandas, NumPy
* scikit-learn (KMeans segmentation)
* Plotly (interactive visuals)

**App / UI**

* Streamlit (multi-page dashboard)

**Utilities**

* Joblib
* Pytest (for model testing)

---

# 📌 **4. Data Description**

Synthetic dataset simulates clickstream-like product usage:

### **Raw Data Columns**

| Column         | Description                                 |
| -------------- | ------------------------------------------- |
| `user_id`      | Unique user ID                              |
| `signup_date`  | When the user joined                        |
| `event_date`   | Date of feature usage                       |
| `feature_name` | Feature used (Search, Dashboard, API, etc.) |
| `events_count` | Number of actions for that feature/day      |

### **Processed Data Columns**

| Column            | Description                    |
| ----------------- | ------------------------------ |
| `last_event_date` | Last active day                |
| `active_days`     | Days user engaged with product |
| `total_events`    | Total feature interactions     |
| `recency`         | Days since last activity       |
| `frequency`       | Number of active days          |
| `monetary`        | Usage intensity score          |
| `cluster`         | KMeans behavioral segment      |

---

# 📌 **5. Architecture**

```
Synthetic Feature Usage Data
          ↓
Preprocessing (clean, transform, aggregate)
          ↓
RFM Feature Builder (recency / frequency / monetary)
          ↓
KMeans Segmentation Model (Power, Regular, At-Risk, Dormant)
          ↓
Streamlit Multi-Page Dashboard
          ↓
Insights for Product & Business Decision-Making
```

---

📁 Folder Structure

```
Product-Feature-Usage-Intelligence/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Feature_Usage.py
│       └── 3_RFM_Segments.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── kmeans_rfm.pkl
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── preprocess_data.py
│   ├── build_rfm.py
│   └── train_model.py
│
├── src/
│   ├── preprocessing.py
│   ├── rfm.py
│   └── viz.py
│
├── tests/
│   └── test_predict.py
│
├── screenshots/
│   ├── overview.png
│   ├── feature_usage.png
│   └── rfm_segments.png
│
├── .gitignore
├── requirements.txt
└── README.md
```
---

# 📌 **6. How to Run (Step-by-step)**

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/<your-username>/product_feature_intel.git
cd Product_Feature_Intel
```

### **2️⃣ Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

### **3️⃣ Install dependencies**

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### **4️⃣ Generate synthetic data**

```bash
python scripts/generate_synthetic_data.py
```

### **5️⃣ Preprocess / build features**

```bash
python scripts/preprocess_data.py
python scripts/build_rfm.py
```

### **6️⃣ Train KMeans model**

```bash
python scripts/train_model.py
```

### **7️⃣ Evaluate KMeans model**

```bash
python src/evaluate_model.py
```

### **8️⃣ Testing the project**

```bash
pytest -v
```

### **9️⃣ Start the dashboard**

```bash
streamlit run app/Home.py
```

Your dashboard opens at:
👉 [http://localhost:8501](http://localhost:8501)

---

# 📌 **7. Dashboard Walkthrough (with screenshots)**

### **🏠 Home Page**

Simple project overview and navigation.

---

### **📌 Overview Page**

Shows high-level KPIs:

* Total users
* Avg events per user
* Avg active days
* RFM summary per cluster


```
![Overview](screenshots/overview.png)
```

---

### **📈 Feature Usage Page**

Includes:

* Total usage per feature
* Interactive filters (date range, feature selection)
* Feature-wise time-series trends


```
![Feature Usage](screenshots/feature_usage.png)
```

---

### **🧮 RFM Segments Page**

* 3D RFM scatter plot (Recency–Frequency–Monetary)
* Cluster distribution bar chart
* Insightful cluster interpretation


```
![RFM Segments](screenshots/rfm_segments.png)
```

---

# 📌 **8. Key Insights & Results**

Some example insights (your numbers may differ):

### 🔹 Feature Usage

* "Dashboard" is the most used feature.
* API & Integrations have lower adoption (good candidate for UX improvement).

### 🔹 Engagement (RFM)

* Power users show low recency and high frequency/monetary.
* At-risk users show high recency with declining frequency.

### 🔹 Segments

Model successfully separates users into:

* **Cluster 0 – Power Users**
* **Cluster 1 – Regular Users**
* **Cluster 2 – At-Risk Users**
* **Cluster 3 – Dormant Users**

These are directly usable for:

* Re-engagement campaigns
* Feature onboarding
* Product roadmap planning

---

# 📌 **9. Future Work**

To expand this into a more advanced product analytics suite:

* 📉 **Churn prediction model**
* 🧩 **Cohort retention heatmaps**
* ⚡ Real-time usage ingestion (Kafka → DB → Dashboard)
* 📊 Feature correlation matrix (which features drive stickiness?)
* 🧭 User journey funnel visualization
* 🚀 Deploy dashboard to Streamlit Cloud / Render / AWS