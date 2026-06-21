# 🛒 Shopper Spectrum
### Customer Segmentation & Product Recommendations in E-Commerce

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://onlineretail-recommender-ronak029.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 🔗 Live App

**👉 [https://onlineretail-recommender-ronak029.streamlit.app/](https://onlineretail-recommender-ronak029.streamlit.app/)**

---

## Overview

This project analyses **500,000+ real transactions** from a UK-based online retail store to surface two core business insights:

1. **Customer Segmentation** — Groups customers by purchasing behavior using RFM analysis + KMeans clustering into four actionable segments: High-Value, Regular, Occasional, and At-Risk.
2. **Product Recommendations** — Recommends 5 similar products for any given item using item-based collaborative filtering with cosine similarity.

Both features are accessible through a live Streamlit web app — no code required.

---

## 📊 Dataset

**UCI Online Retail II** — Real transactions from a UK non-store online retailer (2009–2011)

| Source | Link |
|---|---|
| UCI ML Repository (primary) | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Kaggle Mirror | https://www.kaggle.com/datasets/mathchi/online-retail-ii-data-set-from-ml-repository |

**Dataset columns:**

| Column | Description |
|---|---|
| `InvoiceNo` | Transaction ID (prefix 'C' = cancelled) |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units per transaction |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit (£) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/ronakjha2002/shopper-spectrum.git
cd shopper-spectrum
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `online_retail.xlsx` from one of the dataset links above and place it in the project root (same folder as the notebook).

> The notebook loads it as: `df = pd.read_excel('online_retail.xlsx')`

### 4. Run the notebook end-to-end

Open `Shopper-Spectrum-Analysis-Download.ipynb` and run all cells. This generates the `models/` folder with all required `.pkl` files.

### 5. Launch the Streamlit app locally
```bash
streamlit run app/app.py
```

Or just use the live version: [onlineretail-recommender-ronak029.streamlit.app](https://onlineretail-recommender-ronak029.streamlit.app/)

---

## 📁 Project Structure

```
shopper-spectrum/
│
├── Shopper-Spectrum-Analysis-Download.ipynb   ← Full analysis pipeline
│
├── app/
│   └── app.py                  ← Streamlit web application
│
├── models/                     ← Generated after running the notebook
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── label_map.pkl
│   ├── similarity_matrix.pkl
│   └── product_list.pkl
│
├── online_retail.xlsx          ← Dataset (download separately, not tracked)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Methodology

### Part 1 — Customer Segmentation (RFM + KMeans)

**Step 1 — Feature Engineering:**

| Feature | How it's computed |
|---|---|
| Recency | `snapshot_date − customer's last purchase date` (in days) |
| Frequency | Number of unique invoices per customer |
| Monetary | `sum(Quantity × UnitPrice)` per customer |

**Step 2 — Preprocessing:**
- Dropped rows with missing `CustomerID`
- Removed cancelled invoices (InvoiceNo starting with `'C'`)
- Filtered out zero/negative `Quantity` and `UnitPrice`
- Normalized RFM values using `StandardScaler`

**Step 3 — Clustering:**
- Tested k = 2 to 10 using **Elbow Method** (WCSS) and **Silhouette Score**
- Final model: `KMeans(n_clusters=4, init='k-means++', n_init=15, random_state=42)`

**Step 4 — Segment Labels:**

| Segment | Recency | Frequency | Monetary | Marketing Strategy |
|---|---|---|---|---|
| 💎 High-Value | Low | High | High | Reward & retain — loyalty perks, early access |
| ✅ Regular | Medium | Medium | Medium | Upsell — bundles, personalised recommendations |
| 🕐 Occasional | Any | Low | Low | Re-engage — seasonal promos, flash sales |
| ⚠️ At-Risk | High | Low | Low | Win-back — discount campaigns, feedback surveys |

---

### Part 2 — Product Recommendation System (Collaborative Filtering)

1. Built a **Customer × Product** pivot table with purchase quantities (UK customers only)
2. Filtered to products with ≥ 10 transactions to reduce sparsity
3. Transposed to **Product × Customer** matrix
4. Computed **cosine similarity** between all product pairs
5. For a given product, returns the **top 5 most similar** products by score

Supports **partial name matching** — you don't need to type the exact product name.

---

## 🖥️ App Features

### 📦 Product Recommender
- Type any product name (partial match supported)
- Click **Get Recommendations**
- Returns 5 similar products with cosine similarity scores
- Product browser with live search filter

### 👥 Customer Segmentation
- Enter Recency (days), Frequency (orders), Monetary (£ spent)
- Click **Predict Segment**
- Returns segment label + color-coded badge + recommended marketing actions

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Transactions after cleaning | ~390,000+ |
| Unique customers | ~4,300+ |
| KMeans Silhouette Score | ~0.44 |
| Products in similarity matrix | ~3,500+ |
| Recommendation method | Item-based Cosine Similarity |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (KMeans, StandardScaler, cosine_similarity) |
| App | Streamlit |
| Serialization | Pickle |

