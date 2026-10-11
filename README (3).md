# Customer Segmentation with DBSCAN

Grouping 2,212 retail customers into segments with **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), tuning its parameters properly, and explaining what each segment and the "noise" points mean for marketing.

![DBSCAN clusters](images/dbscan_clusters_pca.png)

---

## Project Overview

Most clustering methods need you to say how many groups you want. DBSCAN doesn't. It looks for **crowded areas** in the data and labels customers that don't belong to any crowded area as **noise**. That makes it a good fit for customer data, where we don't know the number of segments in advance and unusual customers shouldn't distort the groups.

The project covers:

1. Data cleaning and feature engineering
2. Exploratory data analysis (histograms, boxplots, scatter plots, correlation heatmap)
3. Choosing `eps` and `min_samples` (k-distance graph + grid search)
4. Fitting DBSCAN and evaluating it (silhouette score, Davies-Bouldin index, core / border / noise counts)
5. Profiling each cluster and analysing the noise points
6. Comparing the results with the EDA findings

## Repository Structure

```
dbscan-customer-segmentation/
├── DBSCAN_Marketing_Campaign.ipynb   # full analysis with outputs and explanations
├── dbscan_clustering.py              # short script version of the final model
├── data/
│   └── marketing_campaign.csv        # dataset (tab separated)
├── images/                           # key charts used in this README
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

`marketing_campaign.csv` has 2,240 customers and 29 columns. Despite the `.csv` extension, the file is **tab separated** (`sep="\t"`).

| Group | Columns |
|---|---|
| Customer info | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, `Complain` |
| Spending (last 2 years) | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase channels | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Campaign response | `AcceptedCmp1` – `AcceptedCmp5`, `Response` (last campaign) |
| Constant (dropped) | `Z_CostContact`, `Z_Revenue` |

**Cleaning steps**
- Dropped 24 rows with missing `Income`.
- Dropped 3 rows with birth years 1893–1900 and 1 row with an income of 666,666 (placeholder value).
- Dropped the two constant columns.

**Engineered features:** `Age` (relative to 2014, when the data was collected), `Total_Spent`, `Children`, `Total_Purchases`, `Total_Accepted`.

## Method

**Features used for clustering (7, standardised):**
`Income`, `Total_Spent`, `Total_Purchases`, `NumWebVisitsMonth`, `NumDealsPurchases`, `Children`, `Age`

Campaign columns were deliberately **left out** of the clustering and used afterwards as an independent check of whether the clusters behave differently.

**Why scaling matters:** DBSCAN measures distances, so without scaling, `Income` (tens of thousands) would completely outweigh `Children` (0–3).

**Parameter selection**

- `min_samples = 14`, from the rule of thumb of 2 × number of features.
- **k-distance graph:** the elbow suggests `eps ≈ 1.61`, but at that value almost all customers merge into a single cluster. The elbow marks where outliers begin, not where segments separate.

![k-distance graph](images/k_distance_graph.png)

- **Grid search:** I tested 52 combinations (eps from 0.5 to 1.7, min_samples of 5, 10, 14 and 20). Selection rule: **at least 3 clusters, at most 20% noise, then highest silhouette**.

![Grid search](images/parameter_grid_search.png)

**Final model:** `DBSCAN(eps=1.0, min_samples=14)`

## Results

| Metric | Value |
|---|---|
| Clusters | 4 |
| Noise points | 365 (16.5%) |
| Silhouette score (noise excluded) | 0.172 |
| Davies-Bouldin index (noise excluded) | 1.654 |
| Core / Border / Noise | 65.6% / 17.9% / 16.5% |

### Customer Segments

| Cluster | Size | Profile | Avg income | Avg spend | Response rate |
|---|---|---|---|---|---|
| **2 – Affluent, no children** | 406 (18.4%) | Biggest spenders, rarely use deals or the website | ~77.4k | ~1,401 | **28.8%** |
| **1 – One-child mainstream** | 1,013 (45.8%) | Typical middle-income customer | ~45.3k | ~401 | 9.4% |
| **0 – Two-children budget families** | 334 (15.1%) | Low spend, most deal-driven (3 deals on average) | ~40.7k | ~130 | 11.1% |
| **3 – Young, low income, no children** | 94 (4.2%) | Youngest group, highest share of undergraduate education | ~24.9k | ~101 | 9.6% |
| **Noise (-1)** | 365 (16.5%) | 3-children families, extreme deal hunters, a few suspicious records | ~59.3k | ~865 | 20.5% |

![Clusters on income vs spending](images/clusters_income_vs_spent.png)

### Key Insights

- **Household size and income drive the segmentation.** Spending drops sharply with each child, while deal usage and website visits go up.
- **Cluster 2 is the most valuable segment.** It spends about 3.5× the one-child group and responds to campaigns about 3× as often. Campaigns 1 and 5 were accepted almost only by this group (21.7% and 28.3%, compared with about 0–1% elsewhere).
- **Family clusters are deal-driven**, so online promotions and bundles fit them better than premium mailings.
- **Noise is informative, not junk.** All 50 customers with 3 children fall into noise because there are too few of them to form a dense group. Noise also has an above-average response rate, so it should be reviewed rather than deleted.
- The clusters agree with the EDA. Income and spending have a correlation of 0.79, children and spending -0.50, and income and web visits -0.65.

![Spending by cluster](images/spending_by_cluster.png)

### Parameter Sensitivity

A small change in `eps` changes the result completely, which is why tuning matters so much with DBSCAN:

![eps sensitivity](images/eps_sensitivity.png)

| eps | Clusters | Noise |
|---|---|---|
| 0.8 | 5 | 808 |
| 1.0 | 4 | 365 |
| 1.4 | 1 | 54 |

## Limitations

- **Soft borders between segments.** The silhouette score is modest because the customers form one continuous cloud rather than well-separated groups.
- **The `Children` feature shapes the clusters strongly.** Because it is a whole number, customers with the same value stack into dense layers, and each cluster ends up containing a single child count.
- **One eps for the whole dataset.** DBSCAN applies a single eps everywhere, so it struggles when some segments are naturally denser than others. HDBSCAN would be a good next step.
- **The 2D plots lose information.** They use PCA and show only 68% of the variance, so clusters that overlap in the plot can still be separate in 7 dimensions.

## How to Run

```bash
# 1. clone the repository
git clone https://github.com/<your-username>/dbscan-customer-segmentation.git
cd dbscan-customer-segmentation

# 2. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4a. open the full analysis
jupyter notebook DBSCAN_Marketing_Campaign.ipynb

# 4b. or run the quick script
python dbscan_clustering.py
python dbscan_clustering.py --eps 1.1 --min_samples 10   # try other parameters
```

The script prints the scores and cluster profile and saves each customer's label to `outputs/customer_clusters.csv`.

## Tools

Python 3 · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter

## License

This project is released under the [MIT License](LICENSE).
