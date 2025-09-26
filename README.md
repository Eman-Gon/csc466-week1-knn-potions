# KNN Recommender (CSC-466 Week 1)

This project builds a simple **K-Nearest Neighbors (KNN) recommender system** using real data from Potions.gg.  
It is the first assignment for CSC-466 (Knowledge Discovery).

---

## Files
- **knn.py** – Python script that builds the KNN recommender and generates recommendations.  
- **eval.csv** – Output file with 3 adventurers and their top-2 recommended items.  
- **writeup.md** – Reflection write-up answering the assignment questions.  
- **data/** – Folder containing the input `.parquet` files provided (not included in repo).  

---

## Requirements
Only the following libraries are used (per assignment rules):
- pandas  
- numpy  
- scikit-learn  
- pyarrow or fastparquet (for parquet file reading)  

---

## How to Run
1. Place the data files in a folder called `data/`:
   - `content_views.parquet`  
   - `subscriptions.parquet`  
   - (others are optional, not required here)  

2. Run the recommender:
   ```bash
   python3 knn.py
