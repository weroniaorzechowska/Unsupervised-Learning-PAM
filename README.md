# Partitioning Around Medoids (PAM) Clustering Algorithm 📊

## Overview 📌
This repository contains an **implementation of the Partitioning Around Medoids (PAM) algorithm**, a robust clustering method used in **unsupervised learning**. Unlike k-means, which minimizes within-cluster variance using centroids, PAM clusters data by selecting actual **medoids**, making it more resistant to outliers and noise.

## Objectives 🎯
- Implement the **PAM clustering algorithm** from scratch in Python.
- Compare **PAM vs. k-means** in terms of robustness and efficiency.
- Apply the algorithm to **real-world datasets** for clustering analysis.

## Key Features 🚀
- **Medoid-based clustering** – minimizes dissimilarities instead of using centroids.
- **Robust to outliers** – unlike k-means, PAM selects real data points as cluster centers.
- **Custom distance metrics** – supports Euclidean, Manhattan, and other distance measures.
- **Flexible implementation** – allows parameter tuning for optimal clustering results.

## Algorithm Overview 🛠️
### **1️⃣ Initialization**
- Select `k` medoids randomly from the dataset.

### **2️⃣ Iterative Assignment & Optimization**
- Assign each data point to the closest medoid based on a distance metric.
- Swap medoids with non-medoids if it reduces the total dissimilarity within clusters.

### **3️⃣ Convergence & Output**
- Repeat the optimization step until medoids stabilize.
- Return final cluster assignments and medoids.


## Key Insights 🔎
✅ **PAM is more robust to outliers** than k-means.  
✅ **Medoids represent actual data points**, making interpretations more intuitive.  
✅ **Computationally expensive for large datasets**, but efficient for smaller applications.  

## Future Enhancements 🏗️
- Optimize the algorithm for **large-scale datasets**.
- Implement **faster alternatives** such as CLARA (Clustering Large Applications).
- Add **visualization tools** to analyze clustering quality.

## Contribution 🤝
Contributions are welcome! Feel free to submit **pull requests**, report issues, or suggest new features.

## License 📄
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---
### 🚀 Exploring robust clustering techniques with PAM!

