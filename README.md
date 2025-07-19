# 🧪 Logistic Regression Model Evaluation - Airbnb Listings

## 📌 Overview

This lab focuses on the **evaluation phase of the machine learning life cycle**, applying **Logistic Regression** to a real-world classification problem using the **Airbnb "listings" dataset**. You'll define your ML problem, preprocess the data, train baseline and tuned models, evaluate performance, and make the model persistent for future use.

---

## 🗂️ Lab Objectives

1. **Build your DataFrame & Define the ML problem**

   * Load the Airbnb dataset.
   * Define the **label** (target variable).
   * Identify and preprocess **features**.
   * Create labeled examples.
   * Split the dataset into **training** and **test** sets.

2. **Train & Evaluate a Baseline Model**

   * Train a **Logistic Regression** model using default `scikit-learn` hyperparameters.
   * Evaluate performance using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

3. **Hyperparameter Tuning with Grid Search**

   * Perform **GridSearchCV** to find the optimal value of **C** (regularization strength).
   * Re-train the Logistic Regression model using the optimal `C`.
   * Re-evaluate and compare to the baseline model.

4. **Plot Evaluation Curves**

   * Plot the **Precision-Recall Curve** for both models.
   * Plot the **ROC Curve** and compute **AUC** for both models.

5. **Feature Selection**

   * Use methods such as **Recursive Feature Elimination (RFE)** or **SelectKBest** to identify important features.
   * Retrain the model using the selected features and compare performance.

6. **Model Persistence**

   * Save the trained model using `joblib` or `pickle` for future inference.

---

## 🛠️ Technologies Used

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* joblib / pickle

---

## 🧪 How to Run

1. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Run the Jupyter Notebook:

   ```bash
   jupyter notebook lab_logistic_regression.ipynb
   ```

3. Follow the step-by-step tasks in the notebook.

---

## 📊 Sample Outputs

* Confusion matrix for baseline vs tuned models
* Precision-Recall and ROC curves with AUC
* Summary of feature importances
* Saved `.pkl` model file for future use

---

## 📁 Files

* `lab_logistic_regression.ipynb` — Main notebook
* `listings.csv` — Airbnb dataset
* `logistic_model.pkl` — Saved model
* `README.md` — Lab description

---

## ✅ Learning Outcomes

By the end of this lab, you will be able to:

* Define a classification problem and prepare real-world data.
* Train and evaluate logistic regression models.
* Use grid search for hyperparameter tuning.
* Visualize model evaluation metrics.
* Perform feature selection and persist ML models.
