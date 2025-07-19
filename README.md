# 🧪 Logistic Regression Model Evaluation – Airbnb Listings

## 📌 Overview

In this lab, I worked on the **evaluation phase of the machine learning life cycle**, using **Logistic Regression** to solve a classification problem with the **Airbnb "listings" dataset**. I defined the problem, preprocessed the data, trained models, performed hyperparameter tuning, evaluated the models with various metrics, and saved my final model for future use.

---

## 🗂️ What I Did

1. **Built My DataFrame & Defined the ML Problem**

   * Loaded the Airbnb dataset.
   * Chose a **label** (target variable) to predict.
   * Identified and preprocessed the **features** I needed.
   * Created labeled examples and split the data into **training** and **test** sets.

2. **Trained & Evaluated a Baseline Model**

   * Trained a **Logistic Regression** model using default settings from `scikit-learn`.
   * Evaluated the model using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and the **confusion matrix**.

3. **Tuned Hyperparameters with Grid Search**

   * Used **GridSearchCV** to find the best value for **C**, the regularization strength.
   * Trained a new Logistic Regression model using the optimal `C` value.
   * Compared this model's performance against the baseline.

4. **Plotted Evaluation Curves**

   * Plotted the **Precision-Recall Curve** and the **ROC Curve** for both models.
   * Calculated and compared the **AUC** (Area Under the Curve) values.

5. **Selected Important Features**

   * Used techniques like **Recursive Feature Elimination (RFE)** and **SelectKBest** to choose the most important features.
   * Retrained and tested the model using only these features to see how performance changed.

6. **Saved the Final Model**

   * Made the model persistent using `joblib` so I can reuse it later for inference without retraining.

---

## 🛠️ Tools and Libraries I Used

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* joblib / pickle

---

## 🧪 How to Run This Lab

1. Install the necessary libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Open the notebook:

   ```bash
   jupyter notebook lab_logistic_regression.ipynb
   ```

3. Go through each step and run the cells to see the model in action.

---

## 📁 Files Included

* `lab_logistic_regression.ipynb` – My complete lab notebook
* `listings.csv` – The Airbnb dataset
* `logistic_model.pkl` – The saved logistic regression model
* `README.md` – This file

---

## ✅ What I Learned

By completing this lab, I learned how to:

* Frame a classification problem and prepare real-world data
* Train and evaluate logistic regression models
* Tune hyperparameters using grid search
* Visualize performance using precision-recall and ROC curves
* Perform feature selection to improve models
* Save models for future use
