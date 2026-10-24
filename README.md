# Pima Indians Diabetes Prediction using Logistic Regression

**Student:** Rhydham Shukla  
**Course:** Machine Learning Program, Unlox Academy  
**Project:** Minor Project 2  

## Project Overview
This project develops an end to end supervised binary classification pipeline to predict diabetes onset using the Kaggle Pima Indians Diabetes dataset (768 patient records).

The focus of this analysis is on sound biomedical data preprocessing, preventing data leakage, and systematically analyzing classification decision thresholds to optimize clinical recall.

## Key Model Results
* **Default Threshold (0.50):**
  * Accuracy: 0.734 (73.4%)
  * Recall: 0.704 (38 of 54 diabetic patients identified)
  * Precision: 0.603
  * ROC-AUC Score: 0.813
  * False Negatives: 16 undiagnosed patients

* **Clinically Tuned Threshold (0.38):**
  * Accuracy: 0.714
  * Recall: 0.870 (47 of 54 diabetic patients identified)
  * Precision: 0.573
  * ROC-AUC Score: 0.813
  * False Negatives: Reduced to 7 undiagnosed patients (56% decrease)

## Why Threshold Tuning Matters in Medicine
In clinical diagnostics, a False Negative is substantially more hazardous than a False Positive. A False Negative means an undiagnosed diabetic individual receives no medical intervention, risking severe long term vascular and renal complications. Conversely, a False Positive only requires a follow up confirmatory test.

By adjusting the decision threshold from 0.50 to 0.38, our model sensitivity jumps from 70.4% to 87.0%, successfully detecting 9 additional patients who would have otherwise been missed.

## Anti-Leakage Preprocessing
1. **Handling Zero Values:**
   * In clinical testing, 0 values for Glucose, BloodPressure, SkinThickness, Insulin, and BMI represent missing tests rather than true physiological values.
   * These 0 entries were converted to NaN before splitting.
2. **Strict Training Pipeline:**
   * SimpleImputer (median strategy) and StandardScaler were fitted solely on the 80% training split (614 patients) and transformed onto the 20% test split (154 patients) to guarantee zero data leakage.
3. **Class Weighting:**
   * Trained with class_weight='balanced' to handle the 65/35 negative to positive class imbalance.

## Repository Structure
* `Pima_Diabetes_Logistic_Regression.ipynb`: Fully executed Jupyter notebook with data auditing, training, and threshold plots.
* `Rhydham_Shukla_Minor_Project_2_Report.pdf`: One page summary report.
* `diabetes.csv`: Pima Indians Diabetes dataset.
* `requirements.txt`: Python package dependencies.

## How to Run
1. Clone this repository:
   git clone https://github.com/rhydhamshukla/Pima-Diabetes-Logistic-Regression.git
2. Install dependencies:
   pip install -r requirements.txt
3. Open the notebook:
   jupyter notebook Pima_Diabetes_Logistic_Regression.ipynb
