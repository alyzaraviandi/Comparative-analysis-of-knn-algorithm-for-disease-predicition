# Comparative Analysis of KNN-Based Algorithms for Disease Prediction Using Ensemble Soft Voting Method

## Purpose
The objective of this study is to conduct a thorough comparison of ensemble approach K-Nearest Neighbors (KNN) variations in disease prediction. No previous research has precisely examined the performance variations across several soft voting ensemble approach variants of KNN. The major goal is to identify the best performing soft voting ensemble approach KNN for disease prediction tasks.

## Methods/Study Design/Approach
Datasets used were from previous studies, with features normalized. Parameters of all algorithms were optimized using grid search. Accuracy, precision, and recall metrics were used for evaluation. Experiments were conducted using Python, and relevant modules and libraries were utilized.

## Results/Findings
KNN-Random Forest outperformed other soft voting ensemble method KNN variants considered, achieving the highest accuracy (90.82%) and precision (93.22%). KNN-Naïve Bayes had the highest average recall score. Identifying KNN-Random Forest as the best performing variant assists researchers and practitioners in algorithm selection for disease prediction tasks.

## Novelty/Originality/Value
This study provides valuable insights into algorithm selection for disease risk prediction. By highlighting the capabilities of soft voting ensemble approach KNN and identifying KNN-Random Forest as the best performer, it aids informed decision-making. Future research could explore alternative voting methods and ways to enhance KNN ensemble approach performance.

## Datasets Used

| ID  | Datasets                              | Features | Data Size | References                                                                                         |
| --- | ------------------------------------- | -------- | --------- | -------------------------------------------------------------------------------------------------- |
| D1  | Heart Attack Possibilities            | 13       | 303       | [25]                                                                                               |
| D2  | Heart Failure Outcomes                | 12       | 299       | [26]                                                                                               |
| D3  | Diabetes                              | 8        | 768       | [27]                                                                                               |
| D4  | Heart Disease Prediction              | 13       | 270       | [25]                                                                                               |
| D5  | Chronic Kidney Disease Preprocessed   | 24       | 400       | [28]                                                                                               |
| D6  | Chronic Kidney Disease Prediction     | 13       | 400       | [28]                                                                                               |
| D7  | Pima Indians Diabetes                 | 8        | 767       | [29]                                                                                               |
| D8  | Breast Cancer                         | 5        | 596       | [30]                                                                                               |
| D9  | Statlog Heart Disease                 | 13       | 596       | [19]                                                                                               |
| D10 | Cardio Vascular Disease               | 12       | 596       | [20]                                                                                               |
