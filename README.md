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

### References
- [19] Karadeniz, T., Tokdemir, G. & Maraş, H.H. Ensemble Methods for Heart Disease Prediction. New Gener. Comput. 39, 569–581 (2021). [https://doi.org/10.1007/s00354-021-00124-4](https://doi.org/10.1007/s00354-021-00124-4)
- [20] Abdulwahab Ali Almazroi. Survival prediction among heart patients using machine learning techniques. Mathematical Biosciences and Engineering, 2022, 19(1): 134-145. [doi: 10.3934/mbe.2022007](http://doi.org/10.3934/mbe.2022007)
- [25] Bhat, N. Health care: Heart attack possibility, 2020. [https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility](https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility)
- [26] Chicco, D. & Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med. Inform. Decis. Mak. 20, 1–16 (2020).
- [27] Mahgoub, A. Diabetes prediction system with KNN algorithm, 2021. [https://www.kaggle.com/abdallamahgoub/diabetes](https://www.kaggle.com/abdallamahgoub/diabetes)
- [28] Soundarapandian, P. Chronic_Kidney_Disease Data Set, 2015. [https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
- [29] Smith, J. W., Everhart, J. E., Dickson, W., Knowler, W. C. & Johannes, R. S. In Proceedings of the Annual Symposium on Computer Application in Medical Care. 261 (American Medical Informatics Association) (2011).
- [30] Suwal, M. S. Breast Cancer Prediction Dataset, [https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset](https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset) (2018).


## Accuracy of Each Algorithm on Each Dataset

| ID  | KNN     | KNN-DT  | KNN-NB  | KNN-RF  | KNN-SVM | Hassanat-KNN |
| --- | ------- | ------- | ------- | ------- | ------- | ------------ |
| D1  | 91.80%  | 90.8%   | 88.52%  | 91.8%   | 91.80%  | 90.16%       |
| D2  | 73.33%  | 76.66%  | 73.33%  | 76.7%   | 76.67%  | 71.66%       |
| D3  | 76.62%  | 80.51%  | 77.27%  | 82.47%  | 79.22%  | 81.81%       |
| D4  | 98.53%  | 98.53%  | 98.53%  | 98.53%  | 98.53%  | 98.53%       |
| D5  | 98.75%  | 100%    | 98.75%  | 100%    | 100%    | 98.75%       |
| D6  | 98.75%  | 98.75%  | 98.75%  | 98.75%  | 98.75%  | 98.75%       |
| D7  | 76.62%  | 80.51%  | 77.27%  | 82.47%  | 79.22%  | 81.81%       |
| D8  | 98.24%  | 96.49%  | 94.73%  | 98.25%  | 96.49%  | 97.3%        |
| D9  | 92.59%  | 90.74%  | 90.74%  | 92.59%  | 92.59%  | 88%          |
| D10 | 83.33%  | 83.33%  | 80%     | 86.66%  | 83.33%  | 80%          |
| Avg | 88.85%  | 89.63%  | 87.78%  | 90.82%  | 89.66%  | 88.67%       |

## Precision of Each Algorithm on Each Dataset

| ID  | KNN     | KNN-DT  | KNN-NB  | KNN-RF  | KNN-SVM | Hassanat-KNN |
| --- | ------- | ------- | ------- | ------- | ------- | ------------ |
| D1  | 93.54%  | 93.54%  | 90.32%  | 93.5%   | 93.54%  | 90.62%       |
| D2  | 90.90%  | 86.66%  | 80.00%  | 92.3%   | 92.3%   | 83.33%       |
| D3  | 74.35%  | 74.5%   | 70.00%  | 83.33%  | 74.46%  | 78.72%       |
| D4  | 100%    | 100%    | 100.%   | 100%    | 100%    | 100%         |
| D5  | 100%    | 100%    | 100%    | 100%    | 100%    | 100%         |
| D6  | 100%    | 100%    | 100%    | 100%    | 100%    | 100%         |
| D7  | 74.35%  | 74.5%   | 70.00%  | 83.33%  | 74.46%  | 78.72%       |
| D8  | 97.26%  | 95.89%  | 92.22%  | 97.26%  | 95.89%  | 100%         |
| D9  | 89.19%  | 86.84%  | 88.88%  | 89.19%  | 89.19%  | 86.48%       |
| D10 | 92.30%  | 92.30%  | 68%     | 93.33%  | 82.35%  | 90.90%       |
| Avg | 91.18%  | 90.42%  | 85.94%  | 93.22%  | 90.21%  | 90.87%       |

## Recall of Each Algorithm on Each Dataset

| ID  | KNN     | KNN-DT  | KNN-NB  | KNN-RF  | KNN-SVM | Hassanat-KNN |
| --- | ------- | ------- | ------- | ------- | ------- | ------------ |
| D1  | 90.62%  | 90.62%  | 87.5%   | 90.62%  | 90.62%  | 90.62%       |
| D2  | 40%     | 52%     | 48%     | 48%     | 48%     | 40%          |
| D3  | 52.72%  | 69.09%  | 63.63%  | 63.64%  | 63.63%  | 67.27%       |
| D4  | 97.08%  | 97.08%  | 97.08%  | 97.08%  | 97.08%  | 97.08%       |
| D5  | 98.07%  | 100%    | 98.07%  | 100%    | 100%    | 98.07%       |
| D6  | 98.07%  | 98.07%  | 98.07%  | 98.07%  | 98.07%  | 98.07%       |
| D7  | 52.72%  | 69.09%  | 63.63%  | 63.64%  | 63.63%  | 67.72%       |
| D8  | 100%    | 98.59%  | 100%    | 100%    | 98.59%  | 95.77%       |
| D9  | 100%    | 100%    | 96.96%  | 100%    | 100%    | 96.96%       |
| D10 | 57.14%  | 57.14%  | 80.95%  | 66.66%  | 66.66%  | 47.61%       |
| Avg | 78.64%  | 83.16%  | 83.38%  | 82.77%  | 82.62%  | 79.91%       |

## Conclusion

This study concludes that the KNN-Random Forest algorithm stands out as the optimal choice for disease prediction, offering a reliable and effective solution for assessing illness risk. Its robust recall rates, precision, and accuracy demonstrate its potential to accurately identify individuals at risk of disease. Further validation of the KNN-RF algorithm's performance in illness risk prediction using larger datasets and diverse demographics is recommended. Additionally, exploring preprocessing data techniques to address dataset imbalance issues and investigating the interaction between classifiers and voting styles could enhance the algorithm's dependability and efficiency.

Future research should focus on refining the performance of the KNN ensemble technique and its variants by exploring new methods for parameter tuning and data processing techniques. Evaluating different types of classifiers for ensemble combinations, investigating hard-voting techniques, and exploring other ensemble approaches' performance can provide valuable insights into the KNN ensemble approach's potential for disease risk prediction.

This study represents a preliminary step in recognizing the KNN-Random Forest algorithm as a viable option for disease risk prediction tasks. Subsequent research can build upon these findings to advance and broaden the utilization of ensemble approach techniques, contributing to a deeper understanding of their significance in healthcare decision-making.

