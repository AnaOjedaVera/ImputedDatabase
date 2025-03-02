# AsteroidImputationClassify

This repository contains Jupyter notebooks and data used for assessing machine learning models for a data imputation method in asteroid taxonomic classification.

## Project Overview

Missing data is a prevalent issue in many astrophysics studies and poses a significant challenge when applying machine learning techniques. Specifically, missing data in asteroid databases directly affects the accuracy of taxonomic classification. This study proposes an imputation method to address missing data issues and evaluates its effectiveness across several machine learning models, including Gaussian Naive Bayes, Multilayer Perceptron, Multinomial Logistic Regression, Random Forest, and Support Vector Machine. The models were tested on two datasets: one using only spectral data and another that included albedo as an additional feature. Accuracy and balanced accuracy (BAcc) were used to evaluate the models, with results ranging from 59% to 90%. The inclusion of albedo significantly improved model performance, particularly for the Random Forest model, which showed a 23% increase in BAcc. Multilayer Perceptron, and Support Vector Machine demonstrated the most robust performance, balancing bias-variance trade-off and achieving up to 90% accuracy, when albedo was included, whereas Gaussian Naive Bayes performed the worst. These findings demonstrate the potential of the proposed imputation method for achieving high classification accuracy of new unclassified asteroids within asteroid databases.

## Repository Contents

* `imputed_asteroid_data.csv`: The imputed asteroid dataset used for classification.
* `data_preparation.ipynb`: Notebook for data preprocessing.
* `imputation_methods.ipynb`: Notebook detailing the imputation methods used.
* `classification_methods.ipynb`: Notebook containing the machine learning model training and evaluation.

## Original Data Source

The original dataset used in this study is from Mahlke's repository: [https://github.com/maxmahlke/classy/tree/main/classy/data/classy](https://github.com/maxmahlke/classy/tree/main/classy/data/classy)

## Findings

### Results

We evaluated the performance of multiple machine learning models in classifying taxonomic classes, using a dataset with imputed data. To determine the effect of albedo, we performed two sets of experiments. The first set, detailed in Table 1, excluded albedo from the feature set. The second set, presented in Table 2, included albedo. For both experiments, we report the test accuracy, 95% confidence intervals, as well as the mean and standard deviation of the 10-fold cross-validation results of the four evaluation metrics.

Test accuracy provides a direct measure of how well our trained model performs on a single, independent test dataset, giving us a snapshot of its generalization ability. However, because this is a single evaluation, it can be sensitive to the specific characteristics of the test data. To account for this variability and assess the reliability of our test accuracy, we calculated a 95% confidence interval, which provides a range within which the true model accuracy is likely to fall.

Additionally, we employed 10-fold cross-validation, which divides the training data into ten subsets for iterative training and testing. This method yields a more robust estimate of the model's average performance and stability, as reflected by the mean and standard deviation of the evaluation metrics across the ten folds. By reporting both test accuracy with confidence intervals and cross-validation results, we offer a comprehensive evaluation, combining independent assessment with robust stability analysis.

**Table 1: Test classification results for taxonomic classes using multiple models, trained on a dataset with imputed data and excluding the albedo feature. Reported metrics include test accuracy with 95% confidence intervals, and the mean and standard deviation of 10-fold cross-validation runs.**

| Model | Test Accuracy | 95% CI | Accuracy (10-fold CV) | BAcc (10-fold CV) | F1 (10-fold CV) | MCC (10-fold CV) |
|---|---|---|---|---|---|---|
| GNB | 0.6497 | 0.6111-0.6882 | 0.6471±0.0034 | 0.6209±0.006 | 0.6597±0.0033 | 0.5849±0.0037 |
| MLP | 0.8112 | 0.7796-0.8429 | 0.8247±0.0032 | 0.7088±0.0091 | 0.8224±0.0033 | 0.7801±0.0041 |
| MLR | 0.8197 | 0.7887-0.8508 | 0.829±0.0024 | 0.7282±0.0067 | 0.8264±0.0024 | 0.7852±0.003 |
| RF | 0.7772 | 0.7436-0.8108 | 0.7914±0.0034 | 0.6094±0.0071 | 0.7806±0.0038 | 0.7356±0.0045 |
| SVM | 0.8452 | 0.8160-0.8745 | 0.8315±0.002 | 0.7217±0.0081 | 0.8292±0.0021 | 0.7888±0.0025 |

**Table 2: Test classification results for taxonomic classes using multiple models, trained on a dataset with imputed data and including the albedo feature. Reported metrics include test accuracy with 95% confidence intervals, and the mean and standard deviation of 10-fold cross-validation runs.**

| Model | Test Accuracy | 95% CI | Accuracy (10-fold CV) | BAcc (10-fold CV) | F1 (10-fold CV) | MCC (10-fold CV) |
|---|---|---|---|---|---|---|
| GNB | 0.6888 | 0.6514-0.7262 | 0.6865±0.0036 | 0.6712±0.0069 | 0.698±0.0033 | 0.63±0.0039 |
| MLP | 0.8673 | 0.8399-0.8949 | 0.8722±0.0055 | 0.7846±0.0109 | 0.871±0.0055 | 0.8399±0.007 |
| MLR | 0.8724 | 0.8455-0.8994 | 0.8746±0.0025 | 0.7697±0.0048 | 0.8718±0.0026 | 0.8422±0.0032 |
| RF | 0.8452 | 0.8160-0.8745 | 0.839±0.0041 | 0.691±0.009 | 0.8309±0.0046 | 0.7967±0.0052 |
| SVM | 0.8724 | 0.8455-0.8994 | 0.8818±0.0033 | 0.7922±0.009 | 0.8799±0.0032 | 0.8516±0.004 |

**Table 3: Optimal hyperparameter configurations resulting from grid search for each model.**

| Best Parameters | Without Albedo | With Albedo |
|---|---|---|
| GB | Variance smoothing: 1e-12 | Variance smoothing: 1e-12 |
| MLP | Hidden layes size: (32, 32), Learning rate: 0.01, Max iterations: 1000, Solver: sgd | Hidden layes size: (32, 32, 32), Learning rate: 0.01, Max iterations: 1000, Solver: sgd |
| MLR | C value: 20, Max iterations: 10000, Penalty: l1, Solver: Saga | C value: 1.6237, Max iterations: 10000, Penalty: l
