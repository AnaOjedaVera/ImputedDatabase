# AsteroidImputationClassify

This repository contains Jupyter notebooks and data used for assessing machine learning models for a data imputation method in asteroid taxonomic classification. Also, the final imputed database as well as the final results in pdfs.  

## Project Overview

Missing data is a prevalent issue in many astrophysics studies and poses a significant challenge when applying machine learning techniques. Specifically, missing data in asteroid databases directly affects the accuracy of taxonomic classification. This study proposes an imputation method, leveraging a weighted mean of nearest neighbors based on a combined error metric that balances global similarity and local shape preservation, to address missing data issues. We evaluated its effectiveness across 5 machine learning models; Gaussian Naive Bayes, Multilayer Perceptron, Multinomial Logistic Regression, Random Forest and Support Vector Machine. The models were tested on two datasets: one using only spectral data and another that included albedo as an additional feature. Accuracy (Acc), balanced accuracy, F1 score and Matthews correlation coefficient were used to evaluate the models in 10-fold cross-validation runs and Acc was used to evaluate the models in test accuracy, with results ranging from 63.10% to 88.10%. The inclusion of albedo significantly improved model performance, particularly for the Random Forest model, which showed a 3.4%increase in test Acc. Support VectorMachine demonstrated the most robust performance, achieving up to 88.10% test Acc, when albedo was included, whereas Gaussian Naive Bayes performed the worst. These findings demonstrate the potential of the proposed imputation method for achieving high classification accuracy of new unclassified asteroids within asteroid databases.


## Repository Contents

* `ImputedDatabase.csv`: The imputed asteroid dataset used for classification. The first line contains column names representing spectral coverage in wavelength from 0.45 to 2.45, pV (albedo), name, counts, class_bdm, and class_asteroid_sf. The second line and subsequent lines contain the samples.
* `01 Data preparation - NewPreparation.ipynb`: Notebook for data preprocessing in Python.
* `02 Imputation methods-NewPreparation.ipynb`: Notebook detailing the imputation methods used in Python.
* `03 Classification methods-NewPreparation.ipynb`: Notebook containing the machine learning model training and evaluation in Python.
* `all_imputedNewNew.pdf`: The plots from the imputed asteroid dataset derived from the method. Each plot is titled "Target #: Left or Right imputation", where # represents the sample number (e.g., Target 0 for the first sample in the dataframe).
* `ImputedDatasetWithoutClasses.pdf`: The plots from the imputed asteroid dataset derived from the method, but without considering classes in the filling process. Each plot is titled "Target #: Left or Right imputation", where # represents the sample number.
* `pV_plots.pdf`: The plots from the imputed albedos of the asteroids derived from the method.
* `ConfusionMatrices.pdf`: Confusion matrices derived from the machine learning models with the method implemented.


## Original Data Source

The original dataset used in this study is from Mahlke's repository: [https://github.com/maxmahlke/classy/tree/main/classy/data/classy](https://github.com/maxmahlke/classy/tree/main/classy/data/classy)

## Findings

### Results

We evaluated the performance of 5 machine learning models in classifying taxonomic classes using a dataset with imputed data, derived from the method proposed. The taxonomic classification followed the scheme proposed by Mahlke et al. [Mahlke2022]. To determine the effect of albedo, we performed two sets of experiments. The first set, detailed in Table 1, excluded albedo from the feature set. The second set, presented in Table 2, included albedo. As explained before, for both experiments, we report the test accuracy, 95% confidence intervals, as well as the mean and standard deviation of the 10-fold cross-validation results of the four evaluation metrics.

**Table 1: Test classification results for taxonomic classes using multiple models, trained on a dataset with imputed data and excluding the albedo feature. Reported metrics include test accuracy with 95% confidence intervals, and the mean and standard deviation of 10-fold cross-validation runs.**

| Model | Test Accuracy | 95% CI | Accuracy | BAcc | F1 | MCC |
|---|---|---|---|---|---|---|
| GNB | 0.6310 | 0.5919-0.6700 | 0.6330$\pm$0.0025 | 0.6579$\pm$0.0050 | 0.6586$\pm$0.0049 | 0.5807$\pm$0.0051 |
| MLP | 0.8486 | 0.8197-0.8776 | 0.8537$\pm$0.0039 | 0.7737$\pm$0.0063 | 0.8528$\pm$0.0040 | 0.8167$\pm$0.0049 |
| MLR | 0.8452 | 0.8160-0.8745 | 0.8440$\pm$0.0019 | 0.7477$\pm$0.0056 | 0.8413$\pm$0.0017 | 0.8038$\pm$0.0024 |
| RF | 0.8010 | 0.7688-0.8333 | 0.8076$\pm$0.0016 | 0.6330$\pm$0.0027 | 0.7991$\pm$0.0018 | 0.7561$\pm$0.0021 |
| SVM | 0.8537 | 0.8252-0.8823 | 0.8622$\pm$0.0029 | 0.7816$\pm$0.0072 | 0.8609$\pm$0.0031 | 0.8274$\pm$0.0037 |

**Table 2: Test classification results for taxonomic classes using multiple models, trained on a dataset with imputed data and including the albedo feature. Reported metrics include test accuracy with 95% confidence intervals, and the mean and standard deviation of 10-fold cross-validation runs.**

| Model | Test Accuracy | 95% CI | Accuracy | BAcc | F1 | MCC |
|---|---|---|---|---|---|---|
| GNB | 0.6599 | 0.6216-0.6982 | 0.6633$\pm$0.0039 | 0.7005$\pm$0.0047 | 0.6862$\pm$0.0033 | 0.6150$\pm$0.0040 |
| MLP | 0.8690 | 0.8418-0.8963 | 0.8744$\pm$0.0059 | 0.7981$\pm$0.0128 | 0.8731$\pm$0.0067 | 0.8429$\pm$0.0074 |
| MLR | 0.8673 | 0.8399-0.8948 | 0.8808$\pm$0.0031 | 0.7890$\pm$0.0055 | 0.8789$\pm$0.0030 | 0.8503$\pm$0.0039 |
| RF | 0.8350 | 0.8050-0.8650 | 0.8500$\pm$0.0018 | 0.6876$\pm$0.0059 | 0.8436$\pm$0.0022 | 0.8102$\pm$0.0023 |
| SVM | 0.8810 | 0.8548-0.9071 | 0.8911$\pm$0.0032 | 0.8169$\pm$0.0054 | 0.8899$\pm$0.0030 | 0.8636$\pm$0.0040 |

Furthermore, we report the optimal hyperparameters (of *scikit-learn* [Pedregosa2011] package) utilized for each machine learning model, as determined through rigorous grid search. These parameters are detailed in Table 3.

**Table 3: Optimal hyperparameter configurations resulting from grid search for each model.**

| Best Parameters | Without Albedo | With Albedo |
|---|---|---|
| GB | Variance smoothing: 1e-12 | Variance smoothing: 1e-12 |
| MLP | Hidden layes size: (32, 32), Learning rate: 0.01, Max iterations: 1000, Solver: sgd | Hidden layes size: (64, 64), Learning rate: 0.05, Max iterations: 1000, Solver: sgd |
| MLR | C value: 4.2813, Max iterations: 10000, Penalty: l1, Solver: Saga | C value: 5, Max iterations: 10000, Penalty: l1, Solver: Saga |
| RF | Max depth: 15, Max features: sqrt, Min samples leaf: 10, Min samples split: 15, n estimators: 50 | Max depth: 9, Max features: 0.5, Min samples leaf: 10, Min samples split: 15, n estimators: 20 |
| SVM | C value: 29.7635, Gamma: auto, Kernel: rbf | C value: 19, Gamma: auto, Kernel: rbf |

## Conclusion

This study pioneered a novel approach to asteroid taxonomic classification by addressing the persistent challenge of missing data through a robust imputation methodology. Unlike previous studies, except for Mahlke et al. [Mahlke2022T], that either ignored or worked around missing data, we aimed to create a complete and reliable dataset. To validate the reliability of our imputation method, we rigorously tested the resulting complete dataset with five machine learning models. The high classification accuracies achieved demonstrate the method's effectiveness.

We achieved test accuracy scores ranging from **63.10% to 85.37%** with spectral features only, and from **65.99% to 88.10%** with spectra and albedo as features. Moreover, the model achieving the highest test accuracy (SVM) demonstrated a good performance, with a test accuracy of **88.10%** when albedo was included as an extra feature, suggesting that this configuration was the most effective overall and that this model was robust in handling the given dataset. Moreover, our findings suggest that the best model to predict classes is **SVM**, as it displayed the **best result (89.11%)** in the 10-fold cross-validation. Including albedo as an additional feature significantly improved model performance, particularly for models like RF, which saw an improvement of up **to 3.4% in test** accuracy scores.

In summary, the proposed data imputation method proved to be a viable approach, achieving high accuracy **(88.10%)** and demonstrating its potential for improving the accuracy of taxonomic classification in asteroid databases. This is particularly important, considering that many state-of-the-art approaches and previous studies often reduce the dataset size due to missing data at specific wavelengths or limit the sample size because of incomplete observations. The proposed methodology effectively mitigated these limitations by imputing the missing data; however, imputation beyond 50% missing data introduced significant uncertainties, which likely affected our results. Therefore, we recommend excluding samples with over 50% missing data in future studies to maximize accuracy. Ensuring a balance between model complexity, dataset size, and feature richness is key to enhancing classification accuracy.

Future work will focus on implementing this imputation methodology to construct a large, high-quality dataset, which will serve as the basis for developing a novel asteroid taxonomic classification scheme. This approach could also be motivational for other astrophysics studies where missing data is a prevalent challenge.
