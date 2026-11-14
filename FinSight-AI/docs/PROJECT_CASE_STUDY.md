# FinSight AI — Portfolio Case Study

## Business problem

A digital payments business needs to understand payment behavior, investigate unusual transactions, and identify patterns associated with fraud. It also needs to understand how different customer groups use the service.

## Questions investigated

- How do transaction volume and payment outcomes change over time?
- How do observed fraud labels vary by country and payment method?
- Can transaction attributes distinguish the synthetic fraud labels better than a naive baseline?
- Which transactions look unusual when no fraud labels are supplied to the anomaly model?
- Are there distinct customer behavior groups based on frequency, spend and payment outcomes?

## Analytical approach

1. Generate a reproducible, documented synthetic dataset.
2. Use Pandas and NumPy to inspect data quality, derive time features and calculate summaries.
3. Use Matplotlib to visualize volume, fraud patterns and transaction amounts.
4. Build a Scikit-learn preprocessing/model Pipeline. Split data into training, validation and test partitions. Choose a probability threshold on validation data and report final metrics on the untouched test set.
5. Evaluate imbalanced classification with precision, recall, F1, PR-AUC, ROC-AUC and a confusion matrix.
6. Fit Isolation Forest without using the fraud target.
7. Aggregate transactions to customer level, scale features and apply K-Means. Profile clusters before assigning business interpretations.

## Important interpretation

The data is simulated and the fraud labels are generated from a known synthetic process with random variation. Model metrics are useful for demonstrating a workflow, not for claiming real-world fraud-detection capability. In the included run, the Random Forest's test ROC-AUC is approximately 0.67 and PR-AUC approximately 0.07; these figures depend on the generated sample and should be recalculated when the dataset or random seed changes. Do not describe this as production-ready or as a PayPal model.

Anomaly detection and fraud classification answer different questions. Isolation Forest flags unusual feature combinations, not confirmed fraud. A customer segment is a mathematical grouping, not a risk category by itself. Review the segment profile table before naming clusters.

## Next research iteration

Replace or supplement the synthetic dataset with a public, documented benchmark such as IEEE-CIS. Read its data dictionary, check licensing/usage conditions, inspect time fields, and design a leakage-safe split. Do not assume the synthetic feature definitions map directly to the public dataset.

## Interview talking points

- Explain why accuracy is weak when the positive class is rare.
- Explain the precision/recall trade-off and why threshold selection matters.
- Explain why the threshold is selected on validation data rather than the test set.
- Explain the difference between supervised classification and unsupervised anomaly detection.
- Explain why scaling matters for K-Means but is not inherently required by Random Forest.
- Explain why customer IDs and transaction IDs are not used as predictive features.
- Explain why synthetic data limits the conclusions you can draw.
