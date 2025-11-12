# Bank Marketing Dataset (Numerical Subset)

This project uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository.  
For this experiment, we only use the **numerical features**:

- `age`
- `duration`
- `campaign`
- `pdays`
- `previous`
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

These attributes describe customer demographics, contact history, and economic indicators.  
All data are numeric and pre-cleaned (missing rows dropped).

## Preprocessing Notes
- Standardized each feature to zero mean and unit variance using `StandardScaler`.
- Applied **PCA (3 components)** to retain ~63% of variance.
- Used for **K-Means** and **Bisecting K-Means** clustering to identify customer segments.

## File Info
- File name: `bank_marketing.csv`
- Delimiter: `,`
- Size: ~40,000 rows × 10 columns
