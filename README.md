# Boston House Price Prediction — Regularized Regression

Predicts median house value (`medv`) with **Ridge and Lasso regression**. The project covers multicollinearity checks, hyperparameter tuning on a validation set, and over/underfitting evaluation.

## Dataset
- **Boston Housing** (public dataset): 506 rows × 14 columns
- Features: crime rate, rooms per dwelling, pupil-teacher ratio, % lower-status population, property tax, and others

## Tools
Python · Pandas · NumPy · Scikit-learn · Statsmodels · Matplotlib · Seaborn

## Workflow
1. **Data understanding**: no missing values and no duplicates
2. **Multicollinearity check** with **VIF** and a correlation heatmap
   - `rad` (VIF 7.7) and `tax` (VIF 8.9) are highly correlated (> 0.8). **`rad` was dropped** because `tax` correlates more strongly with the target.
   - After dropping it, every VIF is below 4.5
3. **Train / validation / test split**
4. **Hyperparameter tuning**
   - Ridge: α ∈ {0.0001 … 1000}
   - Lasso: α ∈ {0.00001 … 10}
   - Best α chosen by validation RMSE
5. **Experiment**: compared dropping only `rad` vs dropping `rad` + `dis` (VIF > 4)
6. **Evaluation**: RMSE, MAE, and MAPE on train and test

## Results
**Best model: Ridge Regression (α = 100), dropping only `rad`**

| Data | RMSE | MAE | MAPE |
|------|------|-----|------|
| Train | 5.04 | 3.58 | 17.2% |
| Test | 5.01 | 3.23 | 17.1% |

- Train and test errors are almost the same, so the model **does not overfit**.
- Dropping `dis` as well (VIF > 4) made validation performance worse (best RMSE 4.99 vs 4.79), so dropping only `rad` was the better choice.
- **Strongest predictors:** `rm` (rooms, +2.61), `dis` (−1.13), `ptratio` (−0.77), `lstat` (−0.69)
- Lasso automatically set `indus`, `chas`, and `nox` to zero, which acts as built-in feature selection

## Next Steps
MAPE is still above 10%. Non-linear models such as Random Forest, Gradient Boosting, or XGBoost may capture patterns that linear models miss.

## Repository Structure
```
boston-house-price-regression/
├── house_price_regression.ipynb   # full analysis (narrative in Bahasa Indonesia)
└── README.md
```

## How to Run
1. Open the notebook in Google Colab or Jupyter
2. Upload `boston.csv`, then run all cells

---
**Author:** Andi Yuswandi · [GitHub](https://github.com/yuswandi99)
