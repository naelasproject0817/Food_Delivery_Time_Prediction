# 🛵  Food Delivery Time Prediction

Predicting food delivery time (in minutes) with regression models, from exploratory analysis to model evaluation, and turning the prediction into a delivery promise customers can trust.

> **Headline result:** a Linear Regression model cuts the average estimation error from **17.5 minutes** (guessing the average) to **6.05 minutes** (6.68 in 5-fold cross-validation) and explains about **82%** of the variation in delivery time. Adding a data-driven buffer (**8.7 min** for normal orders, **14.5 min** when heavy traffic meets bad weather) makes the promised time hold for about **9 in 10 orders**.

## 📌 Table of Contents

1. [Project Background](#project-background)
2. [Problem Statement, Analysis Objectives, and Business Problem](#problem-statement-analysis-objectives-and-business-problem)
3. [Analysis](#analysis)
   - [Data Understanding](#data-understanding)
   - [Data Preprocessing](#data-preprocessing)
   - [Analysis and Insights](#analysis-and-insights)
4. [Recommendations and Actionable Insights](#recommendations-and-actionable-insights)
5. [Tech Stack](#tech-stack)
6. [Repository Structure](#repository-structure)
7. [Dashboard](#food-delivery-time-prediction-dashboard)
8. [Author](#author)

---

## 🚨Project Background

Food delivery platforms promise a delivery time to every customer. When that promise is wrong, customers lose trust and couriers are scheduled poorly. A common shortcut is to quote the average delivery time for every order, but delivery time depends heavily on the trip itself: how far it is, how long the restaurant takes to prepare the food, and the conditions on the road.

This project is a take-home task that uses a dataset of **1,000 delivery orders** to build and compare regression models that predict delivery time, to find which factors matter most, and to turn the prediction into a realistic time promise.

---

## Problem Statement, Analysis Objectives, and Business Problem

### 🎯Problem Statement

Delivery time estimates based on a simple average are inaccurate. Guessing the average delivery time for every order misses by about **17.5 minutes per order** on held-out data, which hurts customer satisfaction.

Several factors create the delay:

- **Core variables:** trip distance (r = 0.78), preparation time (r = 0.31), and weather set a baseline delay.
- **External friction:** heavy traffic adds unpredictable bottlenecks. The median delivery time is **65 min** in high traffic versus **51 min** in low traffic.

**The solution:** a regression model that cuts the average error from 17.5 to 6.05 minutes and reveals the key drivers of delivery time.

### 🎯Analysis Objectives

1. Understand the distribution of delivery time and which factors are related to it.
2. Prepare the data without data leakage (split first, clean with training statistics only).
3. Build and compare four regression models: Baseline, Linear Regression, Random Forest, and XGBoost.
4. Check that the result is robust with 5-fold cross-validation.
5. Identify the key factors that drive delivery time.
6. Show how the best model predicts delivery time for new orders.
7. Turn the prediction into a delivery promise by adding a buffer based on real errors.

### 💼 Business Problem

Inaccurate delivery estimates frustrate customers and disrupt courier scheduling.

**Success measure:** the model must beat the baseline (predicting the average) on a held-out test set, measured with MAE, RMSE, and R².

---

## 📊 Analysis

### Data Understanding

The dataset has **1,000 orders and 9 columns**.

| Column | Description |
|---|---|
| `Order_ID` | Unique order identifier (dropped, no predictive value) |
| `Distance_km` | Trip distance in kilometers |
| `Weather` | Clear, Foggy, Rainy, Snowy, Windy |
| `Traffic_Level` | Low, Medium, High |
| `Time_of_Day` | Morning, Afternoon, Evening, Night |
| `Vehicle_Type` | Bike, Scooter, Car |
| `Preparation_Time_min` | Restaurant preparation time in minutes |
| `Courier_Experience_yrs` | Courier experience in years |
| `Delivery_Time_min` | **Target:** delivery time in minutes |

**Data quality:** `Weather`, `Traffic_Level`, `Time_of_Day`, and `Courier_Experience_yrs` each have **30 missing values** (970 of 1,000 non-null). The other columns are complete. There are **no duplicate rows**.

**Target summary (`Delivery_Time_min`):** mean 56.7 min, median 55.5 min, standard deviation 22.1 min, minimum 8 min, maximum 153 min.

### Data Preprocessing

The principle is **split first, then clean using training statistics only**, to prevent data leakage.

| Step | What was done | Why |
|---|---|---|
| 1. Split first | 80/20 train-test split (800 / 200 orders, `random_state=42`) before any cleaning. `Order_ID` and the target are removed from the inputs. | Keeps the test set unseen and prevents data leakage. |
| 2. Impute missing values | Mode for `Weather`, `Traffic_Level`, `Time_of_Day`. Median for `Courier_Experience_yrs`. Values are computed on the **train set only** and applied to both sets. 30 gaps per column down to 0. | No information from the test set leaks into training. |
| 3. Check outliers | IQR rule (1.5 × IQR) on `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`, with bounds from the train set. **0 outliers found**, so no capping was needed. | Confirms the numeric data is within a reasonable range. |
| 4. Feature engineering | Four features built from input columns only, never from the target (see below). | Gives the model combined signals such as traffic load and courier skill. |
| 5. One-hot encoding | `Weather`, `Time_of_Day`, `Vehicle_Type` converted to 0/1 columns (`drop_first=True`). Test columns aligned to the train columns. | Models need numeric input. |

**Engineered features**

| Feature | Definition | Purpose |
|---|---|---|
| `Traffic_Ord` | Low, Medium, High → 0, 1, 2 | Ordered traffic scale (replaces `Traffic_Level`) |
| `Dist_x_Traffic` | `Distance_km` × `Traffic_Ord` | Load from distance and congestion |
| `Dist_per_Exp` | `Distance_km` ÷ (`Courier_Experience_yrs` + 1) | Distance relative to courier skill |
| `Bad_Weather` | 1 if Rainy, Snowy, or Foggy | Flags adverse conditions |

**Final feature count: 16** = 3 numeric columns + 4 engineered features + 9 one-hot columns (4 weather, 3 time of day, 2 vehicle type). The cleaned dataset is exported as `cleaned_food_delivery_16features.csv`.

### Analysis and Insights

#### 1. What does delivery time look like?

![Delivery time distribution](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(14).png?raw=true.png)

- The average delivery takes about **57 minutes** (median 55.5), but the spread is wide (standard deviation 22.1 minutes).
- Most orders arrive in roughly 40 to 75 minutes, but the distribution has a long upper tail, reaching **153 minutes**. A small share of orders takes much longer than usual.

#### 2. What drives delivery time?

![Correlation with delivery time](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download.png?raw=true.png)

| Feature | Correlation with delivery time |
|---|---|
| `Distance_km` | **0.78** (strong) |
| `Preparation_Time_min` | 0.31 (moderate) |
| `Courier_Experience_yrs` | -0.09 (very weak) |

The numeric features are almost uncorrelated with each other, so each one adds different information.

![Median delivery time by traffic and weather](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(1).png?raw=true.png)

| Condition | Median delivery time |
|---|---|
| High traffic vs Low traffic | ~65 min vs ~51 min |
| Snowy weather vs Clear weather | ~66 min vs ~52 min |
| Foggy / Rainy weather | ~59 min / ~58 min (also slower than Clear) |

Vehicle type and time of day barely shift delivery time.

#### 3. Model comparison

Four models were trained on the same 800 training orders and evaluated on the same 200 test orders.

| Model | Description |
|---|---|
| Baseline | Predicts the training mean: the bar every model must beat |
| Linear Regression | Simple and interpretable, one weight per feature |
| Random Forest | 200 trees, captures non-linear patterns |
| XGBoost | 200 boosted trees, learning rate 0.05, max depth 3 |

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| **Linear Regression** | **6.05** | **8.94** | **0.822** |
| XGBoost | 6.66 | 9.56 | 0.796 |
| Random Forest | 6.80 | 9.61 | 0.794 |
| Baseline (mean) | 17.50 | 21.23 | -0.006 |

- **Linear Regression wins.** On average its prediction is off by about 6 minutes, versus 17.5 minutes for the baseline.
- XGBoost and Random Forest are close behind. A simple model beating tree ensembles suggests the relationships in this dataset are mostly linear.
- MAE is the average absolute miss in minutes. RMSE penalizes large misses more. R² is the share of variation explained. A baseline R² near 0 means the mean alone explains none of the variation.

#### 4. Is the result robust? 5-fold cross-validation

A single 80/20 split can be lucky, so the three real models were re-tested with 5-fold cross-validation (shuffled, `random_state=42`) on all 1,000 orders. The baseline has no learning, so it is not included.

| Model | MAE (min) | R² |
|---|---|---|
| **Linear Regression** | **6.68** | **0.762** |
| XGBoost | 7.26 | 0.737 |
| Random Forest | 7.51 | 0.723 |

- The **ranking is the same** as on the single split, so the model choice does not depend on one lucky split.
- The numbers are slightly worse than on the test split, so the realistic expectation for new data is an error of **about 6 to 7 minutes**.
- The gap between models is small, but Linear Regression is consistently ahead and is the easiest to explain.

#### 5. What does the model rely on?

![Linear Regression coefficients](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(2).png?raw=true.png)

Standardized coefficients of the Linear Regression model: the change in predicted delivery time (minutes) when a feature increases by **one standard deviation**, with other features held constant.

| Feature | Change in delivery time |
|---|---|
| `Distance_km` | +17.56 min |
| `Preparation_Time_min` | +7.03 min |
| `Traffic_Ord` | +4.31 min |
| `Weather_Snowy` | +2.75 min |
| `Courier_Experience_yrs` | -2.21 min |
| `Weather_Rainy` | +1.90 min |

- **Distance is the strongest driver.** One standard deviation of distance is about 5.73 km, so each extra kilometer adds roughly **3 minutes**. Preparation time and traffic follow.
- More experienced couriers are slightly faster, but the effect is small.
- `Bad_Weather` is left out of this table because it is redundant (it equals Foggy + Rainy + Snowy). Dropping it does not change the predictions (maximum difference 0.00 min).
- Distance appears in several features (`Distance_km`, `Dist_x_Traffic`, `Dist_per_Exp`), so coefficients should be read as indicative, not exact.

#### 6. How close are the predictions?

![Actual vs predicted](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(3).png?raw=true.png)

- Most points sit close to the ideal red dashed line, so predictions track real delivery times well. The typical error is about 6 minutes (MAE 6.05, RMSE 8.94 on the test split).
- Only **5 of 200 test orders** were under-predicted by more than 20 minutes (the order arrived much later than predicted).
- Those 5 orders did **not** have unusual distances (average 9.51 km vs 9.90 km for all orders). What they share is conditions: **4 of 5 had bad weather** (vs 40% of all orders), **3 of 5 were foggy** (vs 10%), and traffic tended to be heavier (average level 1.40 vs 0.78 on a 0 to 2 scale).
- A possible explanation is that a linear model adds up the effect of each factor, while heavy traffic and bad weather together can make each other worse. With only 5 orders, this is an early lead, not a conclusion, and needs more data to confirm.

#### 7. Predictions for new orders

A `preprocess_new()` function applies the same imputation, feature engineering, and column alignment used in training, so new orders can be scored directly.

| Distance (km) | Weather | Traffic | Time of day | Vehicle | Prep (min) | Experience (yrs) | Predicted (min) |
|---|---|---|---|---|---|---|---|
| 5.0 | Clear | Low | Morning | Bike | 10 | 3.0 | **28.9** |
| 12.5 | Rainy | High | Evening | Scooter | 20 | 1.0 | **78.2** |
| 20.0 | Foggy | Medium | Night | Car | 25 | 5.0 | **98.0** |

Longer trips under tougher conditions receive longer predictions, as expected.

#### 8. From prediction to promise

A single predicted number is not a promise: the model misses by about 6 minutes on average, so roughly half of the orders would arrive later than the prediction. Two ways to add a safety margin were compared.

**a) 90% prediction interval (statistical formula).** An OLS model (statsmodels) gives a range around each prediction.

- The range captured **96% of the test orders** (192 of 200) against a 90% target, so it is slightly loose.
- For the **14 risky orders** (heavy traffic + bad weather), coverage was 85.7% (12 of 14). The sample is small, so treat this as a lead, not a conclusion.
- The average buffer (upper bound minus prediction) is **18.3 minutes**. The formula assumes the same error for every order, so it is conservative.

**b) Buffer from real errors (used for the promise).** Errors are computed with 5-fold cross-validation on all 1,000 orders, so every prediction comes from a model that has not seen that order. Error = actual − predicted, and a positive error means the order was late. The buffer is the 90th percentile of the error.

| Group | Orders | Buffer for 90% on time |
|---|---|---|
| All orders | 1,000 | 8.9 min |
| Normal orders | 915 | **8.7 min** |
| Risky orders (heavy traffic + bad weather) | 85 | **14.5 min** |

Risky orders need a buffer about 1.7 times larger.

**Why not just add the MAE (6.05 min)?** MAE is the average miss in both directions, not a safe distance.

| Buffer added to the prediction | Orders not late |
|---|---|
| 0 min | 57.8% |
| 6.05 min (MAE) | 82.6% (about 1 in 6 still late) |
| 8.9 min | 90.1% |

**Promise = prediction + buffer** for the three sample orders:

| Order | Predicted (min) | Group | Buffer (min) | Promise (min) | Formula range (90%) |
|---|---|---|---|---|---|
| 5 km, Clear, Low | 28.9 | Normal | +8.7 | **37.6** | 10.7 – 47.2 |
| 12.5 km, Rainy, High | 78.2 | Risky | +14.5 | **92.7** | 59.9 – 96.5 |
| 20 km, Foggy, Medium | 98.0 | Normal | +8.7 | **106.7** | 79.5 – 116.4 |

#### Limitations

- The dataset is small (1,000 orders). Cross-validation covers the three trained models only, not the baseline.
- Distance appears in several features, so coefficients are indicative rather than exact.
- The pattern behind the under-predicted orders is based on only 5 orders, and the risky group for the buffer has 85 orders. Both need more data to confirm.
- The buffers are estimated from this dataset and should be re-checked as new orders arrive.

---

## Recommendations and Actionable Insights

**Insights**

1. **Distance dominates** delivery time (correlation 0.78, the largest coefficient, and about 3 extra minutes per km in the model).
2. **A simple model wins.** Linear Regression gives the best accuracy (MAE 6.05 min, 6.68 in 5-fold CV, vs a 17.5 min baseline) and is easy to explain.
3. **Conditions add delay.** High traffic (65 vs 51 min) and snowy weather (66 vs 52 min) raise median delivery times.

**Recommendations**

| # | Recommendation | Why |
|---|---|---|
| 1 | Replace average-based delivery estimates with this model. | It cuts the typical estimate error from about 17.5 to about 6 to 7 minutes. |
| 2 | Quote the promise as prediction + buffer: add about 9 minutes, or about 15 minutes when heavy traffic meets bad weather. | Buffers from real errors make about 9 in 10 orders arrive within the promise. A buffer of only 6 minutes would leave about 1 in 6 orders late. |
| 3 | Keep improving the model: use more data, tune the models, and test traffic × weather interaction effects. | The few large misses happen when heavy traffic and bad weather combine, and a linear model adds effects instead of combining them. |

The goal is not to deliver faster, but to make the promised time more accurate.

---

## Tech Stack

- **Language:** Python 3
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Power BI
- **Modeling:** scikit-learn (Linear Regression, Random Forest, DummyRegressor, cross-validation, metrics), XGBoost
- **Statistics:** statsmodels (OLS prediction intervals)
- **Environment:** Google Colab / Jupyter Notebook
- **Presentation:** PowerPoint

---

## Repository Structure

```
food-delivery-time-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── food_delivery_modeling.ipynb
├── data/
│   └── cleaned_food_delivery_16features.csv
├── images/
│   ├── 01_delivery_time_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_median_by_traffic_weather.png
│   ├── 04_linear_regression_coefficients.png
│   └── 05_actual_vs_predicted.png
└── presentation/
    └── Food_Delivery_Time_Prediction.pptx
```

**How to run**

1. Clone the repository and install the dependencies: `pip install -r requirements.txt` (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, statsmodels).
2. Open `notebooks/food_delivery_modeling.ipynb` in Jupyter or Google Colab.
3. Run the cells from top to bottom. The notebook follows the order: EDA → split → cleaning → feature engineering → modeling → evaluation → cross-validation → coefficients → error analysis → prediction interval and buffer → predictions for new orders.

---

## Food Delivery Time Prediction Dashboard

<img width="978" height="552" alt="image" src="https://github.com/user-attachments/assets/9a9977b0-02bc-438e-ba09-b1e60b76300b" />

The Power BI dashboard summarizes the results:

- **Summary cards:** 1,000 orders, average delivery time of 57 minutes, best MAE of 6.05 and RMSE of 8.94.
- **Charts:** average delivery time by weather and by traffic level, order distribution by time of day, correlation matrix, and the main drivers.
- **Filters:** time of day, weather, and vehicle type make the dashboard explorable.

Note: the dashboard shows **averages**, while the analysis above uses **medians**.

---

## 👩‍💻 Author

**Naela Arbach**

* 💼 **LinkedIn:** [linkedin.com/in/naelaarbach](https://www.linkedin.com/in/naelaarbach/)
* 🐙 **GitHub:** [github.com/naelasproject0817](https://github.com/naelasproject0817)
* 📧 **Email:** [naelaproject0817@gmail.com](mailto:naelaproject0817@gmail.com)
