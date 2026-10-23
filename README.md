# 🛵  Food Delivery Time Prediction

Predicting food delivery time (in minutes) with regression models, from exploratory analysis to model evaluation.

> **Headline result:** a Linear Regression model cuts the average estimation error from **17.5 minutes** (guessing the average) to **6.05 minutes**, and explains about **82%** of the variation in delivery time.

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
7. [Author](#author)

---

## 🚨Project Background

Food delivery platforms promise a delivery time to every customer. When that promise is wrong, customers lose trust and couriers are scheduled poorly. A common shortcut is to quote the average delivery time for every order, but delivery time depends heavily on the trip itself: how far it is, how long the restaurant takes to prepare the food, and the conditions on the road.

This project is a take-home task that uses a dataset of **1,000 delivery orders** to build and compare regression models that predict delivery time, and to find which factors matter most.

---

## Problem Statement, Analysis Objectives, and Business Problem

### 🎯Problem Statement

Delivery time estimates based on a simple average are inaccurate. Guessing the average delivery time for every order misses by about **17.5 minutes per order** on held-out data.

Several factors create the delay:

- **Core variables:** trip distance, preparation time, and weather set a baseline delay.
- **External friction:** heavy traffic adds unpredictable bottlenecks.

### 🎯Analysis Objectives

1. Understand the distribution of delivery time and which factors are related to it.
2. Prepare the data without data leakage (split first, clean with training statistics only).
3. Build and compare four regression models: Baseline, Linear Regression, Random Forest, and XGBoost.
4. Identify the key factors that drive delivery time.
5. Show how the best model predicts delivery time for new orders.

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

**Data quality:** `Weather`, `Traffic_Level`, `Time_of_Day`, and `Courier_Experience_yrs` each have **30 missing values** (970 of 1,000 non-null). The other columns are complete.

**Target summary (`Delivery_Time_min`):** mean 56.7 min, median 55.5 min, standard deviation 22.1 min, minimum 8 min, maximum 153 min.

### Data Preprocessing

| Step | What was done | Why |
|---|---|---|
| 1. Split first | 80/20 train-test split (800 / 200 orders, `random_state=42`) before any cleaning. `Order_ID` and the target are removed from the inputs. | Keeps the test set unseen and prevents data leakage. |
| 2. Impute missing values | Mode for `Weather`, `Traffic_Level`, `Time_of_Day`. Median for `Courier_Experience_yrs`. Values are computed on the **train set only** and applied to both sets. | No information from the test set leaks into training. |
| 3. Check outliers | IQR rule (1.5 × IQR) on `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`, with bounds from the train set. **0 outliers found**, so no capping was needed. | Confirms the numeric data is within a reasonable range. |
| 4. Feature engineering | Four features built from input columns only (see below). | Gives the model combined signals such as traffic load and courier skill. |
| 5. One-hot encoding | `Weather`, `Time_of_Day`, `Vehicle_Type` converted to 0/1 columns (`drop_first=True`). Test columns aligned to the train columns. | Models need numeric input. |

**Engineered features**

| Feature | Definition | Purpose |
|---|---|---|
| `Traffic_Ord` | Low, Medium, High → 0, 1, 2 | Ordered traffic scale (replaces `Traffic_Level`) |
| `Dist_x_Traffic` | `Distance_km` × `Traffic_Ord` | Load from distance and congestion |
| `Dist_per_Exp` | `Distance_km` ÷ (`Courier_Experience_yrs` + 1) | Distance relative to courier skill |
| `Bad_Weather` | 1 if Rainy, Snowy, or Foggy | Flags adverse conditions |

**Final feature count: 16** = 3 numeric columns + 4 engineered features + 9 one-hot columns (4 weather, 3 time of day, 2 vehicle type).

### Analysis and Insights

#### 1. What does delivery time look like?

![Delivery time distribution](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(14).png?raw=true.png)

- The average delivery takes about **57 minutes** (median 55.5), but the spread is wide (standard deviation 22.1 minutes).
- The distribution has a long upper tail, reaching **153 minutes**.

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

Vehicle type and time of day showed little difference in delivery time during EDA.

#### 3. Model comparison

Four models were trained on the same 800 training orders and evaluated on the same 200 test orders.

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| **Linear Regression** | **6.05** | **8.94** | **0.822** |
| XGBoost | 6.66 | 9.56 | 0.796 |
| Random Forest | 6.80 | 9.61 | 0.794 |
| Baseline (mean) | 17.50 | 21.23 | -0.006 |

- **Linear Regression wins.** On average its prediction is off by about 6 minutes, versus 17.5 minutes for the baseline.
- XGBoost and Random Forest are close behind. A simple model beating tree ensembles suggests the relationships in this dataset are mostly linear.
- MAE is the average absolute miss in minutes. RMSE penalizes large misses more. R² is the share of variation explained.

#### 4. What does the model rely on?

![Linear Regression coefficients](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(2).png?raw=true.png)

Standardized coefficients of the Linear Regression model: the change in predicted delivery time (minutes) when a feature increases by **one standard deviation**, with other features held constant.

| Feature | Change in delivery time |
|---|---|
| `Distance_km` | +17.56 min |
| `Preparation_Time_min` | +7.03 min |
| `Traffic_Ord` | +4.31 min |
| `Bad_Weather` | +2.43 min |
| `Courier_Experience_yrs` | -2.21 min |
| `Weather_Snowy` | +1.29 min |

- **Distance is the strongest driver**, followed by preparation time and traffic.
- More experienced couriers are slightly faster, but the effect is small.
- A Random Forest feature importance analysis points to the same top factors (distance, then preparation time), which acts as a cross-check from a second model.

#### 5. How close are the predictions?

![Actual vs predicted](https://github.com/naelasproject0817/Food_Delivery_Time_Prediction/blob/main/download%20(3).png?raw=true.png)

- Most points sit close to the ideal line, so predictions track real delivery times well. The typical error is about 6 minutes (MAE 6.05, RMSE 8.94).
- A few **very long deliveries (around 90 minutes and above) are under-predicted**.

#### 6. Predictions for new orders

A `preprocess_new()` function applies the same imputation, feature engineering, and column alignment used in training, so new orders can be scored directly.

| Distance (km) | Weather | Traffic | Time of day | Vehicle | Prep (min) | Experience (yrs) | Predicted (min) |
|---|---|---|---|---|---|---|---|
| 5.0 | Clear | Low | Morning | Bike | 10 | 3.0 | **28.9** |
| 12.5 | Rainy | High | Evening | Scooter | 20 | 1.0 | **78.2** |
| 20.0 | Foggy | Medium | Night | Car | 25 | 5.0 | **98.0** |

Longer trips under tougher conditions receive longer predictions, as expected.

#### Limitations

- The dataset is small (1,000 orders) and the results come from a single train-test split, without cross-validation.
- Distance appears in several features (`Distance_km`, `Dist_x_Traffic`, `Dist_per_Exp`), so coefficients should be read as indicative rather than exact.
- The model under-predicts very long deliveries.

---

## Recommendations and Actionable Insights

**Insights**

1. **Distance dominates** delivery time (correlation 0.78, and the largest coefficient in the model).
2. **A simple model wins.** Linear Regression gives the best accuracy and is easy to explain.
3. **Conditions add delay.** Heavy traffic and snowy weather raise median delivery times.

**Recommendations**

| # | Recommendation | Why |
|---|---|---|
| 1 | Replace average-based delivery estimates with this model. | It cuts the typical estimate error from about 17.5 to about 6 minutes. |
| 2 | Add time buffers for long-distance, high-traffic, and snowy orders. | These orders have the highest delivery times and are the most likely to run late. |
| 3 | Keep improving the model: cross-validate, tune the models, and study the long deliveries that are under-predicted. | A single split is a limited test, and the longest deliveries are where the model misses most. |

---

## Tech Stack

- **Language:** Python 3
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (Linear Regression, Random Forest, DummyRegressor, metrics), XGBoost
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

1. Clone the repository and install the dependencies: `pip install -r requirements.txt`
2. Open `notebooks/food_delivery_modeling.ipynb` in Jupyter or Google Colab.
3. Run the cells from top to bottom. The notebook follows the order: EDA → split → cleaning → feature engineering → modeling → evaluation.

---
## Food Delivery Time Prediction Dasboard
![Food Delivery Time Prediction](https://github.com/user-attachments/assets/31c2df18-44ff-4a24-99fb-16ff40e16943)

---

## 👩‍💻 Author

**Naela Arbach**

* 💼 **LinkedIn:** [linkedin.com/in/naelaarbach](https://www.linkedin.com/in/naelaarbach/)
* 🐙 **GitHub:** [github.com/naelasproject0817](https://github.com/naelasproject0817)
* 📧 **Email:** [naelaproject0817@gmail.com](mailto:naelaproject0817@gmail.com)
