# 📚 Data Science Reference

## Table of Contents

- [📚 Data Science Reference](#-data-science-reference)
  - [Table of Contents](#table-of-contents)
  - [1. Importing \& Displaying Data](#1-importing--displaying-data)
  - [2. Summary Stats](#2-summary-stats)
  - [3. Filtering, Calculations, Min/Max/Avg, Rounding](#3-filtering-calculations-minmaxavg-rounding)
  - [4. Conditions \& Percentage Conversions](#4-conditions--percentage-conversions)
  - [5. Charts \& Plots](#5-charts--plots)
    - [5.1 Pie Chart](#51-pie-chart)
    - [5.2 Bar Plot](#52-bar-plot)
    - [5.3 Scatter Plot](#53-scatter-plot)
  - [6. Missing Values](#6-missing-values)
  - [7. Removing Columns](#7-removing-columns)
  - [8. Text to Numeric Conversions](#8-text-to-numeric-conversions)
    - [Decision Guide](#decision-guide)
  - [9. Encoding Deep Dive + Confirm Numeric](#9-encoding-deep-dive--confirm-numeric)
    - [Label Encoding](#label-encoding)
    - [One-Hot Encoding](#one-hot-encoding)
  - [10. Correlation Vector + Heatmap](#10-correlation-vector--heatmap)
  - [11. Train/Test Split + X and y + Pre-processing](#11-traintest-split--x-and-y--pre-processing)
  - [12. Resampling – Balancing Data](#12-resampling--balancing-data)
    - [When does data need balancing?](#when-does-data-need-balancing)
    - [Undersampling (reduce majority)](#undersampling-reduce-majority)
    - [Oversampling (increase minority using SMOTE)](#oversampling-increase-minority-using-smote)
  - [13. Training Classifiers + K-Fold Cross Validation](#13-training-classifiers--k-fold-cross-validation)
  - [14. Metrics Explained – F1, Precision, Recall, Accuracy](#14-metrics-explained--f1-precision-recall-accuracy)
  - [15. Learning Curves](#15-learning-curves)
  - [16. Best Model Selection + Test Evaluation + Confusion Matrix](#16-best-model-selection--test-evaluation--confusion-matrix)
  - [17. Dummy Classifier Comparison](#17-dummy-classifier-comparison)
  - [18. K-Means Clustering + Elbow + Silhouette + Labels](#18-k-means-clustering--elbow--silhouette--labels)
  - [19. PCA – Reduce Dimensions + Explained Variance](#19-pca--reduce-dimensions--explained-variance)
  - [20. PCA Scatter Plot with Cluster Colours](#20-pca-scatter-plot-with-cluster-colours)
  - [21. Remove Features with \>10 Unique Values (for loop)](#21-remove-features-with-10-unique-values-for-loop)
  - [⚡ Quick Verbal Answer Templates](#-quick-verbal-answer-templates)
    - [On learning curves (overfitting)](#on-learning-curves-overfitting)
    - [On learning curves (underfitting)](#on-learning-curves-underfitting)
    - [On correlation](#on-correlation)
    - [On PCA effectiveness](#on-pca-effectiveness)
    - [On dummy classifier](#on-dummy-classifier)
    - [On data balancing](#on-data-balancing)

---

## 1. Importing & Displaying Data

<!-- tags: import csv read_csv head shape dtypes -->

```python
import pandas as pd

# Import CSV
df = pd.read_csv('filename.csv')

# Import with custom column names (when file has no header)
df = pd.read_csv('filename.csv', names=['col1', 'col2', 'col3'])

# Preview first 5 rows
df.head()

# Preview last 5 rows
df.tail()

# Specific rows (e.g. rows 50 to 60 inclusive)
df.iloc[50:61]       # by position
df.loc[50:60]        # by index label

# Shape: (rows, columns)
print(df.shape)
rows, cols = df.shape
print(f'Records: {rows}, Features: {cols}')
```

> 📖 **What does `.head()` show?** The first 5 rows — a quick sanity check that the file loaded correctly and column names are right.
> `.shape` gives you `(rows, columns)` — the two numbers you'll quote most often.

🔗 [pandas read_csv docs](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

---

## 2. Summary Stats

<!-- tags: describe info dtypes nunique isnull shape memory non-null -->

```python
# Full statistical summary (includes object/text columns too)
df.describe(include='all')

# Concise info: index, dtypes, non-null counts, memory usage
df.info()

# Data type of each column
print(df.dtypes)

# Number of unique values per column
print(df.nunique())

# Number of rows and columns separately
print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')

# Check for missing values
print(df.isnull().sum())
```

> 📖 **Reading `describe(include='all')`:**
> - `count` = non-null entries (lower than total rows → missing values exist)
> - `mean / std` = average and spread (numeric only)
> - `min / 25% / 50% / 75% / max` = distribution shape — if 50% (median) is far from mean, data is skewed
> - `top / freq` = most common value and how often (for text columns)
> - `unique` = number of distinct values in a text column

> 📖 **Reading `info()`:** Scan the `Non-Null Count` column — any number less than the total row count means missing values in that column. The `Dtype` column tells you if something needs converting (e.g. `object` = text that may need encoding).

🔗 [pandas describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) | [pandas info](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)

---

## 3. Filtering, Calculations, Min/Max/Avg, Rounding

<!-- tags: filter loc iloc min max mean average round sum count value_counts -->

```python
# Filter rows by condition
subset = df[df['column'] == 'value']

# Filter with multiple conditions
subset = df[(df['Age'] > 30) & (df['Gender'] == 'Male')]

# Basic calculations
print(df['column'].min())
print(df['column'].max())
print(df['column'].mean())
print(df['column'].sum())
print(df['column'].count())
print(df['column'].median())

# Value counts (frequency of each unique value)
print(df['column'].value_counts())

# Rounding
val = df['column'].mean()
print(round(val, 2))         # round to 2 decimal places

# Round entire column
df['column'] = df['column'].round(2)

# Find the row with the maximum value in a column
max_row = df.loc[df['Price'].idxmax()]
print(max_row['Name'])       # e.g. get the patient name with highest bill
```

🔗 [pandas filtering](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)

---

## 4. Conditions & Percentage Conversions

<!-- tags: condition percentage probability boolean sum len -->

```python
# Probability / percentage of a condition being true
total = len(df)

# Probability (0 to 1)
prob = (df['column'] == 'value').sum() / total
print(round(prob, 2))

# Percentage (0 to 100)
pct = (df['column'] == 'value').sum() / total * 100
print(round(pct, 1))

# Conditional probability: given male AND A+, what % have Diabetes?
filtered = df[(df['Gender'] == 'Male') & (df['Blood Type'] == 'A+')]
prob_diabetes = (filtered['Medical Condition'] == 'Diabetes').sum() / len(filtered) * 100
print(round(prob_diabetes, 1))

# Count based on condition
count = (df['Minute'] <= 45).sum()
print(count)
```

> 📖 **Probability vs Percentage:** Probability is between 0 and 1 (divide by total). Percentage is between 0 and 100 (multiply by 100). The exam usually asks to round to 1 or 2 decimals — always check the question.

---

## 5. Charts & Plots

<!-- tags: pie chart bar plot count plot seaborn matplotlib histogram -->

### 5.1 Pie Chart

```python
import matplotlib.pyplot as plt

counts = df['column'].value_counts()
counts.plot.pie(autopct=lambda p: f'{round(p, 1)}%')
plt.title('Title Here')
plt.ylabel('')   # removes the default 'column' label on y-axis
plt.show()
```

> 📖 **Reading a pie chart:** Each slice = proportion of the total. `autopct` controls how the percentages are displayed on the chart. If one slice dominates (>50%), the dataset is likely imbalanced.

---

### 5.2 Bar Plot

```python
import seaborn as sns

# Count of each category
sns.countplot(data=df, x='column')
plt.title('Count per Category')
plt.show()

# Bar plot with calculated values
df['column'].value_counts().plot(kind='bar')
plt.title('Title')
plt.show()
```

---

### 5.3 Scatter Plot

```python
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Category')
plt.title('Feature1 vs Feature2')
plt.show()
```

> 📖 **Reading a scatter plot:**
> - Points going **up-right** = positive correlation (both increase together)
> - Points going **down-right** = negative correlation (one increases, other decreases)
> - **Random spread** = little to no correlation
> - `hue` colours points by category — clusters of same colour = that category has distinct behaviour

> 📖 **Reading a count/bar plot:** Taller bars = more records in that category. A single bar much taller than others suggests class imbalance.

🔗 [seaborn countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html) | [matplotlib pie](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html)

---

## 6. Missing Values

<!-- tags: isnull isna fillna dropna impute median mode missing percentage -->

```python
# Count of missing values per column
print(df.isnull().sum())

# As percentages
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])

# Fill numeric columns with median (safer than mean — not affected by outliers)
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill text/categorical columns with mode (most common value)
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop rows with missing values (use sparingly)
df = df.dropna()

# Confirm all missing values are handled
print(df.isnull().sum().sum())   # should print 0
```

> 📖 **Why median for numbers?** The median is the middle value — it is not pulled by extreme outliers the way the mean is. For example, if most people earn R20K but one earns R2M, the median stays near R20K; the mean does not.
> **Why mode for categories?** You can't average text — so you use the most common value as the best guess.

🔗 [pandas fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)

---

## 7. Removing Columns

<!-- tags: drop columns iloc range axis -->

```python
# Remove a specific column by name
df = df.drop(columns=['Date'])

# Remove multiple columns
df = df.drop(columns=['col1', 'col2', 'col3'])

# Remove first column and last 7 columns (without specifying names)
df = df.iloc[:, 1:-7]
#            ^  ^  ^
#  all rows  |  |  last 7 excluded
#               first column excluded (index 0)

# Remove columns by index range
# Keep only columns from index 2 to 10
df = df.iloc[:, 2:11]
```

> 📖 **`iloc[:, 1:-7]` explained:**
> - `:` = keep all rows
> - `1:-7` = start from column index 1 (skip first), stop 7 before the end (skip last 7)
> - Negative index counts from the right: `-1` = last column, `-7` = 7th from the end

🔗 [pandas iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)

---

## 8. Text to Numeric Conversions

<!-- tags: LabelEncoder get_dummies one-hot object numeric convert -->

### Decision Guide

| Situation | Method |
|---|---|
| Target column (what you're predicting) | `LabelEncoder` |
| Input column with 2 unique values | `LabelEncoder` (gives 0/1) |
| Input column with ≤10 unique values | `LabelEncoder` (gives 0,1,2...) |
| Input column with >10 unique values | `pd.get_dummies` (one-hot) |

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode target column
df['Target'] = le.fit_transform(df['Target'])

# Encode categorical columns with few unique values
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() <= 10:
        df[col] = le.fit_transform(df[col])

# One-hot encode columns with many unique values
high_card = [col for col in df.select_dtypes(include='object').columns
             if df[col].nunique() > 10]
df = pd.get_dummies(df, columns=high_card)

# Confirm no object columns remain
print(df.dtypes)
print(df.select_dtypes(include='object').columns.tolist())  # should be empty []
```

---

## 9. Encoding Deep Dive + Confirm Numeric

<!-- tags: one-hot label binary encoding confirm object dtype -->

### Label Encoding

Converts categories to integers: `['Yes','No','Maybe']` → `[2, 0, 1]`
- ✅ Use for: target/output column, binary columns, ordinal data (where order matters)
- ⚠️ Avoid for: nominal columns with many values — the model may wrongly assume 3 > 2 > 1

### One-Hot Encoding

Creates a new binary column per unique value: `['Red','Blue','Green']` → three columns `Red(0/1)`, `Blue(0/1)`, `Green(0/1)`
- ✅ Use for: input columns with no natural order and more than 2 values
- ⚠️ Warning: creates many columns if there are many unique values (use sparingly)

```python
# One-hot example
df = pd.get_dummies(df, columns=['Colour'])
# Result: df now has columns Colour_Red, Colour_Blue, Colour_Green (each 0 or 1)

# Confirm all numeric
print(df.dtypes)
# Every column should show int64, float64, or bool — no 'object'

# Quick check
assert df.select_dtypes(include='object').empty, "Still has object columns!"
```

🔗 [sklearn LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) | [pandas get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

---

## 10. Correlation Vector + Heatmap

<!-- tags: correlation corr heatmap seaborn pearson analysis -->

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Full correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Correlation of all features with ONE target column
corr_vector = df.corr()[['Target']].sort_values('Target', ascending=False)
print(corr_vector)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

> 📖 **How to read a correlation value:**
>
> | Value | Meaning |
> |---|---|
> | 1.0 | Perfect positive — as one goes up, other goes up exactly |
> | 0.7 – 0.9 | Strong positive correlation |
> | 0.4 – 0.6 | Moderate positive correlation |
> | 0.1 – 0.3 | Weak correlation |
> | 0.0 | No linear relationship |
> | -0.4 to -0.9 | Negative correlation — as one goes up, other goes down |
> | -1.0 | Perfect negative |
>
> 📖 **Reading the heatmap:**
> - **Dark red** = strong positive correlation
> - **Dark blue** = strong negative correlation
> - **White/light** = little to no correlation
> - The **diagonal** is always 1.0 (a feature perfectly correlates with itself — ignore it)
> - Look for features with high correlation to your **target column** — those are your strongest predictors
>
> 📖 **What to say in verbal analysis:**
> "Feature X has a correlation of 0.72 with the target, indicating a strong positive relationship — as X increases, the likelihood of [target outcome] also increases. Features A and B are highly correlated with each other (0.91), which may indicate multicollinearity and could affect model performance."

🔗 [seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

---

## 11. Train/Test Split + X and y + Pre-processing

<!-- tags: train_test_split X y StandardScaler fit_transform -->

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define X (features) and y (target)
X = df.drop(columns=['Target'])
y = df['Target']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features (important for distance-based models like KNN, SVM, LR)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on train only
X_test  = scaler.transform(X_test)       # transform test using train's parameters

# Print dimensions
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)
```

> 📖 **Why fit only on training data?** If you fit the scaler on the full dataset, test data "leaks" into training — the model has seen test statistics before predicting. Always `fit_transform` on train, `transform` only on test.
> **random_state=42** ensures reproducibility — same split every run.

🔗 [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

## 12. Resampling – Balancing Data

<!-- tags: imbalanced class balance oversample undersample resample majority minority -->

### When does data need balancing?
If the target classes are unequal — e.g. 85% "Existing Customer" vs 15% "Attrited" — the model will learn to always predict the majority class and still score 85% accuracy, even though it never learns the minority class.

### Undersampling (reduce majority)

```python
from sklearn.utils import resample

majority = df[df['Target'] == 'ClassA']
minority = df[df['Target'] == 'ClassB']

majority_downsampled = resample(majority,
                                n_samples=len(minority),
                                random_state=42)

df_balanced = pd.concat([majority_downsampled, minority])
print(df_balanced['Target'].value_counts())
```

### Oversampling (increase minority using SMOTE)

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print(pd.Series(y_resampled).value_counts())
```

> 📖 **Undersampling vs Oversampling:**
> - **Undersampling**: removes records from the majority class — fast, but loses data
> - **Oversampling (SMOTE)**: generates synthetic minority samples — preserves all data but adds artificial records
> - **When to balance**: if one class is more than ~60% of the data, balancing usually helps F1 score
> - **When NOT to balance**: if the imbalance reflects reality (e.g. fraud is rare) and you want the model to reflect that

🔗 [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## 13. Training Classifiers + K-Fold Cross Validation

<!-- tags: GaussianNB DecisionTree LogisticRegression SVM KNN cross_val_score kfold -->

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

classifiers = {
    'Naive Bayes':         GaussianNB(),
    'Decision Tree':       DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM':                 SVC(gamma='auto'),
    'KNN':                 KNeighborsClassifier()
}

k = 10  # number of folds

results = {}
for name, clf in classifiers.items():
    acc = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')
    f1  = cross_val_score(clf, X_train, y_train, cv=k, scoring='f1_weighted')
    results[name] = {'Accuracy': acc.mean(), 'F1': f1.mean()}
    print(f'{name:25s} | Accuracy: {acc.mean():.4f} | F1: {f1.mean():.4f}')

# Find best model
best_name = max(results, key=lambda x: results[x]['F1'])
print(f'\nBest model: {best_name}')
```

> 📖 **What is K-Fold Cross Validation?**
> The training data is split into k equal parts (folds). The model trains on k-1 folds and tests on the remaining 1. This repeats k times, each time using a different fold as the test. The scores are averaged.
> This gives a more reliable estimate of performance than a single train/test split, because every data point gets tested exactly once.

🔗 [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

---

## 14. Metrics Explained – F1, Precision, Recall, Accuracy

<!-- tags: f1 precision recall accuracy confusion matrix TP FP FN TN -->

> 📖 **The four core metrics — plain English:**

| Metric | Formula | Plain English |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Of all predictions, how many were correct overall? |
| **Precision** | TP / (TP + FP) | Of everything I predicted as positive, how many actually were? |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many did I catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance between precision and recall (use when classes are unequal) |

> 📖 **When to care about which metric:**
> - **Accuracy** is misleading on imbalanced data. Don't rely on it alone.
> - **Precision matters** when false positives are costly (e.g. spam filter — you don't want to delete real emails)
> - **Recall matters** when false negatives are costly (e.g. disease detection — you don't want to miss sick patients)
> - **F1 Score** balances both — best single metric for most exam comparisons

> 📖 **Confusion Matrix:**
> ```
>                  Predicted Positive    Predicted Negative
> Actual Positive       TP (correct)         FN (missed)
> Actual Negative       FP (false alarm)     TN (correct)
> ```
> - High **diagonal values** = good model
> - Off-diagonal values = errors

🔗 [sklearn metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 15. Learning Curves

<!-- tags: learning_curve overfitting underfitting training validation gap -->

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(clf, name, X, y, k=10):
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, cv=k, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10))

    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), '--', label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1),   '-',  label='Cross-validation score')
    plt.title(f'Learning Curve – {name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()

for name, clf in classifiers.items():
    plot_learning_curve(clf, name, X_train, y_train)
```

> 📖 **How to read a learning curve — the 3 patterns:**
>
> **Pattern 1 – Overfitting (High Variance):**
> Training score is high (e.g. 0.95), Cross-validation score is much lower (e.g. 0.65). Large gap between the two lines that does not close as training size increases. The model memorised the training data but doesn't generalise.
> → Fix: Add more data, reduce model complexity, add regularisation, use pruning.
>
> **Pattern 2 – Underfitting (High Bias):**
> Both training AND cross-validation scores are low and close together (both converge at a low value, e.g. 0.40). The model is too simple to learn the patterns.
> → Fix: Use a more complex model, add more features, reduce regularisation.
>
> **Pattern 3 – Good Fit:**
> Training score starts high and decreases slightly as more data is added. Cross-validation score starts lower but increases and converges close to the training score. Small gap between the two lines.

🔗 [sklearn learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)

---

## 16. Best Model Selection + Test Evaluation + Confusion Matrix

<!-- tags: best model F1 predict accuracy confusion_matrix classification_report -->

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select model with highest F1 (replace with actual best from section 13)
best_clf = classifiers[best_name]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

# Accuracy
print('Test Accuracy:', round(accuracy_score(y_test, y_pred), 4))

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Full classification report (precision, recall, F1 per class)
print('Classification Report:')
print(classification_report(y_test, y_pred))
```

> 📖 **Reading the classification report:**
> ```
>               precision    recall  f1-score   support
>
>     Class 0       0.88      0.91      0.89       150
>     Class 1       0.85      0.80      0.82       100
>
>   accuracy                           0.87       250
>  macro avg       0.86      0.86      0.86       250
> weighted avg     0.87      0.87      0.87       250
> ```
> - Each row = one class
> - `support` = number of actual records of that class in the test set
> - `macro avg` = simple average across classes (treats all equally)
> - `weighted avg` = average weighted by support (better for imbalanced data)

---

## 17. Dummy Classifier Comparison

<!-- tags: dummy classifier baseline DummyClassifier most_frequent comparison -->

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

print('Dummy Accuracy:  ', round(accuracy_score(y_test, dummy_pred), 4))
print('Model Accuracy:  ', round(accuracy_score(y_test, y_pred), 4))
```

```python
print("""
The dummy classifier always predicts the most frequent class, representing
a naive baseline with no actual learning. If our model's accuracy and F1
score are significantly higher than the dummy classifier's, the model is
learning real patterns from the data and offers meaningful predictive value.
If the gap is small, the model is barely better than random guessing and
further tuning or more informative features are needed.
""")
```

> 📖 **Why use a dummy classifier?** It sets the floor. If your dataset is 90% Class A, a dummy classifier scores 90% accuracy by always predicting A — but it's useless. Your real model must beat this convincingly, especially on F1.

🔗 [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

---

## 18. K-Means Clustering + Elbow + Silhouette + Labels

<!-- tags: KMeans elbow silhouette optimal k inertia labels cluster_centers unsupervised -->

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Pre-process: scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.select_dtypes(include='number'))

# Find optimal k using elbow and silhouette
inertia, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Elbow curve
plt.figure()
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)'); plt.ylabel('Inertia')
plt.title('Elbow Curve'); plt.show()

# Silhouette scores
plt.figure()
plt.plot(K_range, sil_scores, 'ro-')
plt.xlabel('Number of Clusters (k)'); plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k'); plt.show()

# Train with optimal k
optimal_k = 3   # <-- pick from plots
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

# Print labels and cluster centres
print('Cluster labels:', df['Cluster'].values)
print('Cluster centres:\n', km_final.cluster_centers_)
```

> 📖 **Reading the Elbow Curve:**
> The x-axis is k (number of clusters). The y-axis is inertia (sum of distances from each point to its cluster centre — lower = tighter clusters). Look for the "elbow" — the point where inertia stops dropping sharply and flattens out. That k value is optimal. It's a trade-off: more clusters always reduces inertia, but past the elbow you're just overfitting.

> 📖 **Reading the Silhouette Score:**
> Values range from -1 to 1. Higher = better. A score near 1 means points are well within their own cluster and far from others. The k with the **peak silhouette score** is optimal. Use both plots together — if they agree, you have high confidence in your k.

🔗 [sklearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) | [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

---

## 19. PCA – Reduce Dimensions + Explained Variance

<!-- tags: PCA dimensionality reduction explained_variance_ratio components transform -->

```python
from sklearn.decomposition import PCA
import pandas as pd

print('Original dimensions:', X_scaled.shape)

# Reduce to 2 components for visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print('PCA dimensions:', X_pca.shape)

# Principal components (the directions of maximum variance)
print('Principal components:\n', pca.components_)

# How much variance each component explains
print('Explained variance ratio:', pca.explained_variance_ratio_)
print('Total variance retained:', sum(pca.explained_variance_ratio_))
```

> 📖 **What is PCA?**
> PCA finds the directions (components) in which the data varies the most and projects all data points onto those directions. The first component (PC1) captures the most variance, PC2 captures the second most, and so on. Reducing to 2 components lets you plot high-dimensional data on a 2D scatter plot.

> 📖 **Reading explained variance:**
> If `explained_variance_ratio_ = [0.65, 0.20]`, PC1 explains 65% of the data's variance and PC2 explains 20% — together they retain 85%. Generally:
> - Above 80% retained = PCA was effective, visualisation is meaningful
> - Below 60% retained = significant information lost, scatter plot may not show true cluster separation

> 📖 **What to say in verbal analysis:**
> "PCA reduced the dataset from N dimensions to 2. The two principal components together explain X% of the total variance. This is [sufficient/insufficient] to meaningfully visualise the clusters. PC1 captures the largest source of variation in the data, and PC2 the second largest."

🔗 [sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---

## 20. PCA Scatter Plot with Cluster Colours

<!-- tags: scatter plot PCA seaborn clusters colours hue visualise -->

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Build PCA dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster'].values

# Scatter plot
sns.scatterplot(data=pca_df, x='PC1', y='PC2',
                hue='Cluster', palette='tab10', s=60)
plt.title('PCA Scatter Plot – Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
```

> 📖 **How to read the PCA scatter plot:**
> - Each dot = one data record projected into 2D space
> - Colour = cluster assigned by K-Means
> - **Well-separated coloured groups** = PCA was effective, clusters are distinct
> - **Overlapping colours** = either clusters are not truly distinct OR PCA lost too much variance to separate them in 2D
> - The axes (PC1, PC2) are abstract — they don't directly map to original features, but their directions represent the most important patterns in the data

---

## 21. Remove Features with >10 Unique Values (for loop)

<!-- tags: for loop drop unique values high cardinality remove columns nunique -->

```python
# Remove all columns that have more than 10 unique text values
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() > 10:
        df = df.drop(columns=[col])

df
```

```python
# Alternative: keep only columns with <= 10 unique text values
cols_to_drop = [col for col in df.select_dtypes(include='object').columns
                if df[col].nunique() > 10]
df = df.drop(columns=cols_to_drop)
df
```

> 📖 **Why remove high-cardinality columns?** Columns like "Patient Name" or "City" may have thousands of unique values. One-hot encoding them would create thousands of columns, making the model slow and prone to overfitting. It is better to drop them unless they can be meaningfully grouped.

---

## ⚡ Quick Verbal Answer Templates

### On learning curves (overfitting)
> "The model is **overfitting**. The training score is significantly higher than the cross-validation score, indicating that the model has memorised the training data but does not generalise well to unseen data. This high variance can be addressed by adding more training data, reducing model complexity, or applying regularisation."

### On learning curves (underfitting)
> "The model is **underfitting**. Both the training and cross-validation scores are low and converge at a poor accuracy, suggesting the model is too simple to capture the underlying patterns. A more complex algorithm or additional features may improve performance."

### On correlation
> "Feature X shows a strong positive correlation of [value] with the target variable, suggesting it is a useful predictor. Features A and B are highly correlated with each other, which may cause multicollinearity and could affect model stability."

### On PCA effectiveness
> "PCA successfully reduced the dataset from [N] dimensions to 2. The two components together explain [X]% of the variance. The scatter plot shows [well-separated / overlapping] clusters, indicating that PCA was [effective / partially effective] in preserving the structure needed to distinguish between groups."

### On dummy classifier
> "The dummy classifier achieves [X]% accuracy by always predicting the majority class, requiring no learning. Our best model achieves [Y]% accuracy and a weighted F1 score of [Z], which represents a meaningful improvement over the baseline, confirming that the model has learned genuine patterns in the data."

### On data balancing
> "The dataset is imbalanced — [class A] represents [X]% of records while [class B] represents only [Y]%. Training on imbalanced data causes the model to bias toward the majority class, resulting in high accuracy but poor recall for the minority class. [Undersampling / SMOTE oversampling] was applied to ensure equal class representation and improve the model's F1 score across all classes."

---

*End of reference document — good luck!*
