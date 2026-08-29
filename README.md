<div align="center">

# 🚢 Titanic Survival Prediction

### Predicting passenger survival on the Titanic using Logistic Regression — built as a leakage-free, production-style ML pipeline.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## 📌 Overview

The Titanic sank in 1912. This project uses passenger data — age, gender, ticket class, fare, and family details — to predict whether a passenger **survived** or **did not survive**, using **Logistic Regression**.

But this isn't just "load data → fit model." It's built the way a real ML pipeline should be:

> ✅ No data leakage &nbsp;&nbsp; ✅ Feature engineering &nbsp;&nbsp; ✅ Cross-validation &nbsp;&nbsp; ✅ Hyperparameter tuning &nbsp;&nbsp; ✅ Full evaluation

---

## 🎯 Problem Statement

| | |
|---|---|
| **Type** | Binary Classification |
| **Target** | `Survived` → `1` = Survived, `0` = Did Not Survive |
| **Algorithm** | Logistic Regression |
| **Goal** | Predict survival from passenger attributes as accurately and honestly as possible |

---

## 🗂️ Dataset

| Column | Meaning |
|---|---|
| `PassengerId` | Unique ID for each passenger *(dropped — not predictive)* |
| `Survived` | Target variable — 1 = Survived, 0 = Did not survive |
| `Pclass` | Ticket class — 1 = Upper, 2 = Middle, 3 = Lower |
| `Name` | Full name *(used to engineer `Title`)* |
| `Sex` | Male / Female |
| `Age` | Age in years *(has missing values)* |
| `SibSp` | # of siblings/spouse aboard |
| `Parch` | # of parents/children aboard |
| `Ticket` | Ticket number *(dropped — not predictive)* |
| `Fare` | Ticket price |
| `Cabin` | Cabin number *(dropped — mostly missing)* |
| `Embarked` | Port of boarding — C / Q / S |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas / NumPy** — data wrangling
- **Matplotlib / Seaborn** — visualization
- **scikit-learn** — `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, `GridSearchCV`

---

## 🔄 Project Workflow

```
Load Data → EDA → Feature Engineering → Train/Test Split
        → Preprocessing Pipeline (Impute + Scale + Encode)
        → Cross-Validation → Hyperparameter Tuning
        → Final Model → Evaluation → Feature Importance
```

### 1️⃣ Exploratory Data Analysis (EDA)
Visualized survival rate by sex, class, age, and boarding port to understand the story in the data before touching it.

### 2️⃣ Feature Engineering
- 🏷️ **Title** extracted from `Name` (Mr, Mrs, Miss, Master, Rare)
- 👨‍👩‍👧 **FamilySize** = `SibSp` + `Parch` + 1
- 🧍 **IsAlone** flag for solo travelers

### 3️⃣ Train/Test Split — *Before* Any Learning Step
The dataset is split **first**. Only *after* this does any statistic (median, mode, scaling values) get calculated — and only from the **training set**. This prevents **data leakage**.

### 4️⃣ Preprocessing Pipeline
| Feature type | Missing value strategy | Extra step |
|---|---|---|
| Numeric (`Age`, `Fare`, `FamilySize`) | Median (train-only) | `StandardScaler` |
| Categorical (`Pclass`, `Sex`, `Embarked`, `Title`, `IsAlone`) | Most frequent (train-only) | `OneHotEncoder` |

### 5️⃣ Model Building
- 5-Fold **Stratified Cross-Validation** for reliable performance estimates
- **GridSearchCV** to tune the regularization strength (`C`)
- Final model retrained with the best parameters

### 6️⃣ Evaluation
- Accuracy, ROC-AUC, Confusion Matrix, Precision/Recall, ROC Curve
- Feature importance via Logistic Regression coefficients

---

## 📊 Key Insights

| Insight | Direction |
|---|---|
| 👩 Being **female** | 🔼 Strongly increases survival chances |
| 🎩 **1st Class** ticket | 🔼 Increases survival chances |
| 👨 Being an **adult male** | 🔽 Strongly decreases survival chances |
| 🚢 **3rd Class**, traveling alone | 🔽 Decreases survival chances |

> These results match the real historical account: *"women and children first."*

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook titanic_logistic_regression.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── titanic_logistic_regression.ipynb   # Main notebook — full pipeline
├── train.csv                           # Dataset
├── README.md                           # You are here
└── requirements.txt                    # Dependencies
```

---

## 🧠 Why This Project Is Different

Most beginner Titanic notebooks fill in missing values **before** splitting the data — which quietly leaks test-set information into training and makes results look better than they'd be in real life.

This project fixes that by:
- Splitting the data **first**
- Doing all imputation, scaling, and encoding **inside a `Pipeline`, fit only on training data**
- Validating with **cross-validation**, not a single lucky split
- Tuning with **GridSearchCV** instead of guessing settings

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | *fill in after running* |
| ROC-AUC | *fill in after running* |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](../../issues).

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and share.

---

<div align="center">

Made with 🐍 Python and a healthy respect for data leakage.

⭐ If you found this useful, consider giving it a star!

</div>
