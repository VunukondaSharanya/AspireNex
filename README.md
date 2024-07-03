#README: Titanic Survival Prediction

#Author: Vunukonda Sharanya

#Domain: Data Science

#Aim

The aim of this project is to develop a model that predicts whether a passenger on the Titanic survived or not based on various features such as age, sex, passenger class, and other relevant attributes.

#Libraries Used

The following important libraries were used for this project:

- numpy
- pandas
- sklearn (RandomForestClassifier, KNeighborsClassifier, LogisticRegression, StandardScaler, train_test_split, cross_val_score, classification_report, confusion_matrix)
- matplotlib.pyplot
- seaborn

## Dataset

The Titanic dataset was loaded using pandas, which contains information about passengers, including Passenger ID, survival status, ticket class, name, sex, age, number of siblings/spouses, number of parents/children, ticket number, fare, cabin, and port of embarkation.

## Data Exploration and Preprocessing

1. The dataset was loaded using pandas `read_csv` function as a DataFrame, and its first 5 rows were displayed using `df.head()`.
2. Missing values in the dataset were checked using `df.isna().sum()`.
3. Missing values in the 'Age' and 'Fare' columns were filled with their respective median values.
4. The 'Cabin' column, which had excessive missing values, was dropped from the dataset.
5. The 'Sex' and 'Embarked' columns were encoded to numerical values using `pd.factorize`.

## Data Visualization

1. Various plots were created using `seaborn` and `matplotlib.pyplot` to explore the relationship between different features and survival status.
2. Histograms, bar plots, and scatter plots were used to visualize the distribution of data and identify any patterns or trends.

## Feature Engineering

1. Additional features were engineered from the existing data to improve model performance.
2. Features like family size and title were created from the 'SibSp', 'Parch', and 'Name' columns.

## Model Building and Evaluation

1. The dataset was split into training and testing sets using `train_test_split`.
2. Three different machine learning models were trained: Random Forest, KNN, and Logistic Regression.
3. Each model was evaluated using cross-validation to determine its accuracy.
4. The best model was selected based on cross-validation results, and its performance was evaluated using accuracy, classification report, and confusion matrix.

---

# README: Iris Flower Classification

## Author: [Your Name]

## Batch: July

## Domain: Data Science

## Aim

The aim of this project is to develop a model that can classify iris flowers into different species based on their sepal and petal measurements.

## Libraries Used

The following important libraries were used for this project:

- numpy
- pandas
- sklearn (RandomForestClassifier, KNeighborsClassifier, LogisticRegression, StandardScaler, train_test_split, cross_val_score, classification_report, confusion_matrix)
- matplotlib.pyplot
- seaborn

## Dataset

The iris dataset was loaded using seaborn's `load_dataset` function, which contains information about iris flowers, including sepal length, sepal width, petal length, petal width, and species.

## Data Exploration and Preprocessing

1. The dataset was loaded using seaborn's `load_dataset` function as a DataFrame, and its first 5 rows were displayed using `df.head()`.
2. The 'species' column in the DataFrame was encoded to numerical values using `pd.factorize(df['species'])`.
3. Descriptive statistics for the dataset were displayed using `df.describe()`.
4. Missing values in the dataset were checked using `df.isna().sum()`.

## Data Visualization

1. Various plots were created using `seaborn` and `matplotlib.pyplot` to explore the relationship between different features and species.
2. 3D scatter plots were created to visualize the relationship between species, petal length, and petal width, as well as between species, sepal length, and sepal width.
3. 2D scatter plots were created to visualize the relationship between species and sepal length, as well as between species and sepal width.

## Model Building and Evaluation

1. The dataset was split into training and testing sets using `train_test_split`.
2. Three different machine learning models were trained: Random Forest, KNN, and Logistic Regression.
3. Each model was evaluated using cross-validation to determine its accuracy.
4. The best model was selected based on cross-validation results, and its performance was evaluated using accuracy, classification report, and confusion matrix.
