'''
1. Try to build a classifier for the MNIST dataset that achieves over 97%
accuracy on the test set. Hint: the KNeighborsClassifier works quite well
for this task; you just need to find good hyperparameter values (try a
grid search on the weights and n_neighbors hyperparameters).
'''
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
import time

mnist = fetch_openml('mnist_784', as_frame = False)
X, y = mnist.data, mnist.target

knn_model = KNeighborsClassifier()

knn_pipeline = Pipeline([
    ('knn', knn_model)
])

param_grid = [
    {
        'knn__n_neighbors':[2, 3, 4, 5, 6],
        'knn__weights':['uniform', 'distance']
    }
]

grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv = 3,
    scoring = 'accuracy'
)

train_size = 60000
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# t0 = time.time()
# grid_search.fit(X_train, y_train)
# t1 = time.time()
# t1 - t0
# cv_results = pd.DataFrame(grid_search.cv_results_)
# grid_search.best_estimator_.get_params()
# params = {'knn__n_neighbors': 4, 'knn__weights': 'distance'}


'''
2. Write a function that can shift an MNIST image in any direction (left,
right, up, or down) by one pixel. Then, for each image in the training
set, create four shifted copies (one per direction) and add them to the
training set. Finally, train your best model on this expanded training set
and measure its accuracy on the test set. You should observe that your
model performs even better now! This technique of artificially growing
the training set is called data augmentation or training set expansion.

Tip: You can use the shift() function from the scipy.ndimage.interpolation 
module. For example, shift(image, [2, 1], cval=0) shifts the image two pixels 
down and one pixel to the right.
'''
from scipy.ndimage import shift
from sklearn.model_selection import cross_val_score, StratifiedKFold

coordinates = [[1, 0], [0, 1], [-1, 0], [0, -1]]
augmented_train_data = X_train.reshape(len(X_train), 28, 28)
augmented_train_label = np.concatenate([y_train]*(len(coordinates) + 1))

def shift_digit(digit, coordinate):
    return  shift(digit.reshape(28, 28), coordinate)

for coordinate in coordinates: 
    shifted_digits = np.array([shift_digit(x, coordinate) for x in X_train])
    augmented_train_data = np.concatenate([augmented_train_data, shifted_digits])

knn_best_model = KNeighborsClassifier(n_neighbors=4, weights='distance')

knn_best_model.fit(augmented_train_data.reshape(len(augmented_train_data), -1), augmented_train_label)

cv = StratifiedKFold(n_splits=3)

augmented_train_data = augmented_train_data.reshape(len(augmented_train_data), -1)

'''
t0 = time.time()
cv_score = cross_val_score(
    knn_best_model,
    augmented_train_data,
    augmented_train_label,
    cv = cv,
    scoring = 'accuracy',
    n_jobs = 3,
    verbose = 10
)
t1 = time.time()
t1-t0


t00 = time.time()
augmented_accuracy = knn_best_model.fit(augmented_train_data, augmented_train_label)
augmented_accuracy = knn_best_model.score(X_test, y_test)
t11 = time.time()
print(f'cross_validation: {t1-t0}')
print(f'fit_score: {t11-t00}')
'''


'''
3. Tackle the Titanic dataset. A great place to start is on Kaggle.
Alternatively, you can download the data from https://homl.info/titanic.tgz 
and unzip this tarball like you did for the housing data in Chapter 2. 
This will give you two CSV files, train.csv and test.csv, which you can load
using pandas.read_csv(). The goal is to train a classifier that can predict 
the Survived column based on the other columns.
'''
from pathlib import Path
import urllib.request
import tarfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://homl.info/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic/train.csv")), pd.read_csv(Path("datasets/titanic/test.csv"))]

titanic_train, titanic_test = load_titanic_data()

titanic_train.info()
titanic_test.info()

train_feats = titanic_train.drop(columns = ['Survived', 'Cabin', 'Ticket', 'Name']).set_index('PassengerId')
train_label = titanic_train.Survived

test_feats = titanic_test.set_index('PassengerId').loc[:, train_feats.columns]

# Making Sex a numerical categorical value
train_feats['Sex'] = train_feats.Sex.replace({'male':0, 'female':1}).infer_objects(copy=False)
test_feats['Sex'] = test_feats.Sex.replace({'male':0, 'female':1}).infer_objects(copy=False)

# Preprocessing transformers
oh_encoder = OneHotEncoder(sparse_output=False) 
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

num_pipeline = Pipeline([
    ('impute', num_imputer),
    ('scale', scaler)
])

cat_pipeline = Pipeline([
    ('impute', cat_imputer),
    ('encode', oh_encoder)
])

categorical_columns = ['Embarked']
numerical_columns = [x for x in train_feats.columns if x not in categorical_columns]

preprocessing = ColumnTransformer([
    ('numerical', num_pipeline, numerical_columns),
    ('categorical', cat_pipeline, categorical_columns)
])

titanic_train_preprocessed = preprocessing.fit_transform(train_feats)
titanic_test_preprocessed = preprocessing.fit_transform(test_feats)

# SGD
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(titanic_train_preprocessed, train_label)
sgd_pred = sgd_clf.predict(titanic_train_preprocessed)

# KNN
knn_model = KNeighborsClassifier()

knn_pipeline = Pipeline([
    ('knn', knn_model)
])

param_grid = [
    {
        'knn__n_neighbors':[1, 2, 4, 10, 12, 20],
        'knn__weights':['uniform', 'distance']
    }
]

grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv = 3,
    scoring = 'accuracy'
)

t0 = time.time()
grid_search.fit(titanic_train_preprocessed, train_label)
t1 = time.time()
t1 - t0
cv_results = pd.DataFrame(grid_search.cv_results_)
grid_search.best_estimator_.get_params()

final_model = grid_search.best_estimator_
knn_pred = final_model.predict(titanic_train_preprocessed)

# Random Forest
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(titanic_train_preprocessed, train_label)
rf_pred = rf_clf.predict(titanic_train_preprocessed)


sample_weight = (knn_pred != train_label)
ConfusionMatrixDisplay.from_predictions(
    train_label, knn_pred, normalize='pred', values_format='.0%',
)
plt.show()

'''
4. Build a spam classifier (a more challenging exercise):
    a. Download examples of spam and ham from Apache
    SpamAssassin's public datasets.

    b. Unzip the datasets and familiarize yourself with the data format.

    c. Split the data into a training set and a test set.

    d. Write a data preparation pipeline to convert each email into a
    feature vector. Your preparation pipeline should transform an email
    into a (sparse) vector that indicates the presence or absence of each
    possible word. For example, if all emails only ever contain four
    words, “Hello”, “how”, “are”, “you”, then the email “Hello you
    Hello Hello you” would be converted into a vector [1, 0, 0, 1]
    (meaning [“Hello” is present, “how” is absent, “are” is absent,
    “you” is present]), or [3, 0, 0, 2] if you prefer to count the number
    of occurrences of each word.
    You may want to add hyperparameters to your preparation pipeline
    to control whether or not to strip off email headers, convert each
    email to lowercase, remove punctuation, replace all URLs with
    “URL”, replace all numbers with “NUMBER”, or even perform
    stemming (i.e., trim off word endings; there are Python libraries
    available to do this).

    e. Finally, try out several classifiers and see if you can build a great
    spam classifier, with both high recall and high precision.
'''