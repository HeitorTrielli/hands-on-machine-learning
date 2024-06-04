"""
1. Try to build a classifier for the MNIST dataset that achieves over 97%
accuracy on the test set. Hint: the KNeighborsClassifier works quite well
for this task; you just need to find good hyperparameter values (try a
grid search on the weights and n_neighbors hyperparameters).
"""

import email.message
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# mnist = fetch_openml("mnist_784", as_frame=False)
# X, y = mnist.data, mnist.target

# knn_model = KNeighborsClassifier()

# knn_pipeline = Pipeline([("knn", knn_model)])

# param_grid = [
#     {"knn__n_neighbors": [2, 3, 4, 5, 6], "knn__weights": ["uniform", "distance"]}
# ]

# grid_search = GridSearchCV(knn_pipeline, param_grid, cv=3, scoring="accuracy")

# train_size = 60000
# X_train, X_test, y_train, y_test = (
#     X[:train_size],
#     X[train_size:],
#     y[:train_size],
#     y[train_size:],
# )

# t0 = time.time()
# grid_search.fit(X_train, y_train)
# t1 = time.time()
# t1 - t0
# cv_results = pd.DataFrame(grid_search.cv_results_)
# grid_search.best_estimator_.get_params()
# params = {"knn__n_neighbors": 4, "knn__weights": "distance"}


"""
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
"""
from scipy.ndimage import shift
from sklearn.model_selection import StratifiedKFold, cross_val_score

# coordinates = [[1, 0], [0, 1], [-1, 0], [0, -1]]
# augmented_train_data = X_train.reshape(len(X_train), 28, 28)
# augmented_train_label = np.concatenate([y_train] * (len(coordinates) + 1))


# def shift_digit(digit, coordinate):
#     return shift(digit.reshape(28, 28), coordinate)


# for coordinate in coordinates:
#     shifted_digits = np.array([shift_digit(x, coordinate) for x in X_train])
#     augmented_train_data = np.concatenate([augmented_train_data, shifted_digits])

# knn_best_model = KNeighborsClassifier(n_neighbors=4, weights="distance")

# knn_best_model.fit(
#     augmented_train_data.reshape(len(augmented_train_data), -1), augmented_train_label
# )

# cv = StratifiedKFold(n_splits=3)

# augmented_train_data = augmented_train_data.reshape(len(augmented_train_data), -1)

# t0 = time.time()
# cv_score = cross_val_score(
#     knn_best_model,
#     augmented_train_data,
#     augmented_train_label,
#     cv=cv,
#     scoring="accuracy",
#     n_jobs=3,
#     verbose=10,
# )
# t1 = time.time()
# t1 - t0


# t00 = time.time()
# augmented_accuracy = knn_best_model.fit(augmented_train_data, augmented_train_label)
# augmented_accuracy = knn_best_model.score(X_test, y_test)
# t11 = time.time()
# print(f"cross_validation: {t1-t0}")
# print(f"fit_score: {t11-t00}")


"""
3. Tackle the Titanic dataset. A great place to start is on Kaggle.
Alternatively, you can download the data from https://homl.info/titanic.tgz 
and unzip this tarball like you did for the housing data in Chapter 2. 
This will give you two CSV files, train.csv and test.csv, which you can load
using pandas.read_csv(). The goal is to train a classifier that can predict 
the Survived column based on the other columns.
"""
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# def load_titanic_data():
#     tarball_path = Path("datasets/titanic.tgz")
#     if not tarball_path.is_file():
#         Path("datasets").mkdir(parents=True, exist_ok=True)
#         url = "https://homl.info/titanic.tgz"
#         urllib.request.urlretrieve(url, tarball_path)
#         with tarfile.open(tarball_path) as housing_tarball:
#             housing_tarball.extractall(path="datasets")
#     return [
#         pd.read_csv(Path("datasets/titanic/train.csv")),
#         pd.read_csv(Path("datasets/titanic/test.csv")),
#     ]


# titanic_train, titanic_test = load_titanic_data()

# titanic_train.info()
# titanic_test.info()

# train_feats = titanic_train.drop(
#     columns=["Survived", "Cabin", "Ticket", "Name"]
# ).set_index("PassengerId")
# train_label = titanic_train.Survived

# test_feats = titanic_test.set_index("PassengerId").loc[:, train_feats.columns]

# # Making Sex a numerical categorical value
# train_feats["Sex"] = train_feats.Sex.replace({"male": 0, "female": 1}).infer_objects(
#     copy=False
# )
# test_feats["Sex"] = test_feats.Sex.replace({"male": 0, "female": 1}).infer_objects(
#     copy=False
# )

# # Preprocessing transformers
# oh_encoder = OneHotEncoder(sparse_output=False)
# cat_imputer = SimpleImputer(strategy="most_frequent")
# num_imputer = SimpleImputer(strategy="median")
# scaler = StandardScaler()

# num_pipeline = Pipeline([("impute", num_imputer), ("scale", scaler)])

# cat_pipeline = Pipeline([("impute", cat_imputer), ("encode", oh_encoder)])

# categorical_columns = ["Embarked"]
# numerical_columns = [x for x in train_feats.columns if x not in categorical_columns]

# preprocessing = ColumnTransformer(
#     [
#         ("numerical", num_pipeline, numerical_columns),
#         ("categorical", cat_pipeline, categorical_columns),
#     ]
# )

# titanic_train_preprocessed = preprocessing.fit_transform(train_feats)
# titanic_test_preprocessed = preprocessing.fit_transform(test_feats)

# # SGD
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(titanic_train_preprocessed, train_label)
# sgd_pred = sgd_clf.predict(titanic_train_preprocessed)

# # KNN
# knn_model = KNeighborsClassifier(n_neighbors=1)

# knn_pipeline = Pipeline([("knn", knn_model)])

# param_grid = [
#     {
#         "knn__n_neighbors": [1, 2, 4, 10, 12, 20, 50],
#         "knn__weights": ["uniform", "distance"],
#     }
# ]

# knn_grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring="accuracy")

# t0 = time.time()
# knn_grid_search.fit(titanic_train_preprocessed, train_label)
# t1 = time.time()
# t1 - t0
# cv_results = pd.DataFrame(knn_grid_search.cv_results_)

# cv_socre_sgd = cross_val_score(
#     knn_model, titanic_train_preprocessed, train_label, cv=3, scoring="accuracy"
# )

# final_model = knn_grid_search.best_estimator_
# knn_pred = final_model.predict(titanic_train_preprocessed)

# # Random Forest
# rf_clf = RandomForestClassifier(random_state=42)

# rf_pipeline = Pipeline([("rf", rf_clf)])

# param_grid = [
#     {
#         "rf__n_estimators": [1, 12, 20, 50, 100, 150, 500],
#         "rf__criterion": ["gini", "entropy", "log_loss"],
#     }
# ]

# rf_grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="accuracy")

# t0 = time.time()
# rf_grid_search.fit(titanic_train_preprocessed, train_label)
# t1 = time.time()
# t1 - t0
# cv_results = pd.DataFrame(rf_grid_search.cv_results_)

# cv_socre_sgd = cross_val_score(
#     knn_model, titanic_train_preprocessed, train_label, cv=3, scoring="accuracy"
# )

# final_model = rf_grid_search.best_estimator_
# rf_pred = final_model.predict(titanic_train_preprocessed)

# # Best model was Random Forest
# ConfusionMatrixDisplay.from_predictions(train_label, rf_pred)
# plt.show()

"""
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
"""
import email
import email.policy
import os
import re
import string
from random import sample, shuffle

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from scipy.sparse import lil_matrix


# a) (I copied this one straight from the answer, because I was not sure
# where to find the data, then I just didn't want to make another script
# for downloading more data. I already do this as a job haha)
def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (
        ("easy_ham", "ham", ham_url),
        ("spam", "spam", spam_url),
    ):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            tar_bz2_file = tarfile.open(path)
            tar_bz2_file.extractall(path=spam_path)
            tar_bz2_file.close()
    return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]


ham_dir, spam_dir = fetch_spam_data()

# b) I had to look the answer for this also because I was not familiarized with
# the email library


def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_files = [ham_dir / Path(x) for x in os.listdir(ham_dir) if x != "cmds"]
spam_files = [spam_dir / Path(x) for x in os.listdir(spam_dir) if x != "cmds"]

ham_mails = [load_email(ham) for ham in ham_files]
spam_mails = [load_email(spam) for spam in spam_files]

# c)

ham_labeled = list(zip(ham_mails, len(ham_mails) * [0]))
spam_labeled = list(zip(spam_mails, len(spam_mails) * [1]))

mails = ham_labeled + spam_labeled


def stratified_sets(ham, spam, train_proportion):
    def stratify(set, train_proportion=train_proportion):
        train_size = int(len(set) * train_proportion)
        train_set = sample(set, train_size)
        test_set = [x for x in set if x not in train_set]

        return train_set, test_set

    train_ham, test_ham = stratify(ham)
    train_spam, test_spam = stratify(spam)

    train_set = train_ham + train_spam
    test_set = test_ham + test_spam
    shuffle(train_set)
    shuffle(test_set)
    return train_set, test_set


train_set, test_set = stratified_sets(ham_labeled, spam_labeled, 0.75)

# d)
# ([], 0, 0) => (text_content, html indicator, attachment indicator)
train_set_feats = [[x[0], [], 0, 0] for x in train_set]
test_set_feats = [[x[0], [], 0, 0] for x in test_set]
train_set_labels = [x[1] for x in train_set]
test_set_labels = [x[1] for x in test_set]


def decode_payload(mail):
    payload = mail.get_payload(decode=True)
    charset = mail.get_content_charset()
    if (
        charset == "unknown-8bit"
        or charset == "default"
        or charset == "default_charset"
    ):
        return payload.decode("latin1")
    return payload.decode(charset or "latin1", errors="replace")


def parse_feats(feat_set):
    feat_set = feat_set.copy()
    for i, feats in enumerate(feat_set):
        mail = feats[0]
        if mail.is_multipart():
            for part in mail.iter_parts():
                if part.get_content_type() == "text/plain":
                    feat_set[i][1] += [" " + decode_payload(part)]
                elif part.get_content_type() == "text/html":
                    feat_set[i][2] = 1
                    content = decode_payload(part)
                    html_text = BeautifulSoup(content, "lxml").text
                    feat_set[i][1] += [" " + html_text]
                else:
                    feat_set[i][3] = 1

        else:
            if mail.get_content_type() == "text/plain":
                feat_set[i][1] = decode_payload(mail)
            elif mail.get_content_type() == "text/html":
                feat_set[i][2] = 1
                content = decode_payload(mail)
                html_text = BeautifulSoup(content, "lxml").text
                feat_set[i][1] += [" " + html_text]
            else:
                feat_set[i][3] = 1

        if type(feat_set[i][1]) == list:
            feat_set[i][1] = " ".join(feat_set[i][1])

    feat_set_df = pd.DataFrame(
        feat_set, columns=["mail", "content", "has_html", "has_attachment"]
    ).drop(columns="mail")

    return feat_set_df


train_set_df = parse_feats(train_set_feats)
test_set_df = parse_feats(test_set_feats)


def remove_nonprintable(text):
    printable = set(string.printable)
    cleaned_text = " ".join(filter(lambda x: x in printable, text))
    return cleaned_text


def remove_linebreak(text):
    return text.replace("\n", " ")


def replace_url(text):
    url_pattern = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        r"www\\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    replaced_text = re.sub(url_pattern, "URL", text)
    return replaced_text


def replace_email(text):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    replaced_text = re.sub(email_pattern, "EMAIL", text)
    return replaced_text


def replace_punctuation(text):
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)
    return text


def text_to_vector(text):
    text = text.split(" ")
    return list(set([str.strip(x) for x in text if x != ""]))


def stem_vector(vector):
    ps = PorterStemmer()
    return list(set(ps.stem(word) for word in vector))


def preprocess_text(text):
    text = text.lower()
    text = remove_linebreak(text)
    text = replace_email(text)
    text = replace_url(text)
    text = replace_punctuation(text)
    text_vector = text_to_vector(text)
    stemmed_vector = stem_vector(text_vector)
    return stemmed_vector


def vectors_to_sparse(vector_list, total_data: pd.DataFrame = None):
    if type(total_data) != pd.Series:
        raise ValueError(
            "Please, pass all the dataset normalized vectors as the total_data Series"
        )
    possible_words = set(word for sublist in total_data for word in sublist)

    word_to_index = {word: index for index, word in enumerate(possible_words)}

    word_matrix_mapping = [[word_to_index[x] for x in vector] for vector in vector_list]

    sparse_matrix = lil_matrix((len(vector_list), len(possible_words)), dtype=int)

    for idx, vector in enumerate(word_matrix_mapping):
        for word in vector:
            sparse_matrix[idx, word] = 1

    return sparse_matrix


content_to_vector = FunctionTransformer(lambda x: x.apply(preprocess_text))

test_vector = content_to_vector.transform(test_set_df.content)
train_vector = content_to_vector.transform(train_set_df.content)

total_data = pd.concat([train_vector, test_vector])

vector_to_sparse = FunctionTransformer(
    lambda x: vectors_to_sparse(x, total_data=total_data)
)

sparse_content_pipe = Pipeline(
    [("to_vector", content_to_vector), ("to_sparse", vector_to_sparse)]
)

transform_content = ColumnTransformer(
    [("sparse_pipe", sparse_content_pipe, "content")], remainder="passthrough"
)

transformed_train_data = transform_content.fit_transform(train_set_df)
transformed_test_data = transform_content.fit_transform(test_set_df)

# e)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
sgd = SGDClassifier()
lr = LogisticRegression()

# cross_val_score(knn, transformed_train_data, train_set_labels, cv=10)

from sklearn.metrics import precision_score, recall_score

lr.fit(transformed_train_data, train_set_labels)

y_pred = lr.predict(transformed_test_data)

print(f"Precision: {precision_score(test_set_labels, y_pred):.2%}")
print(f"Recall: {recall_score(test_set_labels, y_pred):.2%}")

# sgd and lr were the best
