import time
from scipy.stats import randint, uniform
from random import sample
import pandas as pd
import numpy as np
from chapter_2.data_handling import load_housing_data
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
import chapter_2.pipelines as mypipes


# Doing the data cleaning for the exercises
housing = load_housing_data()

preprocessing = ColumnTransformer(
    [
        ('log', mypipes.log_pipeline, ['population']),
        ('cluster', mypipes.cluster_similarity, ['latitude', 'longitude']),
        ('cat', mypipes.cat_pipelinne, housing.dtypes.loc[housing.dtypes == object].index.to_list())
    ],
    remainder = mypipes.num_pipeline
)


# Creating training and testing sets
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
    labels = [1, 2, 3, 4, 5],
)

train_set, test_set = (
    train_test_split(
        housing, 
        test_size = 0.75, 
        random_state = 42, 
        stratify = housing.income_cat
    )
)

# Removing the income_cat column, since we already have stratified sets
train_set, test_set = (x.drop('income_cat', axis = 1) for x in (train_set, test_set))

train_set_label = train_set.median_house_value
train_set_labeless = train_set.drop(columns = 'median_house_value')

# Exercise 1
''' 
1. Try a support vector machine regressor (sklearn.svm.SVR) with various 
hyperparameters, such as kernel="linear" (with various values for the C
hyperparameter) or kernel="rbf" (with various values for the C and gamma 
hyperparameters). Note that support vector machines don't scale well to 
large datasets, so you should probably train your model on just the first 
5,000 instances of the training set and use only 3-fold crossvalidation, 
or else it will take hours. Don't worry about what the hyperparameters 
mean for now; we'll discuss them in Chapter 5. How does the best SVR 
predictor perform?
'''

svr_model = SVR()

svr_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('svr', svr_model)
])


param_grid = [
    {
        'preprocessing__cluster__n_clusters': [5, 8, 10, 15],
        'svr__gamma':['scale', 'auto', 0.5],
        'svr__C':[1, 2, 5, 0.5]
    },

    {
        'preprocessing__cluster__n_clusters': [5, 8, 10, 15],
        'svr__kernel':['linear'],
        'svr__C':[1, 2, 5, 0.5]
     },
]


grid_search = GridSearchCV(
    svr_pipeline,
    param_grid,
    cv = 3,
    scoring = 'neg_root_mean_squared_error'
)

t0 = time.time()
grid_search.fit(train_set_labeless, train_set_label)
t1 = time.time()
t1 - t0

best_param_predict = grid_search.predict(test_set.drop(columns = 'median_house_value'))
test_set['prediction'] = best_param_predict
test_set[['median_house_value', 'prediction']].describe()
cv_results = pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score')
final_model = grid_search.best_estimator_
final_prediction = final_model.predict(housing.drop(columns = ['median_house_value', 'income_cat']))
housing['prediction'] = final_prediction
housing[['median_house_value', 'prediction']].describe()


# Exercise 2
'''
2. Try replacing the GridSearchCV with a RandomizedSearchCV.
'''
param_distribs = [
    {
        'preprocessing__cluster__n_clusters': randint(low=3, high=50),
        'svr__C':uniform(0.5, 15),
        'svr__gamma':sample(['scale', 'auto', 0.5], 1)           
    },
    {
        'preprocessing__cluster__n_clusters': randint(low=3, high=50),
        'svr__C':uniform(0.5, 15),
        'svr__kernel':sample(['linear'], 1)           
    }
]

rnd_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions = param_distribs,
    n_iter = 64,
    cv = 3,
    random_state = 27,
    scoring = 'neg_root_mean_squared_error'
)

t0 = time.time()
rnd_search.fit(train_set_labeless, train_set_label)
t1 = time.time()
t1 - t0

best_param_predict = rnd_search.predict(test_set.drop(columns = 'median_house_value'))
test_set['prediction'] = best_param_predict
test_set[['median_house_value', 'prediction']].describe()

cv_results = pd.DataFrame(rnd_search.cv_results_).sort_values('rank_test_score')
final_model = rnd_search.best_estimator_

final_prediction = final_model.predict(housing.drop(columns = ['median_house_value', 'income_cat']))
housing['prediction'] = final_prediction
housing[['median_house_value', 'prediction']].describe()


# Exercise 3
'''
3. Try adding a SelectFromModel transformer in the preparation pipeline
to select only the most important attributes.
'''

adapted_srv_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('feature_importance', SelectFromModel(estimator = SVR(kernel = 'linear'))),
    ('svr', svr_model)
])

adapted_rnd_search = RandomizedSearchCV(
    adapted_srv_pipeline,
    param_distributions = param_distribs,
    n_iter = 64,
    cv = 3,
    random_state = 27,
    scoring = 'neg_root_mean_squared_error'
)

t0 = time.time()
adapted_rnd_search.fit(train_set_labeless, train_set_label)
t1 = time.time()
t1 - t0

best_param_predict = adapted_rnd_search.predict(test_set.drop(columns = 'median_house_value'))
test_set['prediction'] = best_param_predict
test_set[['median_house_value', 'prediction']].describe()

cv_results = pd.DataFrame(adapted_rnd_search.cv_results_).sort_values('rank_test_score')
final_model = adapted_rnd_search.best_estimator_

final_prediction = final_model.predict(housing.drop(columns = ['median_house_value', 'income_cat']))
housing['prediction'] = final_prediction
housing[['median_house_value', 'prediction']].describe()


# Exercise 4
'''
4. Try creating a custom transformer that trains a k-nearest neighbors
regressor (sklearn.neighbors.KNeighborsRegressor) in its fit() method,
and outputs the model's predictions in its transform() method. Then add
this feature to the preprocessing pipeline, using latitude and longitude as
the inputs to this transformer. This will add a feature in the model that
corresponds to the housing median price of the nearest districts.
'''


class KNeighborsTrasnform(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors = 5, weights = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        
    def fit(self, X, y=None):
        self.knn_ = KNeighborsRegressor(n_neighbors = self.n_neighbors, weights =  self.weights)
        self.knn_ = self.knn_.fit(X, y)
        return self # always return self!

    def transform(self, X):
        check_is_fitted(self)
        predictions = self.knn_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1) # It has to return 2d array
        return predictions


    def get_feature_names_out(self, names=None):
        n_outputs = getattr(self.knn_, "n_outputs_", 1)
        return [f"knn_regression_{i}" for i in range(n_outputs)]


knn_transform = KNeighborsTrasnform(n_neighbors = 3, weights = "distance")
knn_transform.get_feature_names_out()

preprocessing_knn = ColumnTransformer(
    [
        ('log', mypipes.log_pipeline, ['population']),
        ('knn', knn_transform, ['latitude', 'longitude']),
        ('cat', mypipes.cat_pipelinne, housing.dtypes.loc[housing.dtypes == object].index.to_list())
    ],
    remainder = mypipes.num_pipeline
)

svr_model = SVR(C = 3, kernel = 'rbf')

svr_knn_pipeline = Pipeline([
    ('preprocessing', preprocessing_knn),
    ('svr', svr_model)
])

-cross_val_score(svr_knn_pipeline, train_set_labeless, train_set_label, scoring = 'neg_root_mean_squared_error', cv = 5)


# Exercise 5
'''
5. Automatically explore some preparation options using GridSearchCV.
'''

param_grid_knn = [
    {
        'svr__C':[1, 2, 5, 10, 50, 150, 300],
        'preprocessing__knn__n_neighbors':[3, 5, 10, 40, 100],
        'preprocessing__knn__weights':['distance', 'uniform']
     },
]

grid_search_knn = GridSearchCV(
    svr_knn_pipeline,
    param_grid_knn,
    cv = 3,
    scoring = 'neg_root_mean_squared_error',
    verbose = 10
)

t0 = time.time()
grid_search_knn.fit(train_set_labeless, train_set_label)
t1 = time.time()
t1 - t0

best_param_predict = grid_search_knn.predict(test_set.drop(columns = 'median_house_value'))
test_set['prediction'] = best_param_predict
test_set[['median_house_value', 'prediction']].describe()

cv_results = pd.DataFrame(grid_search_knn.cv_results_).sort_values('rank_test_score')
final_model = grid_search_knn.best_estimator_

final_prediction = final_model.predict(housing.drop(columns = ['median_house_value', 'income_cat']))
housing['prediction'] = final_prediction
housing[['median_house_value', 'prediction']].describe()


# Exercise 6
'''
6. Try to implement the StandardScalerClone class again from scratch,
then add support for the inverse_transform() method: executing scaler.
inverse_transform(scaler.fit_transform(X)) should return an array very
close to X. Then add support for feature names: set feature_names_in_
in the fit() method if the input is a DataFrame. This attribute should be a
NumPy array of column names. Lastly, implement the
get_feature_names_out() method: it should have one optional
input_features=None argument. If passed, the method should check that
its length matches n_features_in_, and it should match
feature_names_in_ if it is defined; then input_features should be
returned. If input_features is None, then the method should either return
feature_names_in_ if it is defined or np.array(["x0", "x1", ...]) with
length n_features_in_ otherwise.
'''
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean = True):
        self.with_mean = with_mean
    
    def fit(self, X, y = None):
        if type(X) == pd.DataFrame:
            self.feature_names_in_ = np.array(X.columns)

        X = check_array(X)
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        self.n_features_in_ = np.shape(X)[1]

        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        if self.with_mean:
            X = X - self.mean_

        return X/self.std_
    
    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        X = X*self.std_

        return X + self.mean_ if self.with_mean else X
    
    def get_feature_names_out(self, input_features = None):
        if input_features:
            check_is_fitted(self)
            
            assert len(input_features) == self.n_features_in_
            
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
                return input_features
            else: 
                return input_features

        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_
        
        else:
            return ['x' + str(i) for i in range(self.n_features_in_)]
            
numerical_train_set = train_set_labeless.loc[:, train_set_labeless.dtypes == float].dropna()
scc = StandardScalerClone()

input_features = [f'f{i}' for i in range(8)]

scc.get_feature_names_out(input_features = input_features)

inverse_transform = pd.DataFrame(scc.inverse_transform(scc.fit_transform(numerical_train_set)), columns = numerical_train_set.columns)

comparison = inverse_transform - numerical_train_set.reset_index(drop = True)
comparison.loc[(comparison > 1e-12).any(axis = 1)] # The differences start to appear around the tolerance value 1e-12.
