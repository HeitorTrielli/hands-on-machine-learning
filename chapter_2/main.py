import time
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import randint
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, train_test_split, cross_val_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import numpy as np
from chapter_2.data_handling import load_housing_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.impute import SimpleImputer
import joblib

# This script is mostly a copy of what the book throws at the reader, but with some minor tweaks and 
# tests on my part

# Setting transform output of sklearn transform to be a Pandas DataFrame
'''
from sklearn import set_config
set_config(transform_output = "pandas")
'''

housing = load_housing_data()

'''
housing.hist(bins = 50, figsize = (12,8))
plt.show()
'''

# Creating a categorical column for the median income
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
    labels = [1, 2, 3, 4, 5],
)

# Splitting the data, stratifying in the categorical median income column
train_set, test_set = (
    train_test_split(
        housing, 
        test_size = 0.2, 
        random_state = 42, 
        stratify = housing.income_cat
    )
)

# Removing the income_cat column, since we already have stratified sets
train_set, test_set = (x.drop('income_cat', axis = 1) for x in (train_set, test_set))

# Histogram of the median income of each district
'''
housing.median_income.hist(bins = 50)
plt.show()
'''

# Barplot of the median income housing categories
'''
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0)
plt.xlabel("Income category")r
plt.ylabel("Number of districts")
plt.show()R
'''

# If we want to have multiple stratified indices
'''
splitter = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 27)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
'''

# Plotting the map, where each circle is a district, their size is 
# the population and the color is the median housing price, the redder, the more expensive
'''
housing.plot(
    kind = 'scatter', 
    x = 'longitude', 
    y = 'latitude', 
    alpha = 0.2,
    s = housing.population/100,
    c = housing.median_house_value,
    label = 'population',
    cmap = 'jet',
    colorbar = True,
    legend = True,
    figsize = (10, 7),
    sharex = False,
    grid = True
)
plt.show()
'''

# Plotting the scatterplot of multiple atributes of interest
'''
atributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[atributes])
plt.show()
'''

# Some data are kinda useless alone, so we create more useful variables
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# Correlation matrix
corr_matrix = housing.drop('ocean_proximity', axis = 1).corr()

# Returning to the train set
housing = train_set.copy()

# Separating the model labels from the rest of the data
housing_labels = housing.median_house_value
housing = housing.drop('median_house_value', axis = 1)

# Creating an inputer ()
numerical_housing = housing.select_dtypes(include = np.number)
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(numerical_housing)


# Imputing with interative imputer (regression to predict the missing value)
'''
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
iterative = IterativeImputer()
X = iterative.fit_transform(numerical_housing)
count_nulls = pd.DataFrame(X, columns = numerical_housing.columns).total_bedrooms.isna().sum()
print(count_nulls)
methods = [method for method in dir(IterativeImputer) if callable(getattr(IterativeImputer, method))] # to get the class methods
'''

# Transforming the categorical data in numbers
ordinal_encoder = OrdinalEncoder()
encoded_housing_cat = ordinal_encoder.fit_transform(housing[['ocean_proximity']])
onehot_encoder = OneHotEncoder()
onehot_housing_cat = onehot_encoder.fit_transform(housing[['ocean_proximity']])

# Scaling the features
std_scaler = StandardScaler()
housing_std_scaled = std_scaler.fit_transform(numerical_housing)

# Creating a similarity measure for the house being 35 years old
# This will be useful if, for instance, house made aroud 35 years ago had
# a particular style that are more or less appreciated today that may 
# change the price
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)


# Taking the regression scaling the target
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
X = imputer.fit_transform(housing_std_scaled)
model = LinearRegression()
model.fit(X, scaled_labels)
some_new_data = housing_std_scaled[:5] # pretend this is new data
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# Making a log transformation of the population
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])


# Custom transformer that takes the ratio of two features:
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])

#############################################################################
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]

cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    value = X[:, [0]] / X[:, [1]]
    return value

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"] # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out = ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=27)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder = default_num_pipeline
) # one column remaining: housing_median_age


lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)

housing_predictions[:5].round(-2)
housing_labels.iloc[:5].values

lin_rmse = root_mean_squared_error(housing_predictions, housing_labels)

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state = 42))
tree_reg.fit(housing, housing_labels)
tree_predictions = tree_reg.predict(housing)

tree_rmse = root_mean_squared_error(tree_predictions, housing_labels)

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
lin_rmses = -cross_val_score(lin_reg, housing, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)

pd.Series(tree_rmses).describe()
pd.Series(lin_rmses).describe()

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state = 42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)

pd.Series(forest_rmses).describe()


# Grid Searching for the Random Forest model
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
     {'preprocessing__geo__n_clusters': [10, 15],
      'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(
    full_pipeline, param_grid, cv = 3, scoring = 'neg_root_mean_squared_error'
)

# Takes a little while:
t0 = time.time()
grid_search.fit(housing, housing_labels)
t1 = time.time()
(t1 - t0)/60

grid_search.best_params_
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.sort_values('mean_test_score')


param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42
)

rnd_search.fit(housing, housing_labels)
 

final_model = rnd_search.best_estimator_ # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)


sorted(
    zip(feature_importances,final_model["preprocessing"].get_feature_names_out()),
        reverse=True
)


X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)


confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(
    stats.t.interval(confidence, len(squared_errors) - 1,
                     loc=squared_errors.mean(),
                     scale=stats.sem(squared_errors))
    )

# Saving my model
joblib.dump(final_model, "my_california_housing_model.pkl")

# Loadiing my model
final_model_reloaded = joblib.load("my_california_housing_model.pkl")