import matplotlib.pyplot as plt
from chapter_2.data_handling import load_housing_data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

housing = load_housing_data()

housing.info()

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
        random_state = 27, 
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
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
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

housing.total_bedrooms.isna().sum()/len(housing)

imputer = SimpleImputer(strategy="median")

