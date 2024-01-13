# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:37:28 2022

@author: nima
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import datetime  as dt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy
import seaborn as sns

data_silage_maize = pd.read_csv('C:/Users/nimam/OneDrive/Desktop/ZALF Thesis/final DATA/silage maize/data_silage_maize.csv', sep=',')
data_silage_maize['yield'] = data_silage_maize['yield'] *0.1
data_silage_maize.columns
data_silage_maize.drop(columns='Unnamed: 0', axis=1, inplace=True)
data_silage_maize.dtypes

np.isinf(data_silage_maize['date']).any()
np.isinf(data_silage_maize['NUTS_CODE']).any()
np.isinf(data_silage_maize['crop']).any()
np.isinf(data_silage_maize['year']).any()
np.isinf(data_silage_maize['NDVI']).any()
np.isinf(data_silage_maize['NDWI']).any()
np.isinf(data_silage_maize['ET']).any()
np.isinf(data_silage_maize['dataLST_Day']).any()
np.isinf(data_silage_maize['yield']).any()

###############################################################################

data_silage_maize.isna().sum()
#nan_values = data_silage_maize[data_silage_maize['NDVI'].isna()]
nan_rows = data_silage_maize[data_silage_maize.isnull().any(axis=1)]

data_silage_maize = data_silage_maize.sort_values(['NUTS_CODE', 'year'])
data_silage_maize.reset_index(drop=True, inplace=True)

data_silage_maize['date'] =pd.to_datetime(data_silage_maize['date'])
data_silage_maize['year'] =pd.to_datetime(data_silage_maize['year'])
#data_silage_maize['date'] = data_silage_maize['date'].dt.date
#data_silage_maize['year'] = data_silage_maize['year'].dt.year

###############################################################################

Filtered = data_silage_maize.loc[data_silage_maize['date'].dt.month == 7]
Filtered = Filtered.loc[Filtered['date'].dt.day <= 4]
Filtered.reset_index(drop=True, inplace=True) # or Filtered = Filtered.reindex(range(0, len(Filtered)))

Filtered.isna().sum()
#sorted_df.drop(sorted_df.index[sorted_df['NUTS_CODE'] != 'DE403'], inplace=True) # or rrr = df_july.loc[df_july['crop'] == 'silage maize']
yield_data = Filtered['yield']
county_data = Filtered['NUTS_CODE']
yield_data.shape

###############################################################################

dropped = data_silage_maize.drop(columns=['date', 'NUTS_CODE', 'crop', 'year', 'yield'])
dropped.columns
#x = np.zeros((285,12,8))
data_np = dropped.values
data_3d = data_np.reshape(2603,12,4)
data_3d[[20],[5],[1]]
len(data_3d)


converted_2D = data_3d.reshape(-1, 48)
len(converted_2D)

clmns = ['NDVI_July_1', 'NDWI_July_1', 'ET_July_1', 'LST_July_1', 'NDVI_July_2', 'NDWI_July_2', 'ET_July_2', 'LST_July_2', 'NDVI_July_3', 'NDWI_July_3', 'ET_July_3', 'LST_July_3', 'NDVI_July_4', 'NDWI_July_4', 'ET_July_4', 'LST_July_4',
 'NDVI_Aug_1', 'NDWI_Aug_1', 'ET_Aug_1', 'LST_Aug_1', 'NDVI_Aug_2', 'NDWI_Aug_2', 'ET_Aug_2', 'LST_Aug_2', 'NDVI_Aug_3', 'NDWI_Aug_3', 'ET_Aug_3', 'LST_Aug_3', 'NDVI_Aug_4', 'NDWI_Aug_4', 'ET_Aug_4', 'LST_Aug_4',
 'NDVI_Sep_1', 'NDWI_Sep_1', 'ET_Sep_1', 'LST_Sep_1', 'NDVI_Sep_2', 'NDWI_Sep_2', 'ET_Sep_2', 'LST_Sep_2', 'NDVI_Sep_3', 'NDWI_Sep_3', 'ET_Sep_3', 'LST_Sep_3', 'NDVI_Sep_4', 'NDWI_Sep_4', 'ET_Sep_4', 'LST_Sep_4']

hrzntal_data = pd.DataFrame(data=converted_2D, columns=clmns)
len(hrzntal_data)
pd.DataFrame(hrzntal_data).isna().sum()

###############################################################################

# =============================================================================
# hrzntal_data['NUTS_CODE'] = county_data
# hrzntal_data['yield'] = yield_data
# 
# nan_rows = hrzntal_data[hrzntal_data.isnull().any(axis=1)]
# nan_rows_valueCounts = pd.DataFrame(nan_rows['NUTS_CODE'].value_counts())
# counties_nonsList = nan_rows_valueCounts.index[nan_rows_valueCounts['NUTS_CODE']>=10].tolist()
# for i in counties_nonsList:
#     hrzntal_data.drop(hrzntal_data[hrzntal_data['NUTS_CODE'] == i].index, inplace = True)
# 
# hrzntal_data.dropna(inplace=True)
# hrzntal_data.reset_index(drop=True, inplace=True)
# len(pd.unique(hrzntal_data['NUTS_CODE']))
# 
# ### to label each county from 0 to n
# from sklearn.preprocessing import LabelEncoder
# lbl = LabelEncoder()
# lbl.fit(hrzntal_data['NUTS_CODE'])
# hrzntal_data['NUTS_CODE'] = lbl.transform(hrzntal_data['NUTS_CODE'])
# hrzntal_data
# 
# X = hrzntal_data.iloc[:, :-1]
# #X = X_1.join(pd.get_dummies(X_1['NUTS_CODE']))
# #X.drop('NUTS_CODE', axis=1, inplace=True)
# y = hrzntal_data.iloc[:, -1]
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
# new_hrzntal_data = X
# new_hrzntal_data['yield'] = y
# =============================================================================

###############################################################################

hrzntal_data['NUTS_CODE'] = county_data
#hrzntal_data = hrzntal_data[hrzntal_data.NUTS_CODE != 'DEA26']

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
lbl.fit(hrzntal_data['NUTS_CODE'])
hrzntal_data['NUTS_CODE'] = lbl.transform(hrzntal_data['NUTS_CODE'])
hrzntal_data

#hrzntal_data = hrzntal_data.join(pd.get_dummies(hrzntal_data['NUTS_CODE']))
#hrzntal_data.drop('NUTS_CODE', axis=1, inplace=True)

hrzntal_data['yield'] = yield_data
hrzntal_data['year'] = Filtered['year']

nan_rows = hrzntal_data[hrzntal_data.isnull().any(axis=1)]
nan_rows_valueCounts = pd.DataFrame(nan_rows['NUTS_CODE'].value_counts())
counties_nonsList = nan_rows_valueCounts.index[nan_rows_valueCounts['NUTS_CODE']>=10].tolist()
for i in counties_nonsList:
    hrzntal_data.drop(hrzntal_data[hrzntal_data['NUTS_CODE'] == i].index, inplace = True)
hrzntal_data.dropna(inplace=True)
hrzntal_data.reset_index(drop=True, inplace=True)

counter = 2015
train = hrzntal_data.loc[hrzntal_data['year'].dt.year != counter]
test = hrzntal_data.loc[hrzntal_data['year'].dt.year == counter]
X_train, y_train, X_test, y_test = train.iloc[:, :-3], train.iloc[:, -2], test.iloc[:, :-3], test.iloc[:, -2]

# this two line are just for preparing the X and y for Decision Tree to obtain the optimal value for alpha
X = hrzntal_data.iloc[:, :-3]
y = hrzntal_data.iloc[:, -2]


###########################################################################################################################################
###########################################################################################################################################

### default hyperparameters

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 66)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

### implement the Random Forest Regression
# in this part I run the RF regression model 4 times and compare the results:
    # 1. without any parameter,

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rf_1 = RandomForestRegressor(random_state=66)
rf_1.fit(X_train, y_train)

y_pred_1 = rf_1.predict(X_test)
y_pred_1
y_pred_1.shape

### Evaluate the model

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    average_error = np.mean(errors)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy, average_error, mape, errors

Default_model = evaluate(rf_1, X_test, y_test)

Default_model_mse = mean_squared_error(y_test, y_pred_1) # or np.mean((y_pred_1 - y_test) ** 2)
Default_model_rmse = mean_squared_error(y_test, y_pred_1, squared=False) # RMSE
Default_model_r2 = r2_score(y_test, y_pred_1) # or rf_1.score(X_test, y_test)
Default_model_mae = np.mean(np.absolute(y_pred_1 - y_test)) #'Mean Absolute Error: %.3f' % np.mean(np.absolute(y_pred_1 - y_test))

Default_model_n_trees = rf_1.get_params()['n_estimators']
Default_model_n_features = rf_1.get_params()['max_features']

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

### Random Hyperparameter Grid: Random search allowed us to narrow down the range for each hyperparameter

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt', None, 0.5, 0.25, 0.75, 40]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

###Random Search Training: Use the random grid to search for best hyperparameters

# First create the base model to tune
rf2 = RandomForestRegressor()
# Random search of parameters, using 10 fold cross validation, 
# search across 500 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf2, param_distributions = random_grid, scoring = 'neg_root_mean_squared_error', 
                               n_iter = 200, cv = 5, random_state=66, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


rf_random.best_params_

### Evaluate Random Search model

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)


y_pred_2 = best_random.predict(X_test)
y_pred_2
y_pred_2.shape

randomSearch_mse = mean_squared_error(y_test, y_pred_2) # or np.mean((y_pred_2 - y_test) ** 2)
randomSearch_rmse = mean_squared_error(y_test, y_pred_2, squared=False) # RMSE
randomSearch_r2 = r2_score(y_test, y_pred_2) # or rf_1.score(X_test, y_test)
randomSearch_mae = np.mean(np.absolute(y_pred_2 - y_test)) #'Mean Absolute Error: %.3f' % np.mean(np.absolute(y_pred_2 - y_test))

randomSearch_n_trees = rf_random.best_params_['n_estimators']
randomSearch_n_features = rf_random.best_params_['max_features']

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

### Grid Search with Cross Validation: evaluates all combinations we define

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [None, 90, 100, 110],
    'max_features': ['sqrt', 0.25, 0.3],
    'min_samples_leaf': [1],
    'min_samples_split': [2, 4, 8, 10],
    'n_estimators': [1500, 1800, 1900, 2000, 2100]
}

# Create a based model
rf3 = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf3, param_grid = param_grid, 
                           scoring = 'neg_root_mean_squared_error', cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

grid_search.best_params_

### evaluate grid search model
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)


y_pred_3 = best_grid.predict(X_test)
y_pred_3
y_pred_3.shape

gridSearch_mse = mean_squared_error(y_test, y_pred_3) # or np.mean((y_pred_1 - y_test) ** 2)
gridSearch_rmse = mean_squared_error(y_test, y_pred_3, squared=False) # RMSE
gridSearch_r2 = r2_score(y_test, y_pred_3) # or rf_1.score(X_test, y_test)
gridSearch_mae = np.mean(np.absolute(y_pred_3 - y_test)) #'Mean Absolute Error: %.3f' % np.mean(np.absolute(y_pred_3 - y_test))

gridSearch_n_trees = grid_search.best_params_['n_estimators']
gridSearch_n_features = grid_search.best_params_['max_features']

###########################################################################################################################################
###########################################################################################################################################

###########################################################################################################################################
###########################################################################################################################################

### Comparison of all techniques

# Comparison
comparison = {'Model': ['Default Model', 'Random Search Model', 'Grid Search Model'],
              'Accuracy': [round(Default_model[0], 3), round(random_accuracy[0], 3), round(grid_accuracy[0], 3)],
              'Average Eerror': [round(Default_model[1], 3), round(random_accuracy[1], 3), round(grid_accuracy[1], 3)],
              'r squared': [round(Default_model_r2, 3), round(randomSearch_r2, 3), round(gridSearch_r2, 3)],
              'RMSE': [round(Default_model_rmse, 3), round(randomSearch_rmse, 3), round(gridSearch_rmse, 3)],
              'n_trees': [Default_model_n_trees, randomSearch_n_trees, gridSearch_n_trees],
              'n_features': [Default_model_n_features, randomSearch_n_features, gridSearch_n_features]}
                        
comparison = pd.DataFrame.from_dict(comparison, orient = 'columns')    
    
### Model Comparison Plot

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.style.use('fivethirtyeight')
xvalues = list(range(len(comparison['Model'])))

plt.subplots(2, 2, figsize=(10, 8))

# Helper function to annotate bars
def annotate_bars(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05,
                 round(yval, 2), ha='center', va='bottom', fontsize=10)

# First subplot
plt.subplot(221)
bars1 = plt.bar(xvalues, comparison['Accuracy'], color = 'g', edgecolor = 'k', linewidth = 1.8)
annotate_bars(bars1)
plt.xticks(xvalues, '', rotation = 45, fontsize = 12)
plt.ylim(ymin = 60.0, ymax = 100.0)
plt.xlabel(''); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Comparison');

# Second subplot
plt.subplot(222)
bars2 = plt.bar(xvalues, comparison['Average Eerror'], color = 'r', edgecolor = 'k', linewidth = 1.8)
annotate_bars(bars2)
plt.xticks(xvalues, '', rotation = 45)
plt.ylim(ymin = 0.0, ymax = 10.0)
plt.xlabel(''); plt.ylabel('Error (ton)'); plt.title('Error Comparison');

# Third subplot
plt.subplot(223)
bars3 = plt.bar(xvalues, comparison['r squared'], color = 'b', edgecolor = 'k', linewidth = 1.8)
annotate_bars(bars3)
plt.xticks(xvalues, comparison['Model'], rotation = 45, fontsize = 12)
plt.ylim(ymin = 0.0, ymax = 1.0)
plt.xlabel('Model'); plt.ylabel('R squared'); plt.title('R squared Comparison');

# Fourth subplot
plt.subplot(224)
bars4 = plt.bar(xvalues, comparison['RMSE'], color = 'y', edgecolor = 'k', linewidth = 1.8)
annotate_bars(bars4)
plt.xticks(xvalues, comparison['Model'], rotation = 45, fontsize = 12)
plt.ylim(ymin = 0.0, ymax = 10.0)
plt.xlabel('Model'); plt.ylabel('RMSE (ton)'); plt.title('RMSE Comparison');

plt.subplots_adjust(wspace=0.30, hspace=0.30)
plt.savefig(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\rr.png', bbox_inches = 'tight', dpi=600)
plt.show()

##############################################################################

aws = pd.DataFrame({'Default Model':Default_model[3], 'Random Search Model':random_accuracy[3], 
                    'Grid Search Model':grid_accuracy[3]})
aq = pd.DataFrame(grid_accuracy[3]).reset_index(drop=True)
aq.rename(columns={"yield": "error"}, inplace=True)
sns.set_context("talk", font_scale=1.1)
plt.figure(figsize=(4,8))
plt.title('2015', fontsize=16)

sns.violinplot(y=aq['error'], inner=None)
sns.stripplot(y=aq['error'], color="black", edgecolor="gray")
plt.ylabel('Error (ton)', fontsize=14)
#sns.boxplot(y=grid_accuracy[3])
plt.savefig(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\rrfr.png',  bbox_inches = 'tight', dpi=600)

######################################################################################
######################################################################################

result_2019 = pd.DataFrame({'Prediction': y_pred_3, 'Actual': y_test.values}, index= lbl.inverse_transform(test['NUTS_CODE']))
result_2019.to_csv(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\out2019')
result_2018 = pd.DataFrame({'Prediction': y_pred_1, 'Actual': y_test.values}, index= lbl.inverse_transform(test['NUTS_CODE']))
result_2018.to_csv(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\out2018')
result_2017 = pd.DataFrame({'Prediction': y_pred_1, 'Actual': y_test.values}, index= lbl.inverse_transform(test['NUTS_CODE']))
result_2017.to_csv(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\out2017')
result_2016 = pd.DataFrame({'Prediction': y_pred_1, 'Actual': y_test.values}, index= lbl.inverse_transform(test['NUTS_CODE']))
result_2016.to_csv(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\out2016')
result_2015 = pd.DataFrame({'Prediction': y_pred_1, 'Actual': y_test.values}, index= lbl.inverse_transform(test['NUTS_CODE']))
result_2015.to_csv(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\out2015')

fig, ax = plt.subplots(5, 2, figsize=(20,18), gridspec_kw={'width_ratios': [2.5, 1.5]})
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

ax[0, 0].plot(result_2019['Prediction'],  color = 'yellowgreen', label='Prediction')
ax[0, 0].plot(result_2019['Actual'], color = 'tomato', label='Actual', alpha=0.6)#  linestyle='--'
ax[0, 0].set_title('a) Actual Yield vs Prediction values (2019)', fontsize=10)
ax[0, 0].legend(bbox_to_anchor=(0, 1.5), loc='upper left', borderaxespad=0)
ax[0, 0].set_xticklabels(result_2019.index, rotation=45, ha='right', rotation_mode='anchor')
ax[0, 0].tick_params(axis='x', which='major', labelsize=4)

sns.regplot(ax=ax[0, 1], x= 'Prediction', y= 'Actual', data=result_2019, scatter_kws={"color": "green"}, 
            line_kws={"color": "blue"})
ax[0, 1].set_xlabel('')

ax[1, 0].plot(result_2018['Prediction'],  color = 'yellowgreen', label='Prediction')
ax[1, 0].plot(result_2018['Actual'], color = 'tomato', label='Actual', alpha=0.6)#  linestyle='--'
ax[1, 0].set_title('a) Actual Yield vs Prediction values (2018)', fontsize=10)
#ax[1, 0].legend(bbox_to_anchor=(0, 1.5), loc='upper left', borderaxespad=0)
ax[1, 0].set_xticklabels(result_2018.index, rotation=45, ha='right', rotation_mode='anchor')
ax[1, 0].tick_params(axis='x', which='major', labelsize=4)

sns.regplot(ax=ax[1, 1], x= 'Prediction', y= 'Actual', data=result_2018, scatter_kws={"color": "green"}, 
            line_kws={"color": "blue"})
ax[1, 1].set_xlabel('')

ax[2, 0].plot(result_2017['Prediction'],  color = 'yellowgreen', label='Prediction')
ax[2, 0].plot(result_2017['Actual'], color = 'tomato', label='Actual', alpha=0.6)#  linestyle='--'
ax[2, 0].set_title('a) Actual Yield vs Prediction values (2017)', fontsize=10)
#ax[2, 0].legend(bbox_to_anchor=(0, 1.5), loc='upper left', borderaxespad=0)
ax[2, 0].set_xticklabels(result_2017.index, rotation=45, ha='right', rotation_mode='anchor')
ax[2, 0].tick_params(axis='x', which='major', labelsize=4)

sns.regplot(ax=ax[2, 1], x= 'Prediction', y= 'Actual', data=result_2017, scatter_kws={"color": "green"}, 
            line_kws={"color": "blue"})
ax[2, 1].set_xlabel('')

ax[3, 0].plot(result_2016['Prediction'],  color = 'yellowgreen', label='Prediction')
ax[3, 0].plot(result_2016['Actual'], color = 'tomato', label='Actual', alpha=0.6)#  linestyle='--'
ax[3, 0].set_title('a) Actual Yield vs Prediction values (2016)', fontsize=10)
#ax[3, 0].legend(bbox_to_anchor=(0, 1.5), loc='upper left', borderaxespad=0)
ax[3, 0].set_xticklabels(result_2016.index, rotation=45, ha='right', rotation_mode='anchor')
ax[3, 0].tick_params(axis='x', which='major', labelsize=4)

sns.regplot(ax=ax[3, 1], x= 'Prediction', y= 'Actual', data=result_2016, scatter_kws={"color": "green"}, 
            line_kws={"color": "blue"})
ax[3, 1].set_xlabel('')

ax[4, 0].plot(result_2015['Prediction'],  color = 'yellowgreen', label='Prediction')
ax[4, 0].plot(result_2015['Actual'], color = 'tomato', label='Actual', alpha=0.6)#  linestyle='--'
ax[4, 0].set_title('a) Actual Yield vs Prediction values (2015)', fontsize=10)
#ax[3, 0].legend(bbox_to_anchor=(0, 1.5), loc='upper left', borderaxespad=0)
ax[4, 0].set_xticklabels(result_2015.index, rotation=45, ha='right', rotation_mode='anchor')
ax[4, 0].tick_params(axis='x', which='major', labelsize=4)
ax[4, 0].set_xlabel('Counties Codes')

sns.regplot(ax=ax[4, 1], x= 'Prediction', y= 'Actual', data=result_2015, scatter_kws={"color": "green"}, 
            line_kws={"color": "blue"})

fig.text(0.09, 0.5, 'Average Yield (metric ton/hactare)', va='center', ha='center', rotation='vertical')
for ax in ax.flat:
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

fig.savefig('D:/rr.jpg', dpi=1200)

###############################################################################################
###############################################################################################

from sklearn.inspection import permutation_importance
import shap

perm_importance = permutation_importance(best_grid, X_test, y_test, n_repeats=30, random_state=66,  scoring='r2')

sorted_idx = perm_importance.importances_mean.argsort()

plt.barh(train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")


explainer = shap.TreeExplainer(best_grid)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values, X)

################################################################################################
################################################################################################

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
features = [('NDWI_Sep_4', 'LST_July_3')]
plt.figure(figsize=(10, 8))
PartialDependenceDisplay.from_estimator(best_grid, X_train, features)
plt.xlabel('NDWI_Sep_4', fontsize=14)    # Adjusting x-axis label font size
plt.ylabel('LST_July_3', fontsize=14)    # Adjusting y-axis label font size
plt.title('PDP 2015', fontsize=16)            # Adjusting title font size
plt.xticks(fontsize=12)                    # Adjusting x-axis ticks font size
plt.yticks(fontsize=12)   
# After all your plotting commands and just before saving:
fig = plt.gcf()           # Get the current figure
fig.set_size_inches(10, 8) # Set the desired size
plt.savefig(r'C:\Users\nimam\OneDrive\Desktop\New folder (3)\partial.png', dpi=600)
plt.show()
