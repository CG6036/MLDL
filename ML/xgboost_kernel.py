# Kernel and XGBoost Hands-on
# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Load Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.head()

# Preprocessing
train_X = train_data.drop(['id', 'defects'], axis = 1)
train_y = train_data['defects']
test_X = test_data.drop(['id'], axis = 1)

# Create a parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'alpha': [0, 0.5, 1.0],
    'lambda': [0, 0.5, 1.0],
}
# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Perform hyperparameter tuning using GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring={'accuracy': 'accuracy', 'f1': 'f1'},
                           refit='f1', cv=cv, verbose=1, n_jobs=-1)
# Model Training
grid_search.fit(train_X, train_y)

# Extract the results
results = pd.DataFrame(grid_search.cv_results_)
df_results = results[['param_learning_rate', 'param_max_depth', 'param_subsample',
          'param_n_estimators','mean_test_f1','mean_test_accuracy']]
print(df_results)

# Finding best parameter combinations
# Sorted the results by F1 score and accuracy and chose the corresponding parameters.
df_results.sort_values(by=['mean_test_f1', 'mean_test_accuracy'], ascending=[False, False])

# train the model with the best performed parameters
# (learning_rate = 0.1, max_depth = 3, subsample = 1.0, n_estimators = 100)
xgb_selected = grid_search.best_estimator_
predictions = xgb_selected.predict(test_X)
df_test_predicted = pd.DataFrame(predictions, columns = ['prediction'])
df_test_predicted.to_csv('2023-29914_pred.csv', index = False)