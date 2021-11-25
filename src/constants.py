from scipy import stats
import json
import src.utils.utils as u

conf = u.load_config_file('../configs/config.json')

random_state = 42

# Data type
most_common_class = 1

# Model parameters from the config file
unknown_surrogate_model_parameters = conf["unknown_model_surrogate"]
classification_model_parameters = conf["classification_model_surrogate"]

# Model parameter grid for randomized search
xgb_param_dist = {'estimator__n_estimators': stats.randint(150, 1000),
              'estimator__learning_rate': stats.uniform(0.01, 0.6),
              'estimator__subsample': stats.uniform(0.3, 0.9),
              'estimator__max_depth': [3, 4, 5, 6, 7, 8, 9],
              'estimator__colsample_bytree': stats.uniform(0.5, 0.9),
              'estimator__min_child_weight': [1, 2, 3, 4]
             }
