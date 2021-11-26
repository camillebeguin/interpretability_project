from scipy import stats
import json
import src.utils.utils as u

conf = u.load_config_file('../configs/config.json')

random_state = 42

# Data type
most_common_class = 1

# Model parameters from the config file
unknown_surrogate_model_parameters = conf["unknown_model_surrogate"]
classification_model_parameters = conf["classification_model"]

# Model parameter grid for randomized search
xgb_param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }
