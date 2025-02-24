"""Schema for hyperparameters inputs of different models."""
# SEE THE 'Vim Page Down Command' PAGE FOR DETAILS ON HOW TO IMPLEMENT INTERNAL VALIDATION

# OBVIOUSLY LOW < HIGH, could do low=high to have a fixed value (should test thhis though first)

HYPERPARAMETER_SCHEMAS = {
    "XGBClassifier": {
        "reg_lambda": {
            "type": float,
            "global_min": 1e-4,
            "global_max": 100,
            "log_allowed": True,
        },
        "reg_alpha": {
            "type": float,
            "global_min": 1e-4,
            "global_max": 100,
            "log_allowed": True,
        },
        "learning_rate": {
            "type": float,
            "global_min": 0.01,
            "global_max": 1.0,
            "log_allowed": True,
        },
        "max_depth": {
            "type": int,
            "global_min": 1,
            "global_max": 5,
            "log_allowed": False,
        },
    },
    "LGBMClassifier": {
        "max_depth": {
            "type": int,
            "global_min": 1,
            "global_max": 15,
            "log_allowed": False,
        },
        "min_child_samples": {
            "type": int,
            "global_min": 1,
            "global_max": 30,
            "log_allowed": False,
        },
        "learning_rate": {
            "type": float,
            "global_min": 0.01,
            "global_max": 0.6,
            "log_allowed": True,
        },
        "num_leaves": {
            "type": int,
            "global_min": 2,
            "global_max": 100,
            "log_allowed": False,
        },
        "n_estimators": {
            "type": int,
            "global_min": 100,
            "global_max": 3000,
            "log_allowed": False,
        },
        "min_split_gain": {
            "type": float,
            "global_min": 1e-6,
            "global_max": 10,
            "log_allowed": True,
        },
        "min_child_weight": {
            "type": float,
            "global_min": 1e-6,
            "global_max": 1e-1,
            "log_allowed": True,
        },
        "reg_alpha": {
            "type": float,
            "global_min": 1e-5,
            "global_max": 10,
            "log_allowed": True,
        },
        "reg_lambda": {
            "type": float,
            "global_min": 1e-5,
            "global_max": 10,
            "log_allowed": True,
        },
    },
    "SVC": {
        "C": {
            "type": float,
            "global_min": 1e-3,
            "global_max": 10000,
            "log_allowed": True,
        },
    },
    "RandomForestClassifier": {
        "n_estimators": {
            "type": int,
            "global_min": 100,
            "global_max": 1000,
            "log_allowed": False,
        },
    },
}
# This is how we should store conidtional params once we include them
# 'xgb_classifier': {
#         'learning_rate': {
#             'type': float,
#             'global_min': 0.01,
#             'global_max': 1.0,
#             'log_allowed': True,
#         },
#         'max_depth': {
#             'type': int,
#             'global_min': 1,
#             'global_max': 15,
#             'log_allowed': False,
#         },
#         'booster': {
#             'type': str,
#             'allowed_values': ['gbtree', 'gblinear', 'dart'],
#             'conditional_params': {
#                 'dart': ['rate_drop', 'skip_drop'],
#                 # Add other conditional parameters for 'dart'
#             },
#         },
#         'rate_drop': {
#             'type': float,
#             'global_min': 0.0,
#             'global_max': 1.0,
#             'log_allowed': False,
#             'depends_on': {
#                 'booster': 'dart',
#             },
#         },
#         'skip_drop': {
#             'type': float,
#             'global_min': 0.0,
#             'global_max': 1.0,
#             'log_allowed': False,
#             'depends_on': {
#                 'booster': 'dart',
#             },
#         },
#         # Add other hyperparameters...
#     },
#     # Define schemas for other models...
# }
