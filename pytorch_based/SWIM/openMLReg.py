from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import time
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor
import pandas as pd
import openml
import optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

from models import GradientRandFeatBoostRegression, RidgeCVModule, RidgeModule


#########################  |
##### Dataset Code  #####  |
#########################  V


def get_openml_reg_ids() -> List[int]:
    """Collects the IDs of the regression datasets from OpenML's study 353.

    Returns:
        _type_: _description_
    """
    # Fetch the collection with ID 353
    collection = openml.study.get_suite(353)
    dataset_ids = collection.data
    metadata_list = []

    # Fetch and process each dataset
    for i, dataset_id in enumerate(dataset_ids):
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        
        # Determine if the dataset has categorical features. Collect metadata
        has_categorical = any(categorical_indicator)
        metadata = {
            'dataset_id': dataset.id,
            'name': dataset.name,
            'n_obs': int(dataset.qualities['NumberOfInstances']),
            'n_features': int(dataset.qualities['NumberOfFeatures']),
            '%_unique_y': len(np.unique(y))/len(y),
            'n_unique_y': len(np.unique(y)),
            'has_categorical': has_categorical
        }
        
        metadata_list.append(metadata)
        print(f" {i+1}/{len(dataset_ids)} Processed dataset {dataset.id}: {dataset.name}")

    # Create a DataFrame from the metadata list
    df_metadata = pd.DataFrame(metadata_list).sort_values('%_unique_y', ascending=False).set_index("dataset_id").sort_index()
    df_metadata.loc[44962, "has_categorical"] = True
    return list(df_metadata.index)



def np_load_openml_dataset(dataset_id: int, 
                        normalize_X: bool = True,
                        normalize_y: bool = True,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    # Fetch dataset from OpenML by its ID
    dataset = openml.datasets.get_dataset(dataset_id)
    df, _, categorical_indicator, attribute_names = dataset.get_data()
    y = np.array(df.pop(dataset.default_target_attribute)).astype(np.float32)
    X = np.array(df).astype(np.float32)

    #normalize
    if normalize_X:
        X = X - X.mean(axis=0, keepdims=True)
        X = X / (X.std(axis=0, keepdims=True) + 1e-5)
        X = np.clip(X, -3, 3)
    if normalize_y:
        y = y - y.mean()
        y = y / (y.std() + 1e-5)
        y = np.clip(y, -3, 3)

    return X, y



def pytorch_load_openml_dataset(
        dataset_id: int, 
        normalize_X: bool = True,
        normalize_y: bool = True,
        device: str = "cpu",
        ) -> Tuple[Tensor, Tensor]:
    X, y = np_load_openml_dataset(dataset_id, normalize_X, normalize_y)
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)

    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    return X, y


#####################################################
######### XGBoost Baseline ##########################
#####################################################


def objective_xgboost_cv_reg(
        trial, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        k_folds: int,
        cv_seed: int,
        ):
    params = {
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    rmse_list = []

    for inner_train_idx, inner_valid_idx in inner_cv.split(X_train):
        X_inner_train, X_inner_valid = X_train[inner_train_idx], X_train[inner_valid_idx]
        y_inner_train, y_inner_valid = y_train[inner_train_idx], y_train[inner_valid_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_inner_train, y_inner_train)

        preds = model.predict(X_inner_valid)
        rmse = root_mean_squared_error(y_inner_valid, preds)
        rmse_list.append(rmse)

    return np.mean(rmse_list)



def evaluate_xgboost_kfoldcv(
        X: np.ndarray, 
        y: np.ndarray, 
        k_folds: int = 5, 
        cv_seed: int = 42,
        n_optuna_trials: int = 50,
    ):
    """Evaluates an XGBoost model using k-fold cross-validation.
    Hyperparameters are tuned with Optuna using an inner k-fold CV.
    The model is then trained on the whole fold train set and evaluated 
    on the fold test set.

    Returns the train RMSE, test RMSE, chosen params, training times, and test set inference times
    for each fold.
    """
    outer_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    outer_train_rmse_scores = []
    outer_test_rmse_scores = []
    chosen_params = []
    fit_times = []
    transform_times = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #hyperparameter tuning with Optuna
        study = optuna.create_study(direction="minimize", )
        objective = lambda trial: objective_xgboost_cv_reg(trial, X_train, y_train, k_folds, cv_seed)
        study.optimize(objective, n_trials=n_optuna_trials)

        #fit model with optimal hyperparams
        t0 = time.perf_counter()
        model = xgb.XGBRegressor(**study.best_params)
        model.fit(X_train, y_train)

        #predict and evaluate
        t1 = time.perf_counter()
        preds_train = model.predict(X_train)
        rmse_train = root_mean_squared_error(y_train, preds_train)
        preds_test = model.predict(X_test)
        rmse_test = root_mean_squared_error(y_test, preds_test)
        t2 = time.perf_counter()

        outer_train_rmse_scores.append(rmse_train)
        outer_test_rmse_scores.append(rmse_test)
        chosen_params.append(study.best_params.copy())
        fit_times.append(t1-t0)
        transform_times.append(t2-t1)


    return (np.array(outer_train_rmse_scores),
            np.array(outer_test_rmse_scores),
            chosen_params,
            np.array(fit_times),
            np.array(transform_times))



def run_all_openMLreg_xgboost(
        dataset_ids: List,
        name_save: str = "XGBoost_OpenML_reg.pkl",
        k_folds: int = 5, 
        cv_seed: int = 42,
        n_optuna_trials: int = 100,
        ):
    # Fetch and process each dataset
    experiments = {}
    for i, dataset_id in enumerate(dataset_ids):
        X, y = np_load_openml_dataset(dataset_id)
        results = evaluate_xgboost_kfoldcv(X, y, k_folds, cv_seed, n_optuna_trials)
        experiments[dataset_id] = results
        print(f" {i+1}/{len(dataset_ids)} Processed dataset {dataset_id}")

    # Save results
    attributes = ["RMSE_train", "RMSE_test", "hyperparams", "t_fit", "t_inference"]
    data_list = []
    for dataset_name, results in experiments.items():
        dataset_data = {}
        print(dataset_name)
        print(results)
        for i, attrib in enumerate(attributes):
            dataset_data[(attrib, "XGBoost")] = [results[i]]
        data_list.append(pd.DataFrame(dataset_data, index=[dataset_name]))

    # Combine all datasets into a single DataFrame
    df = pd.concat(data_list)
    df = df.sort_index(axis=1)
    print(df)
    df.to_pickle(name_save)


###################################################################  |
#####  Boilerplate code for tabular PyTorch model evaluation  #####  |
#####  with Optuna hyperparameter tuning inner kfoldcv        #####  |
###################################################################  V


def get_pytorch_optuna_cv_rmse_objective(
        trial,
        ModelClass: Callable,
        get_optuna_params: Callable,
        X_train: Tensor, 
        y_train: Tensor, 
        k_folds: int,
        cv_seed: int,
        ):
    """The objective to be minimized in Optuna's 'study.optimize(objective, n_trials)' function."""
    
    params = get_optuna_params(trial)

    inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    rmse_list = []
    for inner_train_idx, inner_valid_idx in inner_cv.split(X_train):
        X_inner_train, X_inner_valid = X_train[inner_train_idx], X_train[inner_valid_idx]
        y_inner_train, y_inner_valid = y_train[inner_train_idx], y_train[inner_valid_idx]

        model = ModelClass(**params)
        model.fit(X_inner_train, y_inner_train)

        preds = model(X_inner_valid)
        rmse = torch.sqrt(nn.functional.mse_loss(y_inner_valid, preds))
        rmse_list.append(rmse.item())

    return np.mean(rmse_list)



def evaluate_pytorch_model_kfoldcv(
        ModelClass : Callable,
        get_optuna_params : Callable,
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    """
    Evaluates a PyTorch model using k-fold cross-validation,
    with an inner Optuna hyperparameter tuning loop for each fold.
    The model is then trained on the whole fold train set and evaluated
    on the fold test set.

    Inner and outer kFoldCV use the same number of folds.
    """
    torch.manual_seed(cv_seed)
    np.random.seed(cv_seed)
    outer_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    outer_train_rmse_scores = []
    outer_test_rmse_scores = []
    chosen_params = []
    fit_times = []
    transform_times = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #hyperparameter tuning with Optuna
        study = optuna.create_study(direction="minimize", )
        objective = lambda trial: get_pytorch_optuna_cv_rmse_objective(
            trial, ModelClass, get_optuna_params, X_train, y_train, k_folds, cv_seed
            )
        study.optimize(objective, n_trials=n_optuna_trials)

        #fit model with optimal hyperparams
        t0 = time.perf_counter()
        model = ModelClass(**study.best_params).to(device)
        model.fit(X_train, y_train)

        #predict and evaluate
        t1 = time.perf_counter()
        preds_train = model(X_train)
        rmse_train = torch.sqrt(nn.functional.mse_loss(y_train, preds_train))
        preds_test = model(X_test)
        rmse_test = torch.sqrt(nn.functional.mse_loss(y_test, preds_test))
        t2 = time.perf_counter()

        outer_train_rmse_scores.append(rmse_train.item())
        outer_test_rmse_scores.append(rmse_test.item())
        chosen_params.append(study.best_params.copy())
        fit_times.append(t1-t0)
        transform_times.append(t2-t1)
    
    return (np.array(outer_train_rmse_scores),
            np.array(outer_test_rmse_scores),
            chosen_params,
            np.array(fit_times),
            np.array(transform_times))



def run_all_openMLreg_with_model(
        dataset_ids: List,
        evaluate_model_func: Callable,
        name_save: str, #"GRFBoost_OpenML_reg.pkl",
        k_folds: int = 5,
        cv_seed: int = 42,
        n_optuna_trials: int = 100,
        device: Literal["cpu", "cuda"] = "cuda",
        ):
    # Fetch and process each dataset
    experiments = {}
    for i, dataset_id in enumerate(dataset_ids):
        X, y = pytorch_load_openml_dataset(dataset_id)
        results = evaluate_model_func(
            X, y, k_folds, cv_seed, n_optuna_trials, device
            )
        experiments[dataset_id] = results
        print(f" {i+1}/{len(dataset_ids)} Processed dataset {dataset_id}")
    
    # Save results
    attributes = ["RMSE_train", "RMSE_test", "hyperparams", "t_fit", "t_inference"]
    data_list = []
    for dataset_name, results in experiments.items():
        dataset_data = {}
        for i, attrib in enumerate(attributes):
            dataset_data[(attrib, "GRFBoost")] = [results[i]]
        data_list.append(pd.DataFrame(dataset_data, index=[dataset_name]))

    # Combine all datasets into a single DataFrame
    df = pd.concat(data_list)
    df = df.sort_index(axis=1)
    df.to_pickle(name_save)

###### usage example ###### TODO change fun to not include ids
# run_all_openMLreg_with_model(
#     dataset_ids_no_categorical[0:2], 
#     evaluate_GRFBoost, 
#     "GRFBoost_OpenML_reg.pkl",
#     )
    

##############################################################  |
##### Create "evalute_MODELHERE" function for each model #####  |
##############################################################  V


def evaluate_GRFBoost(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = GradientRandFeatBoostRegression
    get_optuna_params = lambda trial : {
        "seed": trial.suggest_int("seed", 42, 42),                              # Fixed value
        "hidden_dim": trial.suggest_int("hidden_dim", X.size(1), 128, log=True),
        "bottleneck_dim": trial.suggest_int("bottleneck_dim", 64, 128, log=True),
        "out_dim": trial.suggest_int("out_dim", y.size(1), y.size(1)),          # Fixed value
        "n_layers": trial.suggest_int("n_layers", 1, 50, log=True),
        "l2_reg": trial.suggest_float("l2_reg", 1e-6, 0.1, log=True),
        "boost_lr": trial.suggest_float("boost_lr", 0.1, 1.0, log=True),
        "feature_type": trial.suggest_categorical("feature_type", ["SWIM"]),    # Fixed value
        "upscale": trial.suggest_categorical("upscale", ["dense"]),             # Fixed value
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params,
        X, y, k_folds, cv_seed, n_optuna_trials, device,
        )



def evaluate_RidgeCV(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = RidgeCVModule
    get_optuna_params = lambda trial : {
        "lower_alpha": trial.suggest_float("lower_alpha", 1e-7, 0.1, log=True),
        "upper_alpha": trial.suggest_float("upper_alpha", 1e-7, 0.1, log=True),
        "n_alphas": trial.suggest_int("n_alphas", 10, 50),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params,
        X, y, k_folds, cv_seed, n_optuna_trials, device,
        )



def evaluate_Ridge(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = RidgeModule
    get_optuna_params = lambda trial : {
        "l2_reg": trial.suggest_float("l2_reg", 1e-7, 0.1, log=True),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params,
        X, y, k_folds, cv_seed, n_optuna_trials, device,
        )