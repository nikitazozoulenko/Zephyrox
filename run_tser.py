from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray
import aeon
import pandas as pd
from tqdm import tqdm

from features.sig_trp import SigVanillaTensorizedRandProj, SigRBFTensorizedRandProj
from features.sig import SigTransform, LogSigTransform
from features.base import TimeseriesFeatureTransformer, TabularTimeseriesFeatures, RandomGuesser
from features.sig_neural import RandomizedSignature
from utils import print_name, print_shape

from preprocessing.timeseries_augmentation import normalize_mean_std_traindata, normalize_streams, augment_time, add_basepoint_zero
from aeon.regression.sklearn import RotationForestRegressor
from sklearn.metrics import root_mean_squared_error

jax.config.update('jax_platform_name', 'gpu') # Used to set the platform (cpu, gpu, etc.)
np.set_printoptions(precision=3, threshold=5) # Print options

from aeon.datasets.tser_datasets import tser_soton
from aeon.datasets import load_regression

from sklearn.linear_model import RidgeCV
from aeon.transformations.collection.convolution_based import Rocket, MultiRocketMultivariate, MiniRocketMultivariate

trees_n_jobs = -1

def get_aeon_dataset(
        dataset_name:str, 
        #extract_path = "/rds/general/user/nz423/home/Data/TSER/"
        extract_path = "/home/nikita/hdd/Data/TSER/"
        ):
    """Loads a dataset from the UCR/UEA archive using 
    the aeon library.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        Tuple: 4-tuple of the form (X_train, y_train, X_test, y_test)
    """
    X_train, y_train = load_regression(dataset_name, split="train", extract_path=extract_path)
    X_test, y_test = load_regression(dataset_name, split="test", extract_path=extract_path)

    return X_train.transpose(0,2,1), y_train, X_test.transpose(0,2,1), y_test


def train_linear_classifier(t0, t1, feat_train_X, feat_test_X, train_y, test_y, clf=RidgeCV(alphas=np.logspace(-3, 5, 50))):
    # train classifier      
    clf.fit(feat_train_X, train_y)
    t2 = time.time()

    # predict
    pred = clf.predict(feat_test_X)
    test_rmse = root_mean_squared_error(test_y, pred)
    train_rmse = root_mean_squared_error(train_y, clf.predict(feat_train_X))
    alpha = 0 if not hasattr(clf, 'alpha_') else clf.alpha_
    t3 = time.time()
    return train_rmse, test_rmse, t1-t0, t2-t1, alpha


def train_and_test_sigbased(
        train_X, train_y, test_X, test_y,
        transformer:TimeseriesFeatureTransformer,
        apply_augmentation:bool=True,
        normalize_features:bool=True,
    ):
    # augment data
    train_X = lax.stop_gradient(jnp.array(train_X))
    test_X  = lax.stop_gradient(jnp.array(test_X))
    if apply_augmentation:
        train_X = add_basepoint_zero(train_X)
        train_X = augment_time(train_X)
        test_X  = add_basepoint_zero(test_X)
        test_X  = augment_time(test_X)

    # fit transformer
    t0 = time.time()
    transformer.fit(train_X)
    feat_train_X = np.array(transformer.transform(train_X))
    feat_test_X = np.array(transformer.transform(test_X))
    if normalize_features:
        feat_train_X, feat_test_X = normalize_mean_std_traindata(feat_train_X, feat_test_X)
    t1 = time.time()

    # feed into linear classifier
    res_ridge = train_linear_classifier(
        t0, t1, feat_train_X, feat_test_X, train_y, test_y)
    res_rotforest = train_linear_classifier(
        t0, t1, feat_train_X, feat_test_X, train_y, test_y, RotationForestRegressor(n_estimators=100, n_jobs=trees_n_jobs))
    
    return res_ridge, res_rotforest


def train_and_test_ROCKETS(
        train_X, train_y, test_X, test_y,
        transformer,
    ):
    # augment data
    train_X = np.array(train_X).transpose(0,2,1)
    test_X  = np.array(test_X).transpose(0,2,1)

    # fit transformer
    t0 = time.time()
    transformer.fit(train_X)
    feat_train_X = np.array(transformer.transform(train_X))
    feat_test_X = np.array(transformer.transform(test_X))
    feat_train_X, feat_test_X = normalize_mean_std_traindata(feat_train_X, feat_test_X)
    t1 = time.time()

    # feed into linear classifier
    res_ridge = train_linear_classifier(
        t0, t1, feat_train_X, feat_test_X, train_y, test_y)
    res_rotforest = train_linear_classifier(
        t0, t1, feat_train_X, feat_test_X, train_y, test_y, RotationForestRegressor(n_estimators=100, n_jobs=trees_n_jobs))
    
    return res_ridge, res_rotforest


def run_all_experiments(X_train, y_train, X_test, y_test):
    prng_key = jax.random.PRNGKey(999)
    max_batch = 32
    trunc_level = 4
    n_features = 1000

    jax_models = [
        ["Random Guesser", RandomGuesser(prng_key, max_batch=max_batch)],
        ["Tabular", TabularTimeseriesFeatures(max_batch)],
        ["Sig", SigTransform(trunc_level, max_batch)],
        ["Log Sig", LogSigTransform(trunc_level, max_batch)],
        ["Randomized Signature", RandomizedSignature(
            prng_key,
            n_features,
            max_batch=10,
            )],
        ["TRP", SigVanillaTensorizedRandProj(
            prng_key,
            n_features,
            trunc_level,
            max_batch,
            concat_levels=False,
            )],
        ["RBF TRP", SigRBFTensorizedRandProj(
            prng_key,
            n_features,
            trunc_level,
            rbf_dimension = 800,
            max_batch = max_batch,
            concat_levels=False,
            )],
        ["concat TRP", SigVanillaTensorizedRandProj(
            prng_key,
            n_features // (trunc_level-1),
            trunc_level,
            max_batch,
            )],
        ["concat RBF TRP", SigRBFTensorizedRandProj(
            prng_key,
            n_features // (trunc_level-1),
            trunc_level,
            rbf_dimension = 800,
            max_batch = max_batch,
            )],
        
        ]

    numpy_seed = 99
    rocket_models = [
        ["Rocket", Rocket(n_features//2, random_state=numpy_seed)],
        ["MiniRocket", MiniRocketMultivariate(n_features, random_state=numpy_seed)],
        ["MultiRocket", MultiRocketMultivariate(n_features//4, random_state=numpy_seed)],
        ]
    
    # Run experiments
    model_names = [name for (name, _) in jax_models+rocket_models]
    results_ridge = []
    results_rotforest = []
    #jax
    for name, model in jax_models:
        res_ridge, res_rotforest = train_and_test_sigbased(
            X_train, y_train, X_test, y_test, model
            )
        results_ridge.append(res_ridge)
        results_rotforest.append(res_rotforest)
        
    #numpy
    for name, model in rocket_models:
        res_ridge, res_rotforest = train_and_test_ROCKETS(
            X_train, y_train, X_test, y_test, model
            )
        results_ridge.append(res_ridge)
        results_rotforest.append(res_rotforest)
    
    return model_names, results_ridge, results_rotforest


def do_experiments(datasets: List[str]):
    experiments = {}
    experiments_metadata = {}
    failed = {}
    for dataset_name in tqdm(datasets):
        t0 = time.time()
        try:
            print(dataset_name)
            X_train, y_train, X_test, y_test = get_aeon_dataset(dataset_name)
            X_train, X_test = normalize_streams(X_train, X_test, max_T=1000)
            y_train, y_test = normalize_mean_std_traindata(y_train, y_test)
            N_train = X_train.shape[0]
            N_test = X_test.shape[0]
            T = X_train.shape[1]
            D = X_train.shape[2]
            results = run_all_experiments(
                X_train, y_train, X_test, y_test
                )
            experiments_metadata[dataset_name] = {
                "N_train": N_train,
                "N_test": N_test,
                "T": T,
                "D": D,
            }
            experiments[dataset_name] = results
        except Exception as e:
            print(f"Error: {e}")
            failed[dataset_name] = e
        print("Elapsed time", time.time()-t0)
    return experiments, experiments_metadata, failed


if __name__ == "__main__":
    d_res, d_meta, d_failed = do_experiments(list(tser_soton))
    
    # Define the attributes and methods
    attributes = ["RMSE_train", "RMSE_test", "time_transform", "time_fit", "alpha"]
    methods = ["ridge", "rotforest"]

    # Create and save DataFrames for each attribute and method
    for attribute in attributes:
        for method in methods:
            df = pd.DataFrame(columns=model_names)
            for dataset_name, (model_names, results_ridge, results_rotforest) in d_res.items():
                if method == "ridge":
                    results = results_ridge
                elif method == "rotforest":
                    results = results_rotforest

                values = [res[attributes.index(attribute)] for res in results]
                df.loc[dataset_name] = values

            # Save the DataFrame
            print(df)
            df.to_pickle(f"TESR_{attribute}_{method}_results.pkl")