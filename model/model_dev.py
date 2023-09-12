import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        """
        Train a random forest regressor using the given training data.

        Parameters:
            x_train (array-like): The input training data.
            y_train (array-like): The target training data.
            **kwargs: Additional keyword arguments to be passed to the RandomForestRegressor.

        Returns:
            RandomForestRegressor: The trained random forest regressor model.
        """
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimize the model using the given trial and training and testing data.

        Args:
            trial (optuna.Trial): The trial object used for hyperparameter optimization.
            x_train (numpy.ndarray): The training data.
            y_train (numpy.ndarray): The training labels.
            x_test (numpy.ndarray): The testing data.
            y_test (numpy.ndarray): The testing labels.

        Returns:
            float: The score of the optimized model on the testing data.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)

class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        """
        Trains a LightGBM regressor model.

        Args:
            x_train (array-like): The input features for training the model.
            y_train (array-like): The target values for training the model.
            **kwargs: Additional keyword arguments to be passed to the LGBMRegressor constructor.

        Returns:
            LGBMRegressor: The trained LightGBM regressor model.
        """
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimize the model using hyperparameter tuning.

        Args:
            trial (optuna.Trial): The trial object for hyperparameter optimization.
            x_train (array-like): The training data.
            y_train (array-like): The target values for the training data.
            x_test (array-like): The test data.
            y_test (array-like): The target values for the test data.

        Returns:
            float: The score of the optimized model on the test data.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(x_test, y_test)


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        """
        Train the XGBRegressor model on the given training data.

        Parameters:
            x_train (array-like): The training input samples.
            y_train (array-like): The target values.
            **kwargs: Additional arguments to be passed to the XGBRegressor constructor.

        Returns:
            reg (XGBRegressor): The trained XGBRegressor model.
        """
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimize the parameters of the model using Optuna's hyperparameter optimization.

        Parameters:
            trial (optuna.trial.Trial): The Optuna trial object used for hyperparameter search.
            x_train (ndarray): The training input samples.
            y_train (ndarray): The target values for the training input samples.
            x_test (ndarray): The testing input samples.
            y_test (ndarray): The target values for the testing input samples.

        Returns:
            float: The score of the optimized model on the testing data.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(x_test, y_test)


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        """
        Train a linear regression model using the provided training data.

        Parameters:
            x_train (array-like): The input features for training the model.
            y_train (array-like): The target values for training the model.
            **kwargs: Additional keyword arguments to be passed to the LinearRegression constructor.

        Returns:
            reg (LinearRegression): The trained linear regression model.
        """
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimize the model using the given trial data and calculate the score on the test data.

        Parameters:
            trial (object): The trial data for optimization.
            x_train (array-like): The training data for the model.
            y_train (array-like): The target values for the training data.
            x_test (array-like): The test data for evaluating the model.
            y_test (array-like): The target values for the test data.
        
        Returns:
            float: The score of the model on the test data.
        """
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        """
        Optimize the model using Optuna.

        Args:
            n_trials (int, optional): The number of trials for optimization. Default is 100.

        Returns:
            dict: The best trial parameters.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params