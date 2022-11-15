import numpy as np
import pickle
import os

import catboost as cb
import catboost.utils as cbu
import pandas as pd

import hyperopt
from hyperopt import tpe

HYPEROPT_PARAMS_SPACE = {
    'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.1),
    'depth': hyperopt.hp.randint('depth', 3, 7),
    'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 2, 6),
}


class ModelPipeline:
    """
    Catboost regressor model trainer including hyperopt optimization and basic pipeline methods
    """

    def __init__(self, train_pool, val_pool,
                 const_params, max_eval=10):
        self.model = None
        self.hyper_params = None
        self._train_pool = train_pool
        self._val_pool = val_pool
        self._const_params = const_params.copy()
        self.max_eval = max_eval

    def _objective(self, params):
        """
        Hyperopt objective
        """
        model = cb.CatBoostRegressor(**params)
        model.fit(self._train_pool, verbose=0, eval_set=self._val_pool)
        y_pred = model.predict(self._val_pool)
        return cbu.eval_metric(self._val_pool.get_label(), y_pred, 'RMSE')[0]

    def _find_best_hyper_params(self, params={}):
        """
        Find best model parameters by given space
        return: (dict) dictionary with best fitted parameters
        """
        if params:
            parameter_space = params
        else:
            parameter_space = HYPEROPT_PARAMS_SPACE

        parameter_space.update(self._const_params)
        best_model = hyperopt.fmin(
            fn=self._objective,
            space=parameter_space,
            algo=tpe.suggest,
            max_evals=self.max_eval,
            rstate=np.random.default_rng(seed=123))
        return best_model

    def train(self, 
              pretrained_params='',
              param_space={},
              use_pretrained=False,
              use_default=False):
        """
        Train regressor model
        :param pretrained_params: (str) path to pretrained parameters to train with
        :param use_default: (bool) if false will run optimization else model will use pretrained params
        :return: (catboost.core.CatBoostRegressor, dict) model object and parameters
        """
        if use_pretrained:
            # pretrained optimal parameters
            # if not os.path.exists(pretrained_params):
            #     raise ValueError("pretrained params not provided")
            with open(pretrained_params, 'rb') as fp:
                best = pickle.load(fp)

        elif use_default:
            best = self._const_params

        else:
            best = self._find_best_hyper_params(params=param_space)

        hyper_params = best.copy()
        hyper_params.update(self._const_params)

        self.model = cb.CatBoostRegressor(**hyper_params)

        self.model.fit(self._train_pool, eval_set=self._val_pool, verbose=0,
            plot=False)
        hyper_params['iterations'] = self.model.get_best_iteration()
        self.hyper_params = hyper_params
        # params = pd.DataFrame(self.hyper_params)
        # params.to_csv(os.path.join(PATH_TO_DATA, f'params_{self._train_pool}.csv'))

        return self.model, hyper_params

    @staticmethod
    def load_model(model_path):
        """
        Load saved before catboost model
        :param model_path: (str) path to saved model
        :return: catboost.core.CatBoostRegressor object
        """
        model = cb.CatBoostRegressor()
        model.load_model(model_path)
        return model

    def save_params(self, params_path):
        """
        Save model parameters
        :param params_path: (str) path to save parameters once model optimized
        """
        with open(params_path, 'wb') as fp:
            pickle.dump(self.hyper_params, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def save_model(self, model_path):
        """
        Save model
        :param model_path: (str) path to save model once model trained
        """
        self.model.save_model(model_path)
