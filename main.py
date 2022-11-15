# This is a sample Python script.
import argparse
import logging
import os
import pathlib
import warnings

import catboost as cb
import numpy as np
import pandas as pd
from catboost.utils import get_gpu_device_count
from sklearn.metrics import r2_score, mean_squared_error
import pickle

from feature_engineering import FeaturePipeline
from model import ModelPipeline
from utils import PollockSplitter, aij_metric

pd.options.mode.chained_assignment = None
warnings.simplefilter("ignore")

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SEED = 5
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED)

BASE_DIR = pathlib.Path(__file__).parent
PATH_TO_DATA = BASE_DIR / "data/"
PATH_TO_MODELS = BASE_DIR / "models/"
PATH_TO_OUTPUT = BASE_DIR / "output/"
PATH_TO_INPUT = BASE_DIR / "../input"

# provided dataset to solve problem
DATASET_FILENAME = "dataset.csv"
# sample how open/hidden tests files looks like
SUBMISSION_FILENAME = "test_features.csv"
# filenames will create after feature engineering step
TRAIN_FILENAME = 'train.csv'
VAL_FILENAME = 'val.csv'

assert TRAIN_FILENAME != DATASET_FILENAME
assert VAL_FILENAME != DATASET_FILENAME

# constant params to train model
MODEL_CONST_PARAM = dict({
    'task_type': 'GPU' if get_gpu_device_count() else 'CPU',
    'eval_metric': 'RMSE',
    'loss_function': 'RMSE',
    'random_seed': 48})

# HP optimization iteration
HYPEROPT_ITERATIONS = 50
# fraction size of validation set
VAL_SIZE = 0.2
YEARS = [2021, 2020, 2019, 2018, 2017, 2016, 2015]

def _run_pipeline(
                  prepare_data,
                  train_models,
                  optimization,
                  pretrained_params,
                  load_models,
                  save_params_path,
                  save_models,
                  eval_model,
                  make_solution
                  ):
    if prepare_data:
        logger.info(f"Read input data")
        input_data = pd.read_csv(os.path.join(PATH_TO_DATA, DATASET_FILENAME), parse_dates=['datetimes'])
        input_data['year'] = input_data['datetimes'].dt.year
        logger.info(f"Split input data to train/val")
        for year in YEARS:
            feature_obj = FeaturePipeline(logger, PATH_TO_MODELS)
            logger.info(f"Starting feature generation")
            train = input_data[input_data['year'] != year]
            val = input_data[input_data['year'] == year]
            feature_obj.make_features(train, path_to_save=os.path.join(PATH_TO_DATA, f'train_{year}.csv'))
            feature_obj.make_features(val, path_to_save=os.path.join(PATH_TO_DATA, f'val_{year}.csv'))
            
    if train_models:
        logger.info(f"Read training data")
        for year in YEARS:
            train = pd.read_csv(os.path.join(PATH_TO_DATA, f'train_{year}.csv'), index_col=0).sample(frac=1, random_state=year)
            val = pd.read_csv(os.path.join(PATH_TO_DATA, f'val_{year}.csv'), index_col=0)
            logger.info(f"Prepare catboost Pool objects")

            X_train, y_train = train.drop(['ton'], axis=1), train['ton']
            X_val, y_val = val.drop(['ton'], axis=1), val['ton']

            train_pool = cb.Pool(X_train, y_train,
                            cat_features=np.where(X_train.dtypes != float)[0])
            val_pool = cb.Pool(X_val, y_val,
                                        cat_features=np.where(X_val.dtypes != float)[0])
            logger.info(f"Create model {year} object")
            model_obj = ModelPipeline(train_pool, val_pool, MODEL_CONST_PARAM, max_eval=HYPEROPT_ITERATIONS)
            if pretrained_params:
                logger.info(f"Train model with given pretrained params, optimization off")
                pretrained_params = os.path.join(PATH_TO_MODELS, f'params_{year}')
                model, params = model_obj.train(pretrained_params=pretrained_params, use_pretrained=True)
            elif optimization:
                logger.info(f"Train model from scratch, optimization on")
                model, params = model_obj.train(use_default=False)
                with open(f'models/params_{year}', 'wb') as fp:
                    pickle.dump(params, fp, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                logger.info(f"Train model from scratch (user given params), optimization off")
                model, params = model_obj.train(use_default=True)

            if save_models:
                logger.info(f"Saving model {year} weights")
                model_obj.save_model(os.path.join(PATH_TO_MODELS, f'model_{year}'))

            if save_params_path:
                logger.info(f"Saving model params")
                model_obj.save_params(os.path.join(PATH_TO_MODELS, f'params_{year}'))

    if load_models:
        logger.info(f"Loading models weights")
        models = []
        for year in YEARS:
            load_model_path = os.path.join(PATH_TO_MODELS, f'model_{year}')
            models.append(ModelPipeline.load_model(load_model_path))

    if eval_model:
        logger.info(f"Starting model evaluation")
        aij_sum = 0
        for year in YEARS:
            val = pd.read_csv(os.path.join(PATH_TO_DATA, f'val_{year}.csv'), index_col=0)#.drop(columns='Unnamed: 0')
            X_val, y_val = val.drop(['ton'], axis=1), val['ton']
            y_pred = 0
            for model in models:
                y_pred += model.predict(X_val)
            y_pred /= 7
            r2 = r2_score(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            aij = aij_metric(y_val, y_pred)
            aij_sum += aij
            logger.info(f"Calculate metric score:{'R2: {:.2f}'.format(r2)} {'RMSE: {:.3f}'.format(rmse)}")
            logger.info(f"Calculate aij score:{'AIJ metric: {:.2f}'.format(aij)}")
        aij_sum /= 7
        logger.info(aij_sum)

    if make_solution:
        logger.info(f"Preparing solution file")
        sub_data = pd.read_csv(os.path.join(PATH_TO_INPUT, SUBMISSION_FILENAME), parse_dates=['datetimes'])
        feature_obj = FeaturePipeline(logger, PATH_TO_MODELS)
        pred_data = feature_obj.make_features(sub_data, train=False)
        y_pred = 0
        q = 0
        b = 10
        y_pred += models[0].predict(pred_data[models[0].feature_names_]) * np.exp((10 - 3.547)*b)
        q += np.exp((10 - 3.547)*b)
        y_pred += models[1].predict(pred_data[models[1].feature_names_]) * np.exp((10 - 2.779)*b)
        q += np.exp((10 - 2.779)*b)
        y_pred += models[2].predict(pred_data[models[2].feature_names_]) * np.exp((10 - 2.107)*b)
        q += np.exp((10 - 2.107)*b)
        y_pred += models[3].predict(pred_data[models[3].feature_names_]) * np.exp((10 - 3.217)*b)
        q += np.exp((10 - 3.217)*b)
        y_pred += models[4].predict(pred_data[models[4].feature_names_]) * np.exp((10 - 8.543)*b)
        q += np.exp((10 - 8.543)*b)
        y_pred += models[5].predict(pred_data[models[5].feature_names_]) * np.exp((10 - 8.498)*b)
        q += np.exp((10 - 8.498)*b)
        y_pred += models[6].predict(pred_data[models[6].feature_names_]) * np.exp((10 - 4.606)*b)
        q += np.exp((10 - 4.606)*b)

        y_pred /= q
        pred_data['ton'] = y_pred
        pred_data = pred_data.reset_index()
        pred_data = pred_data.rename(columns={'index':'id'})
        pred_data[['id','ton']].to_csv(os.path.join(PATH_TO_OUTPUT, 'submission.csv'), header=True, index=False)
        logger.info(f"Solution file created")


def _handle_args(args):
    """
    Extract command line args and call delegate function.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """

    return _run_pipeline(
                         args.prepare_data,
                         args.train_models,
                         args.optimization,
                         args.pretrained_params,
                         args.load_models,
                         args.save_params,
                         args.save_models,
                         args.eval_model,
                         args.make_solution)


def main(args=None):
    """
    Parse command line args and call handler when run as a script.
    Parameters
    ----------
    args : list
        Command line arguments as a list of strings [optional]
    """
    # Add parser args and default handler
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdcv", "--prepare_data", action='store_true',
                        help="Split input data to train/val cv, create features and save")
    parser.add_argument("-itmcv", "--train_models", action='store_true',
                        help="Train model cv")
    parser.add_argument("-opt", "--optimization", action='store_true',
                        help="Run hyperopt loop to optimize weights")
    parser.add_argument("-p", "--pretrained_params", action='store_true',
                        help="Pretrained catboost params")
    parser.add_argument("-lms", "--load_models", action='store_true',
                        help="Name to files with catboost models weights")
    parser.add_argument("-sms", "--save_models", action='store_true',
                        help="Name to save models weights, works only with '--train_model_cv'")
    parser.add_argument("-sp", "--save_params", type=str, default=None,
                        help="Name to save model hyper params, works only with '--train_model'")
    parser.add_argument("-e", "--eval_model", action='store_true',
                        help="Calculate R2 and RMSE metric on val data")
    parser.add_argument("-sub_cv", "--make_solution", action='store_true',
                        help="Create solution csv file and necessary archive")
    parser.set_defaults(func=_handle_args)
    args = parser.parse_args()

    # Call args default handler
    args.func(args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()