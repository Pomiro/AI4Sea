import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error\


class PollockSplitter:
    """
    Split pollock dataset data by ships and theirs routes
    """

    @staticmethod
    def multi_sort(full_set, subset):
        """
        Sort dataframe on given subset
        :return: pd.DataFrame
        """
        keys = list(subset.columns.values)
        i1 = full_set.set_index(keys).index
        i2 = subset.set_index(keys).index
        df = full_set[i1.isin(i2)]
        return df

    @staticmethod
    def train_test_split(data,
                         test_fraction=0.25,
                         seed=48):
        """
        Split data to train/test by ships, non-unique trawls and daily routes
        :param data: (pd.DataFrame) input data
        :param test_fraction: size of test fraction (as splitting only on non-science trawls, real fraction slightly
        less)
        :param seed: seed value
        :return: (pd.DataFrame) train and test splitted dataframes
        """
        data['date'] = data['datetimes'].dt.date
        trawls_list = data[data['science'] == 0].copy()
        trawls_list = trawls_list.groupby('trawl')['idves'].nunique().reset_index()
        trawls_list = trawls_list[trawls_list['idves'] >= 2]['trawl'].values.tolist()

        train_sample = data.sort_values(['idves', 'datetimes']).groupby(['idves', 'date', 'trawl'],
                                                                        as_index=False).count()[
            ['idves', 'date', 'trawl']]
        unique_trawls_train = train_sample[~train_sample['trawl'].isin(trawls_list)]
        train_sample = train_sample[train_sample['trawl'].isin(trawls_list)]

        test_sample = train_sample.sample(frac=test_fraction,
                                          random_state=seed)
        train_sample.drop(test_sample.index, inplace=True)
        test_sample = PollockSplitter.multi_sort(data, test_sample)

        train_sample = train_sample.append(unique_trawls_train)
        train_sample = PollockSplitter.multi_sort(data, train_sample)

        return train_sample, test_sample


class ModelExplainer:
    """
    Methods to create feature importance and validation metric plots
    """
    @staticmethod
    def features_plot(model, features, target):
        """
        Feature importance plots
        :param model: (catboost.core.CatBoostRegressor, dict) model
        :param features: features data (X)
        :param target: target data (y)
        """
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(features.columns)[sorted_idx])
        plt.title('Feature Importance');

        perm_importance = permutation_importance(model, features, target, n_repeats=10, random_state=1066)
        sorted_idx = perm_importance.importances_mean.argsort()
        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(features.columns)[sorted_idx])
        plt.title('Permutation Importance');

        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        shap_importance = shap_values.abs.mean(0).values
        sorted_idx = shap_importance.argsort()
        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), shap_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(features.columns)[sorted_idx])
        plt.title('SHAP Importance');
        fig = plt.figure(figsize=(12, 6))
        shap.plots.bar(shap_values, max_display=features.shape[0])

    @staticmethod
    def val_plot(model, val_metric='RMSE'):
        """
        Validation metric plots
        :param model: (catboost.core.CatBoostRegressor, dict) model
        :param val_metric: metric was used to eval model during training process e.g.
        main.py: MODEL_CONST_PARAM['eval_metric']
        """
        plt.figure(figsize=(10, 7))
        plt.plot(model.evals_result_["learn"][val_metric], label="Training Correlation")
        plt.plot(model.evals_result_["validation"][val_metric], label="Validation Correlation")
        plt.xlabel("Number of trees")
        plt.ylabel("Correlation")
        plt.legend()


def aij_metric(pred_data, truth_data, threshold=0.1):
    """
    aij main metric if r2 less than threshold(0.1) add extra penalties
    :pred_data: (list) model
    :truth_data: (list) truth values
    :threshold: (float) threshold value
    :return: (float) aij metric value
    """
    r2 = r2_score(pred_data, truth_data)
    rmse = mean_squared_error(pred_data, truth_data, squared=False)
    if r2 < threshold:
        return 10
    else: return rmse