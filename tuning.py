import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score# , balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gaussian_kde
from hyperopt import fmin, hp, tpe
import warnings
from utils import *


def posteriors_pu_cv(preds, target_pu, kde_inner_fun=lambda x: x, kde_outer_fun=lambda dens, x: dens(x),
                     bw_mix=0.05, bw_pos=0.15, threshold=0.55, k_neighbours=None, n_splits=10, random_state=None):

    if k_neighbours is None:
        k_neighbours = int(preds.shape[0] // 10)

    sorted_preds_idx = np.argsort(preds)
    sorted_preds = preds[sorted_preds_idx]

    # kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # posters = np.zeros(sorted_preds.shape[0])
    #
    # for train_index, test_index in kfold.split(preds, target_pu):
    #     train_preds = preds[train_index]
    #     train_target = target_pu[train_index]
    #     test_preds = preds[test_index]
    #
    #     kde_mix = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, train_preds[train_target == 1]), bw_mix)
    #     kde_pos = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, train_preds[train_target == 0]), bw_pos)
    #
    #     posters[test_index] = np.apply_along_axis(
    #         lambda x: kde_outer_fun(kde_pos, x) / (kde_outer_fun(kde_mix, x) + 10 ** -5), axis=0,
    #         arr=test_preds)

    kde_mix = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[target_pu == 1]), bw_mix)
    kde_pos = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[target_pu == 0]), bw_pos)
    posters = np.apply_along_axis(lambda x: kde_outer_fun(kde_pos, x) / (kde_outer_fun(kde_mix, x) + 10 ** -10), axis=0,
                               arr=preds)

    posters = posters[sorted_preds_idx]
    posters = np.append(np.flip(np.maximum.accumulate(np.flip(posters[sorted_preds <= threshold], axis=0)), axis=0),
                        posters[sorted_preds > threshold])
    posters = rolling_apply(posters, k_neighbours, np.median, axis=-1)

    # desorting
    posters = posters[np.argsort(sorted_preds_idx)]
	
    posters = 1 / (posters + 1)# + preds * 10 ** -5 # if optimizing roc-auc, this will increase it in flat regions
    # posters[posters > 1] = 1

    return posters


# function copied from sklearn
def balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def posteriors_pu_cv_metric(preds, target_pu, kde_inner_fun=lambda x: x, kde_outer_fun=lambda dens, x: dens(x),
                            metric=brier_score_loss, if_round=False, reweight=True,
                            bw_mix=0.05, bw_pos=0.15, threshold=0.55, k_neighbours=None):

    posters = posteriors_pu_cv(preds, target_pu, kde_inner_fun, kde_outer_fun,
                               bw_mix, bw_pos, threshold, k_neighbours)
    if if_round:
        posters = posters.round()
    if reweight:
        weights = np.full(posters.shape, target_pu.mean())
        weights[target_pu == 1] = 1 - target_pu.mean()
        return metric(target_pu, posters, sample_weight=weights)
    else:
        return metric(target_pu, posters)


def tune_diff(preds, target_pu, kde_inner_fun=lambda x: x, kde_outer_fun=lambda dens, x: dens(x),
              metric=brier_score_loss, if_round=False, reweight=True,
              bw_mix_l=0.01, bw_mix_h=0.12, bw_pos_l=0.08, bw_pos_h=0.25, threshold_l=0.55, threshold_h=0.55,
              k_neighbours_l=None, k_neighbours_h=None, k_neighbours_default=None,
              bw_mix_default=0.05, bw_pos_default=0.15, threshold_default=0.55, max_evals=20, verbose=False,
              choose_randomly=False):

    if k_neighbours_default is None:
        k_neighbours_default = int(preds.shape[0] // 10)
    if k_neighbours_l is None:
        k_neighbours_l = int(preds.shape[0] // 25)
    if k_neighbours_h is None:
        k_neighbours_h = int(preds.shape[0] // 5)

    # this is to compare randomly chosen parameters with tuned parameters
    if choose_randomly:
        bw_mix_best = loguniform(np.log(bw_mix_l), np.log(bw_mix_h))
        bw_pos_best = loguniform(np.log(bw_pos_l), np.log(bw_pos_h))
        threshold_best = np.random.uniform(threshold_l, threshold_h)
        k_neighbours_best = int(loguniform(np.log(k_neighbours_l), np.log(k_neighbours_h)))

    else:
        space = {'bw_mix': hp.loguniform('bw_mix', np.log(bw_mix_l), np.log(bw_mix_h)),
                 'bw_pos': hp.loguniform('bw_pos', np.log(bw_pos_l), np.log(bw_pos_h)),
                 'k_neighbours': hp.loguniform('k_neighbours', np.log(k_neighbours_l), np.log(k_neighbours_h)),
                 'threshold': hp.uniform('threshold', threshold_l, threshold_h)}

        def objective(x):
            loss = posteriors_pu_cv_metric(preds, target_pu, kde_inner_fun, kde_outer_fun,
                                           metric=metric, if_round=if_round, k_neighbours=int(x['k_neighbours']),
                                           bw_mix=x['bw_mix'], bw_pos=x['bw_pos'], threshold=x['threshold'],
                                           reweight=reweight)
            # print(loss)
            return loss

        params = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)
        bw_mix_best, bw_pos_best, threshold_best, k_neighbours_best = params['bw_mix'], params['bw_pos'], \
                                                                      params['threshold'], int(params['k_neighbours'])

    # best_score = posteriors_pu_cv_metric(preds, target_pu, kde_inner_fun, kde_outer_fun,
    #                                      metric=metric, if_round=if_round, k_neighbours=k_neighbours_best,
    #                                      bw_mix=bw_mix_best, bw_pos=bw_pos_best, threshold=threshold_best,
    #                                      reweight=reweight)
    # default_score = posteriors_pu_cv_metric(preds, target_pu, kde_inner_fun, kde_outer_fun,
    #                                         metric=metric, if_round=if_round,
    #                                         bw_mix=bw_mix_default, bw_pos=bw_pos_default,
    #                                         threshold=threshold_default, k_neighbours=k_neighbours_default,
    #                                         reweight=reweight)

    if verbose:

        print('best params:', round(bw_mix_best, 4), round(bw_pos_best, 4), round(threshold_best, 4), k_neighbours_best)
    #     print('best score:', best_score)
    #     print('default params:', bw_mix_default, bw_pos_default, threshold_default)
    #     print('default score:', default_score)
	#
    # if best_score < default_score:
    #     return bw_mix_best, bw_pos_best, threshold_best
    # else:
    #     return bw_mix_default, bw_pos_default, threshold_default

    return bw_mix_best, bw_pos_best, threshold_best, k_neighbours_best
