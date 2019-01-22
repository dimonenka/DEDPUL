import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, laplace, t
from statsmodels.stats.multitest import multipletests


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    roll_arr = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return roll_arr


def rolling_apply(a, window, fun, apply_to_all=True, **kwargs):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    assert a.shape[0] >= window * 2, 'window is too big'

    if apply_to_all:
        left = np.array([0.] * window)
        right = np.array([0.] * window)
        for i in range(window):
            left[i] = fun(a[: 2 * i + 1], **kwargs)
            right[-i - 1] = fun(a[-2 * i - 1:], **kwargs)

    else:
        left = a[:window]
        right = a[-window:]
    return np.concatenate([left, fun(rolling_window(a, window * 2 + 1), **kwargs), right])


def reg_to_class(s):
    return (s > s.mean()).astype(int)


def mul_to_bin(s, border=None):
    if border is None:
        border = s.median()
    return (s > border).astype(int)


def dummy_encode(df):
    """
   Auto encodes any dataframe column of type category or object.
   """

    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df


def normalize_col(s):
    std = s.std()
    mean = s.mean()
    if std > 0:
        return (s - s.mean()) / s.std()
    else:
        return s - s.mean()


def normalize_cols(df, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = normalize_col(df[col])
    return df


def generate_data(mix_size, pos_size, alpha, delta_mu=2, multiplier_s=1, distribution='normal',
                  random_state=None):
    """
    Generates pu data
    :param mix_size: total size of mixed data
    :param pos_size: total size of positive data
    :param alpha: share of negatives in mixed data
    :param delta_mu: center of negative distribution (positive is centered at 0)
    :param multiplier_s: std of negative distribution (positive has std 1)
    :param distribution: either 'normal' or 'laplace'
    :return: tuple (data, target_pu, target_mix)
        data - generated points, np.array shaped as (n_instances, 1)
        target_pu - label of data: 0 if positive, 1 is unlabeled, np.array shaped as (n_instances,)
        target_mix - label of data: 0 if positive in unlabeled, 1 if negative in unlabeled,
            2 if positive, np.array shaped as (n_instances,)
        target_pu == 0 <=> target_mix == 2; target_pu == 1 <=> target_mix == 0 or target_mix == 1
    """

    if distribution == 'normal':
        sampler = np.random.normal
    elif distribution == 'laplace':
        sampler = np.random.laplace

    np.random.seed(random_state)

    mix_data = np.append(sampler(0, 1, int(mix_size * (1 - alpha))),
                         sampler(delta_mu, multiplier_s, int(mix_size * alpha)))
    pos_data = sampler(0, 1, pos_size)

    data = np.append(mix_data, pos_data).reshape((-1, 1))
    target_mix = np.append(np.zeros((int(mix_size * (1 - alpha)),)), np.ones((int(mix_size * alpha),)))
    target_mix = np.append(target_mix, np.full((pos_size,), 2))

    index = np.arange(data.shape[0])
    np.random.shuffle(index)

    data = data[index]
    target_mix = target_mix[index]
    target_pu = target_mix.copy()
    target_pu[target_pu == 0] = 1
    target_pu[target_pu == 2] = 0

    np.random.seed(None)

    return data, target_pu, target_mix


def estimate_cons_alpha(dmu, ds, alpha, distribution):
    if distribution == 'normal':
        pdf = norm.pdf
    elif distribution == 'laplace':
        pdf = laplace.pdf

    p1 = lambda x: pdf(x, 0, 1)
    p2 = lambda x: pdf(x, dmu, ds)
    pm = lambda x: p1(x) * (1 - alpha) + p2(x) * alpha

    if dmu == 0 and ds == 1:
        cons_alpha = 0
    elif distribution == 'laplace' and ds == 1:
        cons_alpha = (1 - pm(-100) / p1(-100)).item()
    elif distribution == 'normal' and ds == 1:
        cons_alpha = alpha
    elif dmu == 0:
        cons_alpha = (1 - pm(0) / p1(0)).item()
    else:
        raise NotImplemented('only cases where either dmu=0 or ds=1 and distribution is ' + \
                             'either normal or laplace are implemented')
    return cons_alpha


def estimate_cons_poster(points, dmu, ds, distribution, alpha, cons_alpha=None):
    if distribution == 'normal':
        pdf = norm.pdf
    elif distribution == 'laplace':
        pdf = laplace.pdf

    p1 = lambda x: pdf(x, 0, 1)
    p2 = lambda x: pdf(x, dmu, ds)
    pm = lambda x: p1(x) * (1 - alpha) + p2(x) * alpha

    if cons_alpha is None:
        cons_alpha = estimate_cons_alpha(dmu, ds, alpha, distribution)

    cons_poster = np.apply_along_axis(lambda x: (pm(x) - p1(x) * (1 - cons_alpha)) / pm(x), -1, points)
    return cons_poster


def test_significance(res, metric, est_1, est_2, group_by=None, p_value=0.05, correction='holm'):
    if group_by is None:
        group_by = ['dataset', 'alpha']
    res_pivot = res.pivot_table(index=group_by + ['random_state'],
                                columns=['estimator'],
                                values=metric)[[est_1, est_2]]
    res_pivot['diff'] = res_pivot[est_1] - res_pivot[est_2]
    res_pivot.reset_index(inplace=True)
    n = res_pivot['random_state'].nunique()
    res_pivot = res_pivot.groupby(group_by)['diff'].agg((np.mean, np.std))
    res_pivot['std'] *= n / (n - 1)
    res_pivot['t-stat'] = res_pivot['mean'].abs() / res_pivot['std'] * np.sqrt(n)
    res_pivot['p_value'] = 1 - res_pivot['t-stat'].apply(t.cdf, df=n-1)
    if correction is not None:
        res_pivot['p_value'] = multipletests(res_pivot['p_value'].values, alpha=p_value, method=correction)[1]
    return res_pivot['p_value'] < p_value
