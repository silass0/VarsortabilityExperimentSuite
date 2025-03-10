import numpy as np
import itertools
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (LinearRegression,
                                  LassoLarsIC,
                                  LassoLarsCV)
from sklearn.metrics import mean_absolute_error as MSE
from sklearn.model_selection import KFold
import statsmodels.api as sm


def sortnregress(X, regularisation='1-p', random_order=False):
    """ Takex n x d data, assumes order is given by increased variance,
    and regresses each node onto those with lower variance, using
    edge coefficients as structure estimates.

    regularisation:
      bic - adaptive lasso
      1-p - 1-p-vals of OLS LR (such that larger = edge)
      None / other - raw OLS LR coefficients
    """

    assert regularisation in ['bic', '1-p', 'raw', 'LassoLarsCV']

    LR = LinearRegression()
    if regularisation == 'bic':
        LL = LassoLarsIC(criterion='bic')
    elif regularisation == 'LassoLarsCV':
        LL = LassoLarsCV()

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, 0))

    if random_order:
        np.random.shuffle(increasing)

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        if regularisation == '1-p':
            ols = sm.OLS(
                X[:, target],
                sm.add_constant(X[:, covariates])).fit()
            W[covariates, target] = 1 - ols.pvalues[1:]
        elif regularisation in ['bic', 'LassoLarsCV']:
            LR.fit(X[:, covariates], X[:, target].ravel())
            weight = np.abs(LR.coef_)
            LL.fit(X[:, covariates] * weight, X[:, target].ravel())
            W[covariates, target] = LL.coef_ * weight
        elif regularisation == 'raw':
            LR.fit(X[:, covariates],
                   X[:, target].ravel())
            W[covariates, target] = LR.coef_
        else:
            raise ValueError("no such regularization")

    return W


def sortnregress_quad(X, degree=2, max_indegree=5, include_interactions=False, random_order=False):
    """Takex n x d data, assumes order is given by increased variance,
    and regresses each node polynomially onto those with lower variance, using
    edge coefficients as structure estimates. Features are eliminated #TODO

    Args:
        X (ndarray): data array
        degree (int, optional): Degree of polynomial regression
        max_indegree (int, optional): Maximal indegree of every node in network. Defaults to 5.
        include_interactions (bool, optional): Include covariate interactions in regression step.
        random_order (bool, optional): If True, performs randomregress instead. Defaults to False.
    """
    
    d = X.shape[1]
    W = np.zeros((d, d))
    X_intercept = np.ones((X.shape[0], 1))
    increasing = np.argsort(np.var(X, 0))
    n_splits = 5
    kf = KFold(n_splits)

    if random_order:
        np.random.shuffle(increasing)

    if include_interactions:
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                                LassoLarsIC(criterion='bic'))
    else:
        from _utils import quadratic_features
        model = make_pipeline(quadratic_features(),
                            LassoLarsIC(criterion='bic'))
    model_no_par = LassoLarsIC(criterion='bic', fit_intercept=False)

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        subsets = [[c] for c in covariates]
        for s in range(2, min(len(covariates), max_indegree)+1):
            subsets += list(itertools.combinations(covariates, s))

        #First check: no parents. 
        score = 0
        for train_index, validation_index in kf.split(X):
            model_no_par.fit(X_intercept[train_index,:], X[train_index, target].ravel())
            score += model_no_par.score(X_intercept[validation_index,:], X[validation_index, target].ravel()) / n_splits
        best_model = {"subset":[], "score":score} # subset, score
        #Next check: all subsets up to certain size (prioritizing smaller subsets via a threshold)
        for subset in subsets:
            #model selection CV
            score = 0
            for train_index, validation_index in kf.split(X):
                model.fit(X[np.ix_(train_index, subset)], X[train_index, target].ravel())
                score += model.score(X[np.ix_(validation_index, subset)], X[validation_index, target].ravel()) / n_splits
            if score - best_model["score"] > 0.01: #choose new subset if threshold is reached
                best_model["subset"] = subset
                best_model["score"] = score 

        subset = best_model["subset"]

        if len(subset) > 1:
            #compute MSE
            model.fit(X[:, subset], X[:, target].ravel())
            mse = MSE(X[:, target].ravel(), model.predict(X[:, subset]))
            #Make entries in weight matrix for inferred parents
            for p in subset:
                one_out = list(subset)
                one_out.remove(p)
                model.fit(X[:, one_out], X[:, target].ravel())
                mse_one_out = MSE(X[:, target].ravel(), model.predict(X[:, one_out]))
                W[p, target] = (mse_one_out - mse) / mse
        elif len(subset) == 1:
            W[subset[0], target] = 1.0 
    return W


def sortnregress_poly_heu(X, degree=2, max_indegree=5, random_order=False):
    """Takex n x d data, assumes order is given by increased variance,
    and regresses each node polynomially onto those with lower variance, using
    edge coefficients as structure estimates. Features are eliminated #TODO

    Args:
        X (ndarray): data array
        degree (int, optional): Degree of polynomial regression
        max_indegree (int, optional): Maximal indegree of every node in network. Defaults to 3.
        include_interactions (bool, optional): Include covariate interactions in regression step. Defaults to False.
        random_order (bool, optional): If True, performs randomregress instead. Defaults to False.
    """

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, 0))

    if random_order:
        np.random.shuffle(increasing)

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        # Feature selection
        best_model = {"model":"model", "subset":"array", "score":-np.inf} # model, subset, score
        if len(covariates) <= max_indegree:
            subsets = [covariates]
            W1 = np.zeros((len(covariates))) #order analogue to covariates
            W2 = np.zeros((len(covariates), len(covariates)))
        else: #will not check lesser subsets, as they will result in lower score
            subsets = list(itertools.combinations(covariates, max_indegree))
            W1 = np.zeros((max_indegree))
            W2 = np.zeros((max_indegree, max_indegree))
        for subset in subsets:
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
            model.fit(X[:, subset], X[:, target].ravel())
            score = model.score(X[:, subset], X[:, target].ravel())
            if score > best_model["score"]:
                best_model["model"] = model
                best_model["subset"] = subset
                best_model["score"] = score 
        model = best_model["model"]
        subset = best_model["subset"]

        #build W1 and W2
        # feature order: x0, x1, x2, x0^2, x0 x1, x0 x2, x1^2, x1 x2, x2^2 (for 2 covariates)
        for i, feat_name in enumerate(model['polynomialfeatures'].get_feature_names()):
            pair_search = re.search(r"x([0-9]+) x([0-9]+)", feat_name)
            if pair_search: #interactions
                match = [int(gr) for gr in pair_search.groups()]
                W2[match[0], match[1]] = W2[match[1], match[0]] =  model['linearregression'].coef_[i]
                continue
            quad_search = re.search(r"x([0-9]+)\^2", feat_name)
            if quad_search: #quadratics
                match = int(quad_search.groups()[0])
                W2[match, match] = model['linearregression'].coef_[i]
                continue
            lin_search = re.search(r"x([0-9]+)", feat_name)
            if lin_search:
                match = int(lin_search.groups()[0])
                W1[match] = model['linearregression'].coef_[i]
                continue

        #aggregate polynomial weights and interactions into a single output matrix W 
        heuristic = {"lin":0.7, "quad":0.2, "inters":0.1} #for single cov, target
        for i, cov in enumerate(subset):
            w = 0
            w += np.abs(W1[i]) * heuristic["lin"]
            w += np.abs(W2[i, i]) * heuristic["quad"]
            inters = [w for j, w in enumerate(W2[i, :]) if j!=i]
            if len(inters) > 0:
                w += np.abs(np.mean(inters)) * heuristic["inters"]
            W[cov, target] = w
        #Possible TODO: make it so interactions only exists for dependent covariates (but regression should assign interactions with small weights if indep)
        #Possible TODO: Develop better heuristic for weigh aggregation (importance for various weight degrees)
    return W