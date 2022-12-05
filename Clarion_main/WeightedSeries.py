

import numpy as np
from sklearn.base import BaseEstimator, clone




class WeightedSeries(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator


    def fit(self,X,Y):

        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)  #Validate input data and set or check the `n_features_in_` attribute.

        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        self.estimators_mix = [clone(self.base_estimator) for _ in range(Y.shape[1])]

        self.order_ = range(Y.shape[1])


        for chain_idx, extimator in enumerate(self.estimators_):
            y = Y[:,self.order_[chain_idx]]
            x = X
            extimator.fit(x,y)

        for chain_idx, estimator in enumerate(self.estimators_mix):

            y = Y[:, self.order_[chain_idx]]
            order = list(self.order_)
            order.remove(self.order_[chain_idx])
            y_tmp = Y[:,order]
            x = np.hstack((X,y_tmp))
            estimator.fit(x, y)

        return self




    def predict(self, X,weight):
        Y_pred_single = np.zeros((X.shape[0], len(self.estimators_)))  # zero matrix
        Y_pred_mix = np.zeros((X.shape[0], len(self.estimators_)))
        self.order_ = range(len(self.estimators_))
        self.weight = weight

        for chain_idx, estimator in enumerate(self.estimators_):
            Y_pred_single[:,chain_idx] = estimator.predict_proba(X)[:,1]

        for chain_idx, estimator in enumerate(self.estimators_mix):
            order = list(self.order_)
            order.remove(self.order_[chain_idx])
            train_predictions = Y_pred_single[:, order]
            x = np.hstack((X,train_predictions))
            Y_pred_mix[:,chain_idx] = estimator.predict_proba(x)[:,1]

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_)) # = order

        Y_pred_sm = np.zeros((X.shape[0], len(self.estimators_)))
        for i in range(len(self.estimators_)):
            y_s = Y_pred_single[:,i]

            y_m = Y_pred_mix[:,i]
            yy = self.weight*y_s + (1-self.weight)*y_m
            yyy = np.array([round(i) for i in yy])

            Y_pred_sm[:,i] = yyy


        Y_pred = Y_pred_sm[:, inv_order]

        return Y_pred





































