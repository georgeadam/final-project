import copy as copylib

from sklearn.preprocessing import StandardScaler

from .creation import transforms


class Scaler(StandardScaler):
    def __init__(self, cols):
        super(Scaler, self).__init__()
        self.cols = cols

    def fit(self, X, y):
        if self.cols is not None:
            super(Scaler, self).fit(X[:, self.cols])
        else:
            super(Scaler, self).fit(X)

    def transform(self, X, copy=None):
        if self.cols is not None:
            X_copy = copylib.deepcopy(X)
            orig_shape = X_copy.shape

            if len(orig_shape) < 2:
                X_copy = X_copy.reshape(1, -1)

            X_copy[:, self.cols] = super(Scaler, self).transform(X_copy[:, self.cols], copy=copy)

            X_copy = X_copy.reshape(*orig_shape)

            return X_copy
        else:
            orig_shape = X.shape

            if len(orig_shape) < 2:
                X = X.reshape(1, -1)

            X = X.reshape(*orig_shape)

            return super(Scaler, self).transform(X)


transforms.register_builder("scaler", Scaler)
