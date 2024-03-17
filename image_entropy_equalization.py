from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils._param_validation import StrOptions, Options


class ImageEntropyEqualization(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "method": [StrOptions({"entropy", "class", "all"})],
        "entropy_value": [Options(Integral, {2, 3, 4, 5, 6, 7, 8, 9}), None],
        "label_value": [Options(Integral, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), None]
    }

    def __init__(self, *, method: str = None, entropy_value: int = None, label_value: int = None):
        self.pixel_set_ = []
        self.method = method
        self.entropy_value = entropy_value
        self.label_value = label_value

    def fit(self, X, y=None):
        if self.method == "entropy":
            if self.entropy_value is None:
                raise ValueError("entropy_value is None, please set the entropy_value")
            self._fit_entropy(X, self.entropy_value)
        elif self.method == "class":
            if self.label_value is None:
                raise ValueError("label_value is None, please set the label_value")
            self._fit_class(X, y, self.label_value)
        elif self.method == "all":
            self._fit_all(X)

        return self

    def _fit_entropy(self, X, entropy_value: int):
        train = X
        entropy = entropy_value
        area = 2 ** entropy
        Q = int(train.shape[1] * train.shape[2] / area)
        mod = train.shape[1] * train.shape[2] % area
        pixel_set = []
        for n in range(area):
            if n < mod:
                for i in range(Q + 1):
                    pixel_set.append(n)
            else:
                for i in range(Q):
                    pixel_set.append(n)
        # side effect set the pixel_set
        self.pixel_set_ = pixel_set

    def _fit_class(self, X, labels, label_value):
        train = X
        x_train = train[labels == label_value]
        pixel_list = dict()
        for n in range(256):
            pixel_list[n] = 0
        for data in x_train:
            for raw in data:
                for pixel in raw:
                    pixel_list[pixel] += 1
        hist_list = dict()
        for num in pixel_list:
            hist_list[num] = pixel_list[num] / len(x_train)
        hist_num = dict()
        residual = 0
        for num in hist_list:
            hist_num[num] = int(np.round(hist_list[num] + residual, 0))
            residual = hist_list[num] + residual - int(np.round(hist_list[num] + residual, 0))

        pixel_set = []
        for num in hist_num:
            repeat = hist_num[num]
            for n in range(repeat):
                pixel_set.append(num)

        # side effect set the pixel_set
        self.pixel_set_ = pixel_set

    def _fit_all(self, X):
        train = X
        x_train = train
        pixel_list = dict()
        for n in range(256):
            pixel_list[n] = 0
        for data in x_train:
            for raw in data:
                for pixel in raw:
                    pixel_list[pixel] += 1
        hist_list = dict()
        for num in pixel_list:
            hist_list[num] = pixel_list[num] / len(x_train)
        hist_num = dict()
        residual = 0
        for num in hist_list:
            hist_num[num] = int(np.round(hist_list[num] + residual, 0))
            residual = hist_list[num] + residual - int(np.round(hist_list[num] + residual, 0))

        pixel_set = []
        for num in hist_num:
            repeat = hist_num[num]
            for n in range(repeat):
                pixel_set.append(num)

        # side effect set the pixel_set
        self.pixel_set_ = pixel_set

    def transform(self, X):
        x_test = X
        pixel_set = self.pixel_set_
        equalized_test = np.zeros([x_test.shape[0], x_test.shape[1] * x_test.shape[2]])
        a = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
        ind = np.argsort(a, axis=1, kind="mergesort")
        ind2 = np.argsort(ind, axis=1)
        for i in range(len(pixel_set)):
            equalized_test[ind2 == i] = int(pixel_set[i])
        equalized_test = equalized_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

        return equalized_test
