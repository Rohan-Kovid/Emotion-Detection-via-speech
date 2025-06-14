import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Model, load_model

class KerasSklearnHybrid(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_model_path, classifier):
        self.keras_model_path = keras_model_path
        self.classifier = classifier
        self._load_feature_extractor()

    def _load_feature_extractor(self):
        keras_model = load_model(self.keras_model_path)
        self.feature_extractor = Model(inputs=keras_model.input,
                                       outputs=keras_model.layers[-2].output)

    def fit(self, X, y):
        features = self.feature_extractor.predict(X)
        self.classifier.fit(features, y)
        return self

    def predict(self, X):
        features = self.feature_extractor.predict(X)
        return self.classifier.predict(features)

    def predict_proba(self, X):
        features = self.feature_extractor.predict(X)
        return self.classifier.predict_proba(features)
