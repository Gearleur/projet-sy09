from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, metric="euclidean"):
        # Le nombre de centres à apprendre sur le jeu de données de
        # départ
        self.n_clusters = n_clusters

        # La métrique utilisée pour calculer les distances entre les
        # centres et les exemples du jeu de données
        self.metric = metric

    def fit(self, X, y=None):
        # Apprentissage des centres et stockage dans l'attribut
        # `centers`.
        ...

        return self

    def transform(self, X):
        # Retourner les données transformées en utilisant les centres
        # disponibles avec l'attribut `centers`
        return ...
