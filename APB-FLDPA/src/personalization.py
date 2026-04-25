import numpy as np
from sklearn.cluster import KMeans


class PersonalizedFL:
    def __init__(self, k=2):
        self.k = k
        self.clusters = {}

    def cluster(self, client_stats):
        features = [np.concatenate([s["mean"], s["std"]]) for s in client_stats.values()]
        labels = KMeans(self.k, random_state=42, n_init="auto").fit_predict(features)
        self.clusters = dict(zip(client_stats.keys(), labels))
        return self.clusters

    def personalize(self, client_id, global_weights, local_weights, alpha=0.15):
        return [alpha * local + (1 - alpha) * global_w for global_w, local in zip(global_weights, local_weights)]
