from collections import defaultdict
import numpy as np


class ClientReliabilityScorer:
    def __init__(self, window=5):
        self.window = window
        self.history = defaultdict(list)

    def update(self, client_id, accuracy, loss):
        self.history[client_id].append({"accuracy": accuracy, "loss": loss})
        if len(self.history[client_id]) > self.window:
            self.history[client_id].pop(0)

    def score(self, client_id):
        if not self.history[client_id]:
            return 0.5
        h = self.history[client_id]
        accs = [x["accuracy"] for x in h]
        losses = [x["loss"] for x in h]
        reliability = 0.6 * np.mean(accs) * (1 - min(np.std(accs), 0.3)) + 0.4 / (1 + np.mean(losses))
        return np.clip(reliability, 0, 1)

    def is_malicious(self, client_id, threshold=0.25):
        return self.score(client_id) < threshold


def adaptive_aggregate(weight_list, client_ids, scorer, client_sizes):
    scores = [scorer.score(cid) for cid in client_ids]
    filtered = [
        (weights, size, score)
        for weights, size, score, cid in zip(weight_list, client_sizes, scores, client_ids)
        if not scorer.is_malicious(cid)
    ]

    if not filtered:
        filtered = list(zip(weight_list, client_sizes, scores))

    weight_list, client_sizes, scores = zip(*filtered)
    agg_weights = [size / sum(client_sizes) * score for size, score in zip(client_sizes, scores)]
    agg_weights = [w / sum(agg_weights) for w in agg_weights]

    averaged = [sum(layer * agg_weights[i] for i, layer in enumerate(layers)) for layers in zip(*weight_list)]
    return averaged, agg_weights
