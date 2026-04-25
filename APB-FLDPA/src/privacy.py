import numpy as np


class DifferentialPrivacy:
    def __init__(self, eps=10.0, delta=1e-5, clip=3.0):
        self.eps = eps
        self.delta = delta
        self.clip = clip
        self.noise = (clip * np.sqrt(2 * np.log(1.25 / delta))) / eps

    def privatize_weights(self, weights):
        clipped = []
        for w in weights:
            norm = np.linalg.norm(w)
            factor = min(1.0, self.clip / norm) if norm > 0 else 1.0
            clipped.append(w * factor)
        return [w + np.random.normal(0, self.noise, w.shape) for w in clipped]
