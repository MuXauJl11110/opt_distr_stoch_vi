from typing import Tuple

import numpy as np
from joblib import Memory
from sklearn.datasets import fetch_openml, load_digits


def normalize(array: np.ndarray, eps: float = 0.00001):
    array[array == 0] = eps
    array /= np.absolute(array).sum()
    return array


def get_gaussian(d: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    def gaussian(mu, sigma, z):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((z - mu) ** 2) / (2 * sigma ** 2))

    z = np.linspace(-10, 10, d)
    gaus = np.zeros((K, d))
    mu = np.random.uniform(-5, 5, K)
    sigma = np.random.uniform(0.8, 1.8, K)

    for i in range(K):
        gaus[i] = gaussian(mu[i], sigma[i], z)
        gaus[i] = normalize(gaus[i])

    barmu = np.sum(mu) / len(mu)
    barsigma = (np.sum(np.sqrt(sigma)) / len(sigma)) ** 2
    bar_true = normalize(gaussian(barmu, barsigma, z))

    return gaus, bar_true, z


def load_mnist64(target_digit: int):
    digits = load_digits()
    normalized_digits = []

    for i, digit in enumerate(digits.images):
        if digits.target[i] == target_digit:
            normalized_digits.append(normalize(digit))

    return np.array(normalized_digits)


def load_mnist784(target_digit: int):
    memory = Memory("./mnist784_cache")
    fetch_openml_cached = memory.cache(fetch_openml)
    mndata = fetch_openml_cached("mnist_784", version=1, as_frame=False)
    images, labels = mndata["data"], mndata["target"]
    normalized_digits = []

    for digit, label in zip(images, map(int, labels)):
        if label == target_digit:
            digit = np.array(digit)
            normalized_digits.append(normalize(digit))

    return np.array(normalized_digits)
