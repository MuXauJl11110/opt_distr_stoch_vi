from typing import Tuple

import networkx as nx
import numpy as np

from .compute_params import compute_lam


def gen_matrices_decentralized(
    num_matrices: int, l: int, d: int, mean: float, std: float, noise: float, seed=0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    num_matrices: int
        Number of generated matrices

    l: int
        Number of rows in each matrix

    d: int
        Dimension

    mean: float
        Mean of random normal distribution to generate matrix entries

    std: float
        Standard deviation of normal distribution to generate matrix entries

    noise: float
        Amplitude of random noise added to each matrix held by an agent

    Returns
    -------
    A: np.ndarray, shape (num_matrices * l, d)
        Stacked array of matrices

    b: np.ndarray, shape (num_summands * l,)
        Stacked array of vectors
    """

    np.random.seed(seed)
    A_one = mean + std * np.random.randn(l, d)
    A = np.tile(A_one.T, num_matrices).T
    A[l:] += noise * np.random.randn(l * (num_matrices - 1), d)

    b_one = mean + std * np.random.randn(l)
    b = np.tile(b_one, num_matrices)
    b[l:] += noise * np.random.randn(l * (num_matrices - 1))

    return A, b


def line_adj_mat(n: int):
    """
    Adjacency matrix of a line graph over n nodes

    Parameters
    ----------
    n: int
        Number of nodes
    """

    mat = np.zeros((n, n), dtype=np.int32)
    ids = np.arange(n)
    mat[ids[:-1], ids[1:]] = 1
    mat[ids[1:], ids[:-1]] = 1
    return mat


def ring_adj_mat(n: int):
    """
    Adjacency matrix of a ring graph over n nodes

    Parameters
    ----------
    n: int
        Number of nodes
    """

    mat = line_adj_mat(n)
    mat[0, n - 1] = 1
    mat[n - 1, 0] = 1
    return mat


def grid_adj_mat(n: int, m: int):
    """
    Adjacency matrix of a rectangle grid graph over n x m nodes

    Parameters
    ----------
    n: int
        Vertical size of grid

    m: int
        Horizontal size of grid

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.generators.grid_2d_graph(n, m)
    return np.array(nx.linalg.graphmatrix.adjacency_matrix(graph).todense()).astype(np.float64)


def star_adj_mat(n: int):
    """
    Adjacency matrix of a start graph over n nodes (1 center and n - 1 leaves)

    Parameters
    ----------
    n: int
        Number of vertices

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.generators.star_graph(n - 1)
    return np.array(nx.linalg.graphmatrix.adjacency_matrix(graph).todense()).astype(np.float64)


def ring_gos_mat(n: int):
    """
    Gossip matrix of a ring graph over n nodes

    Parameters
    ----------
    n: int
        Number of nodes

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.cycle_graph(n)
    L = nx.linalg.laplacianmatrix.laplacian_matrix(graph).toarray().astype(np.float64)
    return L / compute_lam(L)[1]


def grid_gos_mat(n: int, m: int):
    """
    Gossip matrix of a rectangle grid graph over n x m nodes

    Parameters
    ----------
    n: int
        Vertical size of grid

    m: int
        Horizontal size of grid

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.generators.grid_2d_graph(n, m)
    L = nx.linalg.laplacianmatrix.laplacian_matrix(graph).toarray().astype(np.float64)
    return L / compute_lam(L)[1]


def star_gos_mat(n: int):
    """
    Gossip matrix of a star graph over n nodes (1 center and n - 1 leaves)

    Parameters
    ----------
    n: int
        Number of vertices

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.generators.star_graph(n - 1)
    L = nx.linalg.laplacianmatrix.laplacian_matrix(graph).toarray().astype(np.float64)
    return L / compute_lam(L)[1]


def metropolis_weights(adj_mat: np.ndarray):
    """
    Computes Metropolis weights for a graph with a given adjacency matrix

    Parameters
    ----------
    adj_mat: np.ndarray
        Adjacency matrix
    """

    weights = adj_mat / (1 + np.maximum(adj_mat.sum(1, keepdims=True), adj_mat.sum(0, keepdims=True)))
    ids = np.arange(adj_mat.shape[0])
    weights[ids, ids] = 1 - np.sum(weights, axis=0)
    return weights
