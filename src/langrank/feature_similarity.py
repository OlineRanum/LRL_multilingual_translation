import globals
import langrank as lr
import numpy as np
import eval_utils as eu


def feature_similarity_top_k(k=10):
    # Use the ground truth ranking
    rank = eu.get_subset_rankings(globals.I2L)
    # Don't consider the source language as candidate transfer language
    # 2 * cols ensure that the diagonals have the worst ranking
    rows, cols = rank.shape
    np.fill_diagonal(rank, 2 * cols)
    # Get the top-k transfer candidates
    top_k = np.argsort(rank, axis=1)[:, :k]
    # Load the precompute features
    train_file = "tmp/train_mt.csv"
    X_train, _ = lr.load_svmlight_file(train_file)
    n_features = X_train.shape[-1]
    features = X_train.toarray().reshape((rows, cols - 1, n_features))
    # Correct for index mismatch since the features don't have the source language==candidate transfer cases in the array
    top_k_shifted = top_k.copy()
    for i, ranking in enumerate(top_k_shifted):
        ranking[ranking > i] -= 1
    # Get the top k features for each source language
    features_top_k = features[np.arange(rows)[:, None], top_k_shifted]
    # Reweight the features according to the importance weight of the ranker
    # We use their checkpoint now and don't normalize because of possible underflow
    features_top_k *= np.array(globals.FEATURE_IMPORTANCE_BEST_RANKER)[None, :][None, :]
    # Compute Euclidean distances
    distances = np.linalg.norm(
        features_top_k[:, :, np.newaxis, :] - features_top_k[:, np.newaxis, :, :],
        axis=-1,
    )
    # Distance is symmetric and we don't care about the diagonals
    mask = np.tri(k, k=-1, dtype=bool)
    n_meaningfull_indices = mask.sum()
    distances[:, mask] = -np.inf
    feature_similarity = np.empty((rows, n_meaningfull_indices, 2), dtype=int)
    for i, dist in enumerate(distances):
        # Find the indices that would sort the distances in descending order
        sorted_indices = np.argsort(dist, axis=None)[::-1]
        # Calculate the corresponding row and column indices
        sorted_row_indices, sorted_col_indices = np.unravel_index(
            sorted_indices, dist.shape
        )
        # Create a 2D array of pairs of indices with duplicate (by symmetry) or diagonals
        pairs = np.column_stack((sorted_row_indices, sorted_col_indices))[
            :n_meaningfull_indices
        ]
        # Translate these indices into the corresponding language indices
        feature_similarity[i] = top_k[i][pairs]
    # (ranking by feature similarity score, the rankings by ground truth)
    # TODO also add actual similarity score. Prob just make two function and sort only in the other one
    return feature_similarity, top_k


# feature_similarity, top_k = feature_similarity_top_k()
# np.array(globals.I2L)[feature_similarity]
