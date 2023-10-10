import globals
import langrank as lr
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os


def shap_within_tolerance(
    tolerance=1,
    LRLs=globals.I2L,
    modelfilename=os.path.join("pretrained", "MT", "lgbm_model_mt_all.txt"),
    X_file=os.path.join("tmp", "train_mt.csv"),
):
    n_candidates = 53
    LRLs = np.array([globals.L2I[LRL] for LRL in LRLs])
    # Load precomputed features
    X, _ = lr.load_svmlight_file(X_file)
    n_features = X.shape[-1]
    total_n_lang = 54
    X = X.toarray().reshape((total_n_lang, n_candidates, n_features))
    X = X[LRLs].reshape((LRLs.shape[0] * n_candidates, n_features))
    bst = lgb.Booster(model_file=modelfilename)
    # Predict with pred_contrib
    predict_contribs = bst.predict(X, pred_contrib=True)
    predict_contribs = predict_contribs.reshape(
        (LRLs.shape[0], n_candidates, n_features + 1)
    )
    # Compute actual predication scores per candidate
    predict_scores = predict_contribs.sum(-1)
    # Remove the bias and you are left with the SHAP values
    predict_contribs = predict_contribs[:, :, :-1]
    # Find k within tolerance
    k_mask = predict_scores - predict_scores.max(axis=1)[:, None] > -tolerance
    assert np.all(k_mask.sum(axis=1) > 1), "use a smaller LRLs or a bigger tolerance"
    candidates_within_tol = []
    predict_contribs_within_tol = []
    distances_within_tol = []
    order_by_contribs_within_tol = []
    for LRL, mask_LRL, predict_contribs_LRL in zip(LRLs, k_mask, predict_contribs):
        # Track the indices of these candidates
        candidates_indices = np.where(mask_LRL)[0]
        # Correct indices for diagonal
        candidates_indices[candidates_indices > LRL] += 1
        candidates = np.array(globals.I2L)[candidates_indices]
        candidates_within_tol.append(candidates)
        # Get features of candidates with score within tolerance
        features_candidates = predict_contribs_LRL[mask_LRL]
        predict_contribs_within_tol.append(features_candidates)
        # Compute distances between features of candidates with score within tolerance
        distances = np.linalg.norm(
            features_candidates[:, np.newaxis, :]
            - features_candidates[np.newaxis, :, :],
            axis=2,
        )
        distances_within_tol.append(distances)
        # find the ordering of these candidates based on the distances
        mask = np.tri(mask_LRL.sum(), k=-1, dtype=bool)
        distances_masked = distances.copy()
        distances_masked[mask] = -np.inf
        sorted_by_distance = np.argsort(distances_masked, axis=None)[::-1]
        sorted_row_indices, sorted_col_indices = np.unravel_index(
            sorted_by_distance, distances_masked.shape
        )
        # Create a 2D array of pairs of indices with duplicate (by symmetry) or diagonals
        pairs = np.column_stack((sorted_row_indices, sorted_col_indices))[
            : mask_LRL.sum()
        ]
        # Translate these indices into the corresponding languages
        order_by_contribs_within_tol.append(candidates[pairs])

    return (
        candidates_within_tol,
        predict_contribs_within_tol,
        distances_within_tol,
        order_by_contribs_within_tol,
    )


def heatmap_distances(LRLs, candidates_within_tol, distances_within_tol):
    for LRL, candidates, distance in zip(
        LRLs, candidates_within_tol, distances_within_tol
    ):
        plt.imshow(distance, cmap="viridis")
        plt.colorbar()
        plt.title(f"Heatmap source: {LRL}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.xticks(np.arange(distance.shape[0]), candidates)
        plt.yticks(np.arange(distance.shape[0]), candidates)
        plt.show()


def plot_shap_pairs(
    LRL,
    LRLs,
    candidates_within_tol,
    predict_contribs_within_tol,
    distances_within_tol,
    order_by_contribs_within_tol,
):
    # Get info for the specific LRL
    assert LRL in LRLs
    index = LRLs.index(LRL)
    language_pairs = order_by_contribs_within_tol[index]
    shap_values = predict_contribs_within_tol[index]
    candidates = candidates_within_tol[index]
    distances = distances_within_tol[index]

    C2S = {candidate: shap for candidate, shap in zip(candidates, shap_values)}
    C2I = {candidate: i for i, candidate in enumerate(candidates)}

    num_pairs = len(language_pairs)
    fig, axs = plt.subplots(num_pairs, figsize=(10, 6 * num_pairs))
    x_values = np.arange(shap_values.shape[-1])
    bar_width = 0.35

    # Create a color map based on unique languages
    unique_languages = list(set(np.ravel(language_pairs)))
    color_map = {
        lang: plt.cm.get_cmap("tab10")(i) for i, lang in enumerate(unique_languages)
    }

    # Plot bar plots for each pair
    for i, pair in enumerate(language_pairs):
        ax = axs[i]
        for lang_idx, lang in enumerate(pair):
            ax.bar(
                x_values + lang_idx * bar_width,
                C2S[lang],
                width=bar_width,
                label=lang,
                color=color_map[lang],
            )
        distance_pair = distances[C2I[pair[0]], [C2I[pair[1]]]].item()

        ax.set_xlabel("Features")
        ax.set_ylabel("SHAP Values")
        ax.set_title(
            f"SHAP Values for {pair[0]} and {pair[1]} with distance {distance_pair:0.4f}"
        )
        ax.set_xticks(x_values + bar_width / 2)
        ax.set_xticklabels(
            globals.FEATURES_NAMES_MT, rotation=45, ha="right", fontsize=10
        )
        ax.legend()

    plt.tight_layout()
    plt.show()


def shap_language_group(
    LRL,
    language_group,
    modelfilename=os.path.join("pretrained", "MT", "lgbm_model_mt_all.txt"),
    X_file=os.path.join("tmp", "train_mt.csv"),
):
    n_candidates = 53
    # LRLs = np.array([globals.L2I[LRL] for LRL in LRLs])
    X, _ = lr.load_svmlight_file(X_file)
    n_features = X.shape[-1]
    total_n_lang = 54
    X = X.toarray().reshape((total_n_lang, n_candidates, n_features))
    X = X[globals.L2I[LRL]]
    X = X.reshape((n_candidates, n_features))
    X = X[np.array([globals.L2I[language] for language in language_group])]
    bst = lgb.Booster(model_file=modelfilename)
    # Predict with pred_contrib
    predict_contribs = bst.predict(X, pred_contrib=True)
    predict_contribs = predict_contribs.reshape((len(language_group), n_features + 1))
    predict_contribs = predict_contribs[:, :-1]
    return predict_contribs


def plot_shap_groups(
    LRL,
    language_group,
    shap_values,
):
    plt.figure(figsize=(10, 6))
    x_values = np.arange(shap_values.shape[-1])
    bar_width = 0.25

    # Plot bar plots for each pair
    for lang_idx, (lang, shap) in enumerate(zip(language_group, shap_values)):
        plt.bar(
            x_values + lang_idx * bar_width,
            shap,
            width=bar_width,
            label=lang,
        )

    plt.xlabel("Features")
    plt.ylabel("SHAP Values")
    plt.title(f"SHAP Values for {language_group} with source {LRL}")
    plt.xticks(
        x_values + bar_width / 3,
        labels=globals.FEATURES_NAMES_MT,
        rotation=45,
        ha="right",
        fontsize=10,
    )
    plt.legend()

    plt.tight_layout()
    plt.show()


# # example:
# LRLs = [
#     "ur",
#     "eo",
#     "ms",
#     "az",
#     "ta",
#     "bn",
#     "kk",
#     "be",
#     "eu",
#     "bs",
# ]
# # Pick one of the languages from LRLs. Picking one is easier because then you get it's easier to tune theg tolerance
# # For example:
# LRLs = ["be"]
# TEDHEADER2L = dict((v, k) for k, v in globals.L2TEDHEADER.items())
# LRLs = [TEDHEADER2L[LRL] for LRL in LRLs]
# # So here a bit tuning of the tolerance parameter is needed.
# # Try to find a small tolerance that still gives enough candidates so there is a interesting difference difference in the distances.
# # That will differ per LRL
# # Yes, this tolerance measure is hard to qualitatively interpret.
# (
#     candidates_within_tol,
#     predict_contribs_within_tol,
#     distances_within_tol,
#     order_by_contribs_within_tol,
# ) = shap_within_tolerance(LRLs=LRLs, tolerance=0.7)

# # Leave this unchanged
# heatmap_distances(LRLs, candidates_within_tol, distances_within_tol)
# plot_shap_pairs(LRLs[0], LRLs, candidates_within_tol, predict_contribs_within_tol, distances_within_tol, order_by_contribs_within_tol)


"""DEPRECATED!:"""

# def feature_similarity_top_k(
#     k=10,
#     model_filename=os.path.join("pretrained", "MT", "lgbm_model_mt_all.txt"),
#     feature_importance_type="gain",
# ):
#     # Use the ground truth ranking
#     rank = eu.get_subset_rankings(globals.I2L)
#     # Don't consider the source language as candidate transfer language
#     # 2 * cols ensure that the diagonals have the worst ranking
#     rows, cols = rank.shape
#     np.fill_diagonal(rank, 2 * cols)
#     # Get the top-k transfer candidates
#     top_k = np.argsort(rank, axis=1)[:, :k]
#     # Load the precompute features
#     train_file = "tmp/train_mt.csv"
#     X_train, _ = lr.load_svmlight_file(train_file)
#     n_features = X_train.shape[-1]
#     features = X_train.toarray().reshape((rows, cols - 1, n_features))
#     # Standardize the features
#     features_shape = features.shape
#     features_reshaped = features.reshape(-1, 14)
#     mean = np.mean(features_reshaped, axis=0)
#     std = np.std(features_reshaped, axis=0)
#     features = ((features_reshaped - mean) / std).reshape(features_shape)
#     # Correct for index mismatch since the features don't have the source language==candidate transfer cases in the array
#     top_k_shifted = top_k.copy()
#     for i, ranking in enumerate(top_k_shifted):
#         ranking[ranking > i] -= 1
#     # Get the top k features for each source language
#     features_top_k = features[np.arange(rows)[:, None], top_k_shifted]
#     # Reweight the features according to the importance weight of the ranker
#     # We use their best checkpoint
#     feature_importance = (
#         lgb.Booster(model_file=model_filename)
#         .feature_importance(feature_importance_type)
#         .astype("float64")
#     )
#     feature_importance /= feature_importance.sum()
#     features_top_k *= feature_importance[None, :][None, :]
#     # Compute Euclidean distances
#     distances = np.linalg.norm(
#         features_top_k[:, :, np.newaxis, :] - features_top_k[:, np.newaxis, :, :],
#         axis=-1,
#     )
#     # Distance is symmetric and we don't care about the diagonals
#     mask = np.tri(k, k=-1, dtype=bool)
#     n_meaningfull_indices = mask.sum()
#     distances[:, mask] = -np.inf
#     feature_similarity = np.empty((rows, n_meaningfull_indices, 2), dtype=int)
#     for i, dist in enumerate(distances):
#         # Find the indices that would sort the distances in descending order
#         sorted_indices = np.argsort(dist, axis=None)[::-1]
#         # Calculate the corresponding row and column indices
#         sorted_row_indices, sorted_col_indices = np.unravel_index(
#             sorted_indices, dist.shape
#         )
#         # Create a 2D array of pairs of indices with duplicate (by symmetry) or diagonals
#         pairs = np.column_stack((sorted_row_indices, sorted_col_indices))[
#             :n_meaningfull_indices
#         ]
#         # Translate these indices into the corresponding language indices
#         feature_similarity[i] = top_k[i][pairs]
#     # (ranking by feature similarity score, the rankings by ground truth)
#     # TODO also add actual similarity score. Prob just make two function and sort only in the other one
#     return feature_similarity, top_k, distances


# feature_similarity, top_k, distances = feature_similarity_top_k()
# # np.array(globals.I2L)[feature_similarity]

# for i in range(5):
#     plt.imshow(distances[i], cmap='viridis')  # You can change the colormap as needed
#     plt.colorbar()  # Add a colorbar to the plot
#     plt.title(f"Heatmap source: {globals.I2L[i]}")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     lang_names = np.array(globals.I2L)[top_k[i]]
#     plt.xticks(np.arange(distances[i].shape[0]), lang_names)
#     plt.yticks(np.arange(distances[i].shape[0]), lang_names)
#     plt.show()
