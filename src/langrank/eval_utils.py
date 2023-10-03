from scipy.stats import rankdata
import numpy as np
import os
import globals
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file


def get_subset_rankings(
    language_subset,
    dir=globals.GROUND_TRUTH_DIR,
    file_name=globals.ORIGINAL_PAPER_GROUND_TRUTH_FILENAME,
):
    assert len(language_subset) > 0, "give at least one language"
    language_subset_indices = np.array([globals.L2I[lang] for lang in language_subset])

    raw_file_path = os.path.join(dir, file_name)
    full_raw = np.genfromtxt(
        raw_file_path,
        delimiter=",",
        skip_header=1,
        usecols=range(1, len(globals.I2L) + 1),
        dtype=float,
    )[:-2]
    subset_raw = full_raw[language_subset_indices][:, language_subset_indices]
    subset_ordinal = subset_raw.shape[0] - rankdata(
        subset_raw, method="ordinal", axis=0
    )
    return subset_ordinal.T


# # basic check:
# import langrank as lr
# languages = ["aze", "ben", "fin"]
# # Change path to your needs
# path = "/home/job/Documents/DL4NLP/project/DLNLP_Project/src/langrank/sample-data/ted-train-fragment.orig."
# datasets = [path + lang for lang in languages]
# ranks = get_subset_rankings(languages)
# lr.prepare_train_file(datasets, languages, ranks)
# lr.train("tmp", "first_test_model")
