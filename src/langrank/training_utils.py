from scipy.stats import rankdata
import numpy as np
import os
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file

# globals, hardcoded for the single language joint training from the paper
I2L = [
    "ara",
    "aze",
    "bel",
    "ben",
    "bos",
    "bul",
    "ces",
    "cmn",
    "dan",
    "deu",
    "ell",
    "epo",
    "est",
    "eus",
    "fas",
    "fin",
    "fra",
    "glg",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ind",
    "ita",
    "jpn",
    "kat",
    "kaz",
    "kor",
    "kur",
    "lit",
    "mar",
    "mkd",
    "mon",
    "msa",
    "mya",
    "nld",
    "nob",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "spa",
    "sqi",
    "srp",
    "swe",
    "tam",
    "tha",
    "tur",
    "ukr",
    "urd",
    "vie",
]
L2I = {l: i for i, l in enumerate(I2L)}
GROUND_TRUTH_DIR = "ground_truth_rankings"
ORIGINAL_PAPER_GROUND_TRUTH_FILENAME = (
    "LangRank Transfer Language Raw Data - MT Results.csv"
)
ORIGINAL_RANKER_HYPERPARAMS = {
    "boosting_type": "gbdt",
    "num_leaves": 16,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_child_samples": 5,
}


def get_subset_rankings(
    language_subset,
    dir=GROUND_TRUTH_DIR,
    file_name=ORIGINAL_PAPER_GROUND_TRUTH_FILENAME,
):
    assert len(language_subset) > 0, "give at least one language"
    language_subset_indices = np.array([L2I[lang] for lang in language_subset])

    raw_file_path = os.path.join(dir, file_name)
    full_raw = np.genfromtxt(
        raw_file_path,
        delimiter=",",
        skip_header=1,
        usecols=range(1, len(I2L) + 1),
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
