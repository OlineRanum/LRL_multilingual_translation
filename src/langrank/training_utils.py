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
l2I = {l: i for i, l in enumerate(I2L)}
RAW_DIR = os.path.join("ground_truth_rankings", "raw")
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


def get_ranking_from_raw_score(
    dir=RAW_DIR,
    raw_scores_file_name=ORIGINAL_PAPER_GROUND_TRUTH_FILENAME,
):
    file_path = os.path.join(dir, raw_scores_file_name)
    ground_truth = np.genfromtxt(
        file_path,
        delimiter=",",
        skip_header=1,
        usecols=range(1, len(I2L) + 1),
        dtype=float,
    )[:-2]
    ordinal_ranking = ground_truth.shape[0] - rankdata(
        ground_truth, method="ordinal", axis=0
    )
    return ordinal_ranking


def train_ranker(
    train_file, train_size, output_model, rank_hyperparams=ORIGINAL_RANKER_HYPERPARAMS
):
    # train_file = os.path.join(tmp_dir, "train_mt.csv")
    # train_size = os.path.join(tmp_dir, "train_mt_size.csv")
    X_train, y_train = load_svmlight_file(train_file)
    model = lgb.LGBMRanker(**rank_hyperparams)
    model.fit(X_train, y_train, group=np.loadtxt(train_size))
    model.booster_.save_model(output_model)
