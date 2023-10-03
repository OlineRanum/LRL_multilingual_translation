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


# preparing the file for training
def prepare_train_file(
    datasets, langs, rank, segmented_datasets=None, task="MT", tmp_dir="tmp"
):
    """
    dataset: [ted_aze, ted_tur, ted_ben]
    lang: [aze, tur, ben]
    rank: [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
    """
    num_langs = len(langs)
    REL_EXP_CUTOFF = num_langs - 1 - 9

    if not isinstance(rank, np.ndarray):
        rank = np.array(rank)
    BLEU_level = -rank + len(langs)
    rel_BLEU_level = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)

    features = {}
    for i, (ds, lang) in enumerate(zip(datasets, langs)):
        with open(ds, "r") as ds_f:
            lines = ds_f.readlines()
        seg_lines = None
        if segmented_datasets is not None:
            sds = segmented_datasets[i]
            with open(sds, "r") as sds_f:
                seg_lines = sds_f.readlines()
        features[lang] = prepare_new_dataset(
            lang=lang, dataset_source=lines, dataset_subword_source=seg_lines
        )
    uriel = uriel_distance_vec(langs)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    train_file = os.path.join(tmp_dir, "train_mt.csv")
    train_file_f = open(train_file, "w")
    train_size = os.path.join(tmp_dir, "train_mt_size.csv")
    train_size_f = open(train_size, "w")
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if i != j:
                uriel_features = [u[i, j] for u in uriel]
                distance_vector = distance_vec(
                    features[lang1], features[lang2], uriel_features, task
                )
                distance_vector = [
                    "{}:{}".format(i, d) for i, d in enumerate(distance_vector)
                ]
                line = " ".join([str(rel_BLEU_level[i, j])] + distance_vector)
                train_file_f.write(line + "\n")
        train_size_f.write("{}\n".format(num_langs - 1))
    train_file_f.close()
    train_size_f.close()
    print("Dump train file to {} ...".format(train_file_f))
    print("Dump train size file to {} ...".format(train_size_f))


# # basic check:
# import langrank as lr
# languages = ["aze", "ben", "fin"]
# # Change path to your needs
# path = "/home/job/Documents/DL4NLP/project/DLNLP_Project/src/langrank/sample-data/ted-train-fragment.orig."
# datasets = [path + lang for lang in languages]
# ranks = get_subset_rankings(languages)
# lr.prepare_train_file(datasets, languages, ranks)
# lr.train("tmp", "first_test_model")
