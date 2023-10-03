import globals
import eval_utils as eu
import langrank as lr

if __name__ == "__main__":
    format_ted_file = "data/dataset/ted-train.orig."
    datasets = [format_ted_file + lang for lang in globals.I2L]
    rank = eu.get_subset_rankings(globals.I2L)
    lr.prepare_train_file(
        datasets, globals.I2L, rank, segmented_datasets=None, task="MT", tmp_dir="tmp"
    )
