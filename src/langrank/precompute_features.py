import globals
import eval_utils as eu
import langrank as lr

if __name__ == "__main__":
    datasets = [globals.UNSEGMENTED_FILE_FORMAT + lang for lang in globals.I2L]
    datasets_subwords = [globals.SEGMENTED_FILE_FORMAT + lang for lang in globals.I2L]
    rank = eu.get_subset_rankings(globals.I2L)
    lr.prepare_train_file(
        datasets,
        globals.I2L,
        rank,
        segmented_datasets=datasets_subwords,
        task="MT",
        tmp_dir="tmp",
    )
