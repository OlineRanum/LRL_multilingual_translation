import globals


def preprocess_datasets(
    raw_ted_file="../preprocess/raw_ted_data/all_talks_train.tsv",
    output_file_format="data/dataset/ted-train.orig.",
):
    """ "
    Perhaps not most efficient with many reads, but this only needs to be computed once
    Possible TODOS:
    TODO: Why not also include the dev and test set?
    TODO: "if "NULL" not in sentence" may be getting some false negatives. Probably no big deal
    """
    for i, lang in enumerate(globals.L2TEDHEADER, 1):
        # Specify the column you want to extract (0-based index)
        column_index = globals.TEDHEADER2I[globals.L2TEDHEADER[lang]]

        # Read the input TSV file, extract the desired column, and filter out lines containing "NULL"
        with open(raw_ted_file, "r") as input_f, open(
            output_file_format + lang, "w"
        ) as output_f:
            # Skip the header
            next(input_f)
            for line in input_f:
                # Split the line by tabs (assuming it's a TSV file)
                sentence = line.strip().split("\t")[column_index]

                # Check if the line has enough columns and does not contain "NULL"
                if "NULL" not in sentence:
                    # Write it to the output file
                    output_f.write(sentence + "\n")
        print(f"Done with {lang}. Now done {i} out of {len(globals.I2L)} languages.")


def preprocess_subword_datasets():
    """
    TODO
    """
    pass


if __name__ == "__main__":
    preprocess_datasets()
