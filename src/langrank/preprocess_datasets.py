import globals
import os
import time
import sentencepiece as spm


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
    os.makedirs('data/dataset', exist_ok=True)

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
                # print(line)
                sentence = line.strip().split("\t")[column_index]

                # Check if the line has enough columns and does not contain "NULL"
                if "NULL" not in sentence:
                    # Write it to the output file
                    output_f.write(sentence + "\n")
        print(f"Done with {lang}. Now done {i} out of {len(globals.I2L)} languages.")


def preprocess_subword_datasets(unsegmented_file_format="data/dataset/ted-train.orig.",
    output_file_format="data/dataset/subword/ted-train.orig.spm8000.",
                                model_directory = 'tokenizer/train/',
                                file_type="train" # file type can be "train, dev, test and all"
):
    """ "
    Perhaps not most efficient with many reads, but this only needs to be computed once
    Possible TODOS:
    TODO: Why not also include the dev and test set?
    """
    # create the directory for subword files, checkpoint files
    os.makedirs("./data/dataset/subword", exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)
    # these langs have less vocabs on train files
    special_langs_size = {'ben': 6782, 'msa': 6456, 'tam': 6991, 'urd': 5789}

    # loop for languages
    for i, lang in enumerate(globals.L2TEDHEADER, 1):
        # the unsegmented file of one specific language
        file_path = unsegmented_file_format + lang
        # ignore if a specific languge has no unsegmented file
        if not os.path.isfile(file_path):
            print(f"{file_path} does not exist, ignore to segment for {lang} !")
            continue
        # settings for command of sentencepiece
        if file_type in ['train', 'dev', 'test']:
            vocab_size = special_langs_size[lang] if lang in special_langs_size.keys() else 8000
        else:
            vocab_size = 8000

        model_name = lang + '-' + str(vocab_size)
        model_format = model_directory + model_name
        model_path = model_format + '.model'
        # check if the checkpoint is there or not
        if not os.path.isfile(model_path):
            print(f"{model_path} is not found, requiring the tokenizer to segment words !")
            cmd = f'--input={file_path} --model_prefix={model_format} --vocab_size={vocab_size}'
            # train the sentencepiece
            print(f"{model_path} train start ... ")
            spm.SentencePieceTrainer.train(cmd)
            # to give a buffer for the checkpoint saved
            time.sleep(1)
            print(f"{model_path} train end ... ")

        # check if the checkpoint is trained or not
        if not os.path.isfile(model_path):
            raise Exception(f"{model_path} fails to train !!!")
        
        # load specific tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        # sp = spm.SentencePieceProcessor(model_file=model_path)
        # tokenizer handle the files sentence by sentence
        with open(file_path, "r") as input_f, open(
            output_file_format + lang, "w"
        ) as output_f:
            for line in input_f:
                sentence = line.strip()
                # split the sentence into pieces
                segmented_list = sp.encode_as_pieces(sentence)
                segmented_sentence = " ".join(segmented_list)
                # write the segmented tokens subword file
                output_f.write(segmented_sentence + "\n")
        print(f"Done with {lang}. Now done {i} out of {len(globals.I2L)} languages.")
        # in case the current tokenizer resued for the following language
        del sp

if __name__ == "__main__":
    #preprocess_datasets()
    preprocess_subword_datasets()
