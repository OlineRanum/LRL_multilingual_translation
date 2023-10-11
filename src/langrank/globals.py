import os

# Based on the ordering from https://docs.google.com/spreadsheets/d/1yo9Zlnk_oMRshZeUMCHQmztnuC6VOCnP0wsgD7adRYQ/
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
L2full_name = {
    "ara": "Arabic",
    "aze": "Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "ces": "Czech",
    "cmn": "Chinese",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "epo": "Esperanto",
    "est": "Estonian",
    "eus": "Basque",
    "fas": "Persian",
    "fin": "Finnish",
    "fra": "French",
    "glg": "Galician",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kor": "Korean",
    "kur": "Kurdish",
    "lit": "Lithuanian",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mon": "Mongolian",
    "msa": "Malay",
    "mya": "Burmese",
    "nld": "Dutch",
    "nob": "Norwegian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "spa": "Spanish",
    "sqi": "Albanian",
    "srp": "Serbian",
    "swe": "Swedish",
    "tam": "Tamil",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "vie": "Vietnamese",
}
# Based on the ordering in the header of the raw TED file
I2TEDHEADER = [
    "talk_name",
    "en",
    "es",
    "pt-br",
    "fr",
    "ru",
    "he",
    "ar",
    "ko",
    "zh-cn",
    "it",
    "ja",
    "zh-tw",
    "nl",
    "ro",
    "tr",
    "de",
    "vi",
    "pl",
    "pt",
    "bg",
    "el",
    "fa",
    "sr",
    "hu",
    "hr",
    "uk",
    "cs",
    "id",
    "th",
    "sv",
    "sk",
    "sq",
    "lt",
    "da",
    "calv",
    "my",
    "sl",
    "mk",
    "fr-ca",
    "fi",
    "hy",
    "hi",
    "nb",
    "ka",
    "mn",
    "et",
    "ku",
    "gl",
    "mr",
    "zh",
    "ur",
    "eo",
    "ms",
    "az",
    "ta",
    "bn",
    "kk",
    "be",
    "eu",
    "bs",
]
TEDHEADER2I = {l: i for i, l in enumerate(I2TEDHEADER)}
L2TEDHEADER = {
    "ara": "ar",  # Arabic
    "aze": "az",  # Azerbaijani
    "bel": "be",  # Belarusian
    "ben": "bn",  # Bengali
    "bos": "bs",  # Bosnian
    "bul": "bg",  # Bulgarian
    "ces": "cs",  # Czech
    "cmn": "zh-cn",  # Chinese (Simplified)
    "dan": "da",  # Danish
    "deu": "de",  # German
    "ell": "el",  # Greek
    "epo": "eo",  # Esperanto
    "est": "et",  # Estonian
    "eus": "eu",  # Basque
    "fas": "fa",  # Persian (Farsi)
    "fin": "fi",  # Finnish
    "fra": "fr",  # French
    "glg": "gl",  # Galician
    "heb": "he",  # Hebrew
    "hin": "hi",  # Hindi
    "hrv": "hr",  # Croatian
    "hun": "hu",  # Hungarian
    "hye": "hy",  # Armenian
    "ind": "id",  # Indonesian
    "ita": "it",  # Italian
    "jpn": "ja",  # Japanese
    "kat": "ka",  # Georgian
    "kaz": "kk",  # Kazakh
    "kor": "ko",  # Korean
    "kur": "ku",  # Kurdish
    "lit": "lt",  # Lithuanian
    "mar": "mr",  # Marathi
    "mkd": "mk",  # Macedonian
    "mon": "mn",  # Mongolian
    "msa": "ms",  # Malay
    "mya": "my",  # Burmese
    "nld": "nl",  # Dutch
    "nob": "nb",  # Norwegian (Bokmål)
    "pol": "pl",  # Polish
    "por": "pt",  # Portuguese
    "ron": "ro",  # Romanian
    "rus": "ru",  # Russian
    "slk": "sk",  # Slovak
    "slv": "sl",  # Slovenian
    "spa": "es",  # Spanish
    "sqi": "sq",  # Albanian
    "srp": "sr",  # Serbian
    "swe": "sv",  # Swedish
    "tam": "ta",  # Tamil
    "tha": "th",  # Thai
    "tur": "tr",  # Turkish
    "ukr": "uk",  # Ukrainian
    "urd": "ur",  # Urdu
    "vie": "vi",  # Vietnamese
}
FEATURES_NAMES_MT = [
    "Overlap word-level",
    "Overlap subword-level",
    "Transfer lang dataset size",
    "Target lang dataset size",
    "Transfer over target size ratio",
    "Transfer lang TTR",
    "Target lang TTR",
    "Transfer target TTR distance",
    "GENETIC",
    "SYNTACTIC",
    "FEATURAL",
    "PHONOLOGICAL",
    "INVENTORY",
    "GEOGRAPHIC",
]

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
RAW_TED_FILE = os.path.join("..", "preprocess", "raw_ted_data", "all_talks_train.tsv")
UNSEGMENTED_FILE_FORMAT = os.path.join("data", "dataset", "ted-train.orig.")
SEGMENTED_FILE_FORMAT = os.path.join(
    "data", "dataset", "subword", "ted-train.orig.spm8000."
)
TOKENIZER_CP_DIR = os.path.join("tokenizer", "train/")
