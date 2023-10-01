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