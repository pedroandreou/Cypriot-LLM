import regex


class PatternRemover:
    def __init__(self, text):
        self.text = text

        # Each pattern is a tuple where the first element is the regex pattern and the second element is the replacement string
        self.patterns_to_remove = [
            # doc 0
            ("_x000C_", ""),  # Remove the "_x000C_" string from the text
            # doc 1
            (
                r"^\d+\.\s*",
                "",
                "multiline",
            ),  # Remove numbers at the start of the line followed by a period
            (r"^\d+$", "", "multiline"),  # Remove lines that consist of only numbers
            (
                r"^\s*\S{1,2}\s*$",
                "",
                "multiline",
            ),  # Remove lines that consist only of one or two characters (with possible leading or trailing spaces)
            (
                r"^\(\w\)$",
                "",
                "multiline",
            ),  # Remove lines that consist only of an open parenthesis, any single character, and a close parenthesis
            # doc 2
            (r"\.{2,}", "."),  # Replace two or more consecutive periods with one
            (
                r"^\s*\S\s*$",
                "",
                "multiline",
            ),  # Remove lines that consist only of a single character (with possible leading or trailing spaces)
            (
                r"^\s*\d+—\d+\s*$",
                "",
                "multiline",
            ),  # Remove lines that consist only of <number>—<number> pattern (with possible leading or trailing spaces)
            (
                r"^.*\s\s=\s*[A-Za-z]+.*$",
                "",
                "multiline",
            ),  # Remove lines containing '  =' followed by one or more English letters
            (
                r"^[Α-Ωα-ω]\.[Α-Ωα-ω]\.$",
                "",
                "multiline",
            ),  # Remove lines that consist only of a Greek letter followed by a period, followed by another Greek letter and a period
            (r"·|\*", ""),  # Match either '·' or '*'
            (
                r"^—|—$",
                "",
                "multiline",
            ),  # Remove '—' at the beginning or end of the line
            (
                "^(?=.*[a-zA-Z])[a-zA-Z0-9\s.,;:!?]*$",
                "",
                "multiline",
            ),  # Remove lines containing only English characters, numbers and specified punctuation
            (r"ISBN.*\d+-\d+-\d+-\d+", ""),
            # doc 3
            (r"ΣΕΛΙΔΑ\s\d+", ""),
            (r"^-{3,}", "", "multiline"),
            (
                r"([Α-ω]\s--(?!\-))",
                "-",
            ),  # for exactly two hyphens, replace with one hyphen
            (
                r"([Α-ω])\s---+",
                r"\1.",
            ),  # Matches a Greek letter followed by a whitespace and three or more hyphens. Replaces it with the Greek letter followed by a period.
            (r"ΣΚΗΝΗ\s+\d+\w*?\s?(–|-)\s?", ""),
            (r"^ΣΚΗΝΗ\s+\d+\s*$", "", "multiline"),
            (r"^ΣΚΗΝΗ\s+\d+\w*?\s*", "", "multiline"),
            (
                r"(?:\((?:ΣΥΝΕΧΕΙΑ|ΑΜΕΣΗ ΣΥΝΕΧΕΙΑ ΜΕ|ΠΑΡΑΛΛΗΛΗ ΔΡΑΣΗ ΜΕ) ?ΣΚΗΝΗ ?\d*\)?)|(?:ΣΥΝΕΧΕΙΑ|Συνέχεια|Συνδεση με την) ΣΚΗΝΗΣ ?\d*",
                "",
                "ignorecase",
            ),
            ("ΣΚ\s{2,}", ""),
            # doc 4
            (r"(άα{2,})", "ά"),
            (r"(έε{2,})", "έ"),
            (r"(ήη{2,})", "ή"),
            (r"(ίι{2,})", "ί"),
            (r"(όο+)", "ό"),
            (r"(ύυ+)", "ύ"),
            (r"(ώω{2,})", "ώ"),
            (r"(\w+)�(\w+)", r"\1\2"),
            # # # remove � and connect surrounding characters
            # (r"(ι)�(τ)", r"\1\2"),
            # (r"(ζ)�(᾽)", r"\1\2"),
            # (r"(ζ)�(ί)", r"\1\2"),
            # (r"(ζ)�(ή)", r"\1\2"),
            # (r"(ζ)�(ες)", r"\1\2"),
            # (r"(ζ)�(εί)", r"\1\2"),
            # (r"(ζ)�(ελ)", r"\1\2"),
            # (r"(ζ)�(ι)", r"\1\2"),
            # (r"(ζ)�(αμ)", r"\1\2"),
            # (r"(ζ)�(οί)", r"\1\2"),
            # (r"(ξ)�(α)", r"\1\2"),
            # (r"(ξ)�(ι)", r"\1\2"),
            # (r"(Σ)�(α)", r"\1\2"),
            # # replace � with ι
            # (r"(ζ)�(ύ)", r"\1ι\2"),
            # (r"(ζ)�(υ)", r"\1ι\2"),
            # (r"(ζ)�(\')", r"\1ι\2"),
            # (r"(ζ)�(ε)", r"\1ι\2"),
            # (r"(ζ)�(α)", r"\1ι\2"),
            # (r"(ζ)�(οι)", r"\1ι\2"),
            # (r"(Σ)�(σ)", r"\1ι\2"),
            # (r"(Σ)�(ι)", r"\1ι\2"),
            # (r"(ψ)�(ι)", r"\1ι\2"),
            # (r"(ω)�(ν)", r"\1ι\2"),
            # doc 8
            (r"(\w)_(\w)", r"\1\2"),  # Remove underscore between two alphanumeric chars
            (r"_", ""),  # Remove any remaining underscore
            #
            # (r'(^[\p{Lu}]+)(?:…|\.{3,4})(\p{Lu}\p{Ll}+)', r'\1: \2', 'unicode'),
            # consecutive punctuation marks
            (
                r"(\?{2,})",
                "?",
            ),  # Replace sequences of 2 or more question marks with a single question mark
            (
                r"(;{2,})",
                ";",
            ),  # Replace sequences of 2 or more question marks with a single question mark
            (
                r"(\!{2,})",
                "!",
            ),  # Replace sequences of 2 or more exclamation marks with a single exclamation mark
            (r"(,{2,})", ","),
            (r"(\({2,})", "("),
            (r"(\){2,})", ")"),
            (r"(\[{2,})", "["),
            (r"(\-{2,})", ""),
            (r"(>{2,})", ""),
            (r"(<{2,})", ""),
            # whitespace before right parenthesis and whitespace after left parenthesis
            (r"\s\)", r")"),
            (r"\(\s", r"("),
            (r"\(\)", ""),  # Remove empty parenthesis
            (r"\(\s*\)", ""),  # Remove empty parenthesis
            # elipsis followed by a period
            (r"\s…\.\s", ""),
            (r"^([Α-Ω]+)…(\.)*(.+)", r"\1: \3", "multiline"),
            (r"^[Α-Ω]+…\.", "", "multiline"),
            # different cases of //
            (
                r"(\/\/)$",
                "",
                "multiline",
            ),  # Remove '//' if it is the last character sequence of a line
            (r"\/\/\s", " "),  # Remove '//' if there is a space on its right
            (
                r"\/\/(?!\s)",
                " ",
            ),  # Replace '//' with a space if there is not a space on its right
            (
                r"([Α-Ω]+) / ([Α-Ω]+)",
                r"\1 - \2",
                "",
            ),  # Replaces ' / ' between two Greek uppercase words with ' - '
            # social
            (r"www\.[a-zA-Z0-9-]+\.[a-zA-Z]+\S*", ""),  # urls
            (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", ""),  # email addresses
            (r"@\w+", ""),  # twitter-style usernames
            # specific phrases
            (r"Κ Ε Ι Μ Ε Ν Α Κ Υ Π Ρ Ι Α Κ Η Σ Λ Ο Γ Ο Τ Ε Χ Ν Ι Α Σ", ""),
            (r"Κ Ε ΙΜ Ε ΝΑ Κ Υ Π Ρ Ι Α Κ Η Σ Λ Ο Γ Ο Τ Ε Χ Ν Ι Α Σ", ""),
            (r"(\b[Α-Ω]\b\s){2,}", ""),
            (r"Μ ΙΧ ΑΛΗΣ Π ΙΕΡΗΣ", ""),
            # dots
            ("…", "."),
            (r"\.{4,}", ""),
            (r"\[…\]", ""),
            (r"(\…){2,}\.", ""),  # '…' occurring at least twice and ending with '.'
            (r"·|˙|°|© ", ""),
            (r";;", ";"),
            (r"\?|;,|;!", ";"),
            ("–", "-"),
            (r"\.!", "!"),
            ("@", "α"),
            (r"\s*\(\s*\u03BF\s*f\s*f\s*\)\s*", ""),  # (off)
            (r"\|\|", ""),
            (r"\[\.\]", ""),
            ("΄", "‘"),  # apostrophe before a word
            (r"\[\.\]", ""),
            (r";\.|\.;", ";"),
            (r"\s+;", ";"),
            (r"^-", "", "multiline"),
            ("§§", "Ενότητες "),
            ("§", "Ενότητα "),
            # .!.
            (
                r"(?<=[α-ωά-ώΑ-ΩΆ-Ώ]).!.",
                ".",
            ),  #  '.!.' with a period if there is a Greek character to the left
            (r"\.!\.", "."),  #  all other instances
            (r"\s!\s", ""),
            # whitespace before and after a period
            (r"((?:[Α-Ω]+)) \. ", r"\1:"),  # e.g. "ΜΕΡΟΠΗ . " --> "ΜΕΡΟΠΗ:"
            (
                r"(?<=[α-ωάέήίϊΐόύϋΰώκν]) \. (?=[α-ωάέήίϊΐόύϋΰώκν])",
                "",
                "multiline",
            ),  # κακόν . ναι
            (
                r"((?:[α-ωά-ώ]+)) \. ((?:[Α-Ω][α-ωΑ-Ω]*))",
                r"\1. \2",
            ),  # e.g. "καφέ . ΒΑΡΒΑΡΑ" --> "καφέ. ΒΑΡΒΑΡΑ"
            (
                r"^([Α-ΩΪΫ]+) \.(.*)$",
                r"\1: \2",
                "multiline",
            ),  # ΔΕΣΠΟΙΝΑ .ο --> ΔΕΣΠΟΙΝΑ: ο
            (r":\s\.\s", r": "),
            # Doctor Melas
            (r"\bΔ[ρΡ]\s*\.", "Δρ", "ignorecase"),
            ("ΔρΜΕΛΑΣ", "Δρ ΜΕΛΑΣ"),
            (r"Δρ ΜΕΛΑΣ\.{2,}", "Δρ ΜΕΛΑΣ: "),
            (r"(Δρ ΜΕΛΑΣ)\.([Α-Ωα-ωάέήίϊΐόύϋΰώΆΈΉΊΪΌΎΫΏ]+)", r"\1: \2"),
            (r"^Δρ ΜΕΛΑΣ\.?", "Δρ ΜΕΛΑΣ: "),
            # quotation marks
            ('[«»‘’"“”]', "'"),
            (
                r"'{2,}",
                "'",
            ),
            # pages
            (r"σ\.\s\d+,", ""),
            (r"σ\.\s\d+-\d+,", ""),
            # " . "
            (
                r"(?<=[α-ωάέήίϊΐόύϋΰώ]) \. (?=[α-ωάέήίϊΐόύϋΰώ])",
                "",
            ),  # . after a whitespace and before a lowercase Greek letter
            (
                r"([α-ω]+)\s([.;])\s([Α-Ω]+)",
                r"\1\2 \3",
            ),  # "ναι . ΜΑΡΟΥΛΛΑ." --> "ναι. ΜΑΡΟΥΛΛΑ."
            # Replace repeated sequences of two uppercase Greek words with the first Greek word in the sequence
            (r"\b\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} (π\.μ\.|μ\.μ\.)", ""),
            #
            (r":\)", ""),
            (r"\(:", ""),
            (r"\[([A-Z]+)\]", ""),  # Remove roman numerals presenting the paragraphs
            (r"\[([a-z]+)\]", ""),
            # rejoin words split by a hyphen at end-of-line, continuing with lowercase Greek letters on next line.
            (r"(-\n)([α-ωάέήίϊΐόύϋΰώ]+)", r"\2"),  # "αποβιβά-\nση"--> "αποβιβάση"
            # "ΠΑΝΙΚΟΣ Πε" --> "ΠΑΝΙΚΟΣ: Πε"
            (
                r"(^)([Α-ΩΪΫ]{2,})( [Α-ΩΪΫ][α-ωίϊΐόύϋΰώ]+)",
                r"\1\2:\3",
                "multiline",
            ),  # multiline does not work here
            # :.
            (r":\.", ":"),
            (r":\s\.", ": "),  # : .
            (r"^:\.$", ""),  # Removing ':.' if it is the only character(s) on the line
            (r":-\s*$", "."),  # Replaces ':-' at the end of a line with a period '.'
            (
                r"ΗΧΟΣ\s*:\s*[-\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]",
                "ΗΧΟΣ -",
            ),  # Replace ':.' with '-' when it comes after 'HXOΣ' either straight away or after a whitespace
            (
                r"PROPS\.?\s*:\s*[-\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]",
                "PROPS -",
            ),  # e.g. 'PROP :-', 'PROPS :-', 'PROPS. :-'
            (
                r"ΧΩΡΟΣ\s*:\s*[-\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]",
                "ΧΩΡΟΣ -",
            ),  # Same for 'ΧΩΡΟΣ'
            # (.)
            (r"\(\.\)", ""),
            # <punctuation> <.> <greek letter>
            (
                r"([;.])(\s)\.(\s)([α-ωΑ-Ω])",
                r"\1\2\4",
            ),  # 'καταφέρνω; . κάποτε.' --> 'καταφέρνω; κάποτε.'
            # <uppercase greek letter>:<greek letters>
            (r":([α-ωΑ-Ω])", r": \1"),  # ΑΝΘΟΥΣΑ:α; --> ΑΝΘΟΥΣΑ: α;
            # <start><uppercase greek letter>:<greek letters>
            (
                r"^([Α-Ω]{2,})\.([Α-Ω][α-ω]*)",
                r"\1. \2",
                "multiline",
            ),  # ΦΡΟΣΩ.Που --> ΦΡΟΣΩ. Που
            #'ο Αλέξαντρος ,η Μαρόυλλα' --> 'ο Αλέξαντρος, η Μαρόυλλα'
            # 'ο δοικητής , αρνιέται' --> 'ο δοικητής, αρνιέται'
            (
                r"([\u0370-\u03FF]+)\s*,\s*([\u0370-\u03FF]+\.?)([\u0370-\u03FF]+)",
                r"\1, \2 \3",
            ),
            # Greek consonant at the end of a word followed by a space and a Greek vowel
            (
                r"([βγδζθκλμνξπρσςτφχψ]) \b([αεηιουωάέήίόύώ])\b",
                r"\1\2",
            ),  # γύρ ω --> γύρω
            # whitespaces
            (
                r"\n{3,}",
                "\n\n",
            ),  # Replace 3 or more consecutive newline characters with just two
            (r"\s{2,}", " "),  # Replace 2 or more consecutive whitespaces by one
            # normalizing dots
            (r"\.{2,}", "..."),
            # Separate words that contain a dot but no space between them
            (
                r"([α-ωάέήίόύώϊϋΐΰ])\.([Α-ΩΆΈΉΊΪΌΎΏ])",
                r"\1. \2",
            ),  # "Χρήστο.Χρήστο." --> "Χρήστο. Χρήστο."
            # join 'ς' or 'ν' to the previous greek word if its last char is a vowel
            (
                r"(\b[α-ωΑ-Ωά-ώ]*[αειουωήύόάέώ]\b) (\b[νς]\b)",
                r"\1\2",
            ),  # κυρίε ς  --> κυρίες ||  # ότα ν --> όταν
            # remove parenthesis containing a single lowercase greek char that is part of a word
            (
                r"(?:(?<=\S)\(([α-ωΑ-Ω])\)(?=\s)|(?<=\s)\(([α-ωΑ-Ω])\)(?=\S)|(?<=\S)\(([α-ωΑ-Ω])\)(?=\S))",
                r"\1\2\3",
            ),  # ε(α)ρομένες --> εαρομένες
            # greek vowel after 'ς'
            (r"(ς)([αεηιοωυ])", r"ς \1"),
        ]

    def remove_patterns(self):
        for pattern_info in self.patterns_to_remove:
            pattern, replacement = pattern_info[
                :2
            ]  # First two elements are always pattern and replacement
            flag = (
                pattern_info[2] if len(pattern_info) > 2 else None
            )  # Flag is optional, could be not provided

            flag_value = 0  # default, no flag

            if flag == "multiline":
                flag_value = regex.MULTILINE
            elif flag == "unicode":
                flag_value = regex.UNICODE
            elif flag == "ignorecase":
                flag_value = regex.IGNORECASE

            self.text = regex.sub(pattern, replacement, self.text, flags=flag_value)

        return self.text
