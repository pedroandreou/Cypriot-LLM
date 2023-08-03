import unicodedata


class DiacriticRemover:
    def __init__(self, text):
        self.text = text

    def remove_diacritic_generic(self, diacritic_code):
        """
        A generic method to remove a certain type of diacritic from the given text.

        Parameters:
        diacritic_code (str): The Unicode code point of the diacritic to be removed.

        Returns:
        self: The instance of the class with updated text.
        """
        normalized_text = unicodedata.normalize("NFD", self.text)
        cleaned_text = normalized_text.replace(diacritic_code, "")
        self.text = unicodedata.normalize("NFC", cleaned_text)

        return self

    def remove_caron(self):
        """
        Remove caron diacritic (ˇ) from the given text.
        """
        caron_combining_code = "\u030C"
        self.remove_diacritic_generic(caron_combining_code)

        return self

    def remove_breve(self):
        """
        Remove breve diacritic (˘) from the given text.
        """
        breve_combining_code = "\u0306"
        self.remove_diacritic_generic(breve_combining_code)

        return self

    def remove_low_acute(self):
        """
        Remove low acute diacritic (ˏ) from the given text.
        """
        low_acute_code = "\u02CF"
        self.text = self.text.replace(low_acute_code, "")

        return self

    def remove_diaeresis(self):
        """
        Remove diaeresis diacritic (¨) from the given text.
        """
        diaeresis_code = "\u0308"
        self.remove_diacritic_generic(diaeresis_code)

        return self
