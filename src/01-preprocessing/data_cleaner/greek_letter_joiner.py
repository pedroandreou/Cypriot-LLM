import re
import unicodedata


class GreekLetterJoiner:
    def __init__(self, text):
        self.text = text

    def reverse_text(self):
        self.text = self.text[::-1]
        return self

    def handle_single_letters(self):
        # Join single lowercase letters with the first Greek uppercase letter or a Greek lowercase word on their left
        # This regex operation is performed on the reversed string
        self.text = re.sub(
            r"(\b[α-ωάέήίόύώϊϋΐΰ]\b)\s+(?=\b[Α-ΩΆΈΉΊΪΌΎΏα-ωάέήίόύώϊϋΐΰ]\b)",
            r"\1",
            self.text,
        )
        return self

    def handle_uppercase(self):
        # Remove space after a Greek uppercase letter followed by Greek lowercase letters
        self.text = re.sub(
            r"(\b[Α-ΩΆΈΉΊΪΌΎΏ]\b)\s(?=[α-ωάέήίόύώϊϋΐΰ]\b)", r"\1", self.text
        )
        return self

    def handle_oti(self):
        # Handles 'ό, τ ι', 'Ό, τ ι' and 'ό, τι'/'Ό, τι' cases
        self.text = re.sub(r"(ό|Ό), τ ?ι", r"\1,τι", self.text)
        return self

    def combine_vowels(self):
        self.text = unicodedata.normalize("NFC", self.text)
        self.text = re.sub(
            r"([αεηιουωάέήίόύώ]) \b([αεηιουωάέήίόύώ])(?=\b)", r"\1\2", self.text
        )
        return self
