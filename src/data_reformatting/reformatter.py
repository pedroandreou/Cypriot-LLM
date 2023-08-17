import os
from glob import glob
from itertools import chain

import nltk
from dotenv import find_dotenv, load_dotenv
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

load_dotenv(find_dotenv())


class BookReformatter:
    """
    Taken from
    https://gist.github.com/marrrcin/bcc115fbadf79eba9d9c8ca711da9e20
    """

    def __init__(self):
        directory_path = os.getenv("CLEANED_FILES_DIR_PATH")
        self.book_paths = list(glob(os.path.join(directory_path, "*.txt")))[
            :20
        ]  # Get first 10 for demonstration

    @staticmethod
    def flatten(iterable):
        return chain.from_iterable(iterable)

    def reformat_book(self, book_path):
        with open(book_path, "r", encoding="utf-8") as file:
            content = file.read()

        sentences = [s for s in sent_tokenize(content) if len(s) >= 16]
        windowed_sentences = []
        for snt in range(len(sentences)):
            windowed_sentences.append(" ".join(sentences[snt : snt + 8]))

        # print(f"Reformatted {len(windowed_sentences)} sentences from {book_path}.")

        return windowed_sentences

    def reformat_all_books(self):
        """
        The approach to split sentences every 10k samples was inspired by:
        https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
        """

        buffer = []
        BUFFER_SIZE = 10000  # Adjusted to 10,000
        file_count = 0

        for i, sentence in enumerate(
            self.flatten(self.reformat_book(bpath) for bpath in self.book_paths)
        ):
            buffer.append(sentence)

            if len(buffer) >= BUFFER_SIZE:
                with open(
                    f"reformatted_book_{file_count}.txt", "wt", encoding="utf-8"
                ) as file:
                    file.write("\n".join(buffer))
                    buffer.clear()
                    print(
                        f"Written to file: reformatted_book_{file_count}.txt with {i} sentences",
                        end="\r",
                    )
                file_count += 1

        # Write any remaining sentences that haven't reached the 10k threshold
        if buffer:
            with open(
                f"reformatted_book_{file_count}.txt", "wt", encoding="utf-8"
            ) as file:
                file.write("\n".join(buffer))
                print(
                    f"Written remaining sentences to: reformatted_book_{file_count}.txt"
                )


if __name__ == "__main__":
    reformatter = BookReformatter()
    reformatter.reformat_all_books()
