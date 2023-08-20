import os
from dataclasses import dataclass, field
from glob import glob
from itertools import chain

import nltk
from dotenv import find_dotenv, load_dotenv
from nltk.tokenize import sent_tokenize
from transformers import HfArgumentParser

nltk.download("punkt")

load_dotenv(find_dotenv())


class BookReformatter:
    """
    Taken from
    https://gist.github.com/marrrcin/bcc115fbadf79eba9d9c8ca711da9e20
    """

    def __init__(self, all_files_path, sliding_window_size):
        self.book_paths = list(glob(os.path.join(all_files_path, "*.txt")))
        self.sw = sliding_window_size

    @staticmethod
    def flatten(iterable):
        return chain.from_iterable(iterable)

    def reformat_book(self, book_path):
        with open(book_path, "r", encoding="utf-8") as file:
            content = file.read()

        sentences = [s for s in sent_tokenize(content) if len(s) >= 16]
        windowed_sentences = []
        for snt in range(len(sentences)):
            windowed_sentences.append(" ".join(sentences[snt : snt + self.sw]))

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
                with open(f"book_{file_count}.txt", "wt", encoding="utf-8") as file:
                    file.write("\n".join(buffer))
                    buffer.clear()
                    print(
                        f"Written to file: book_{file_count}.txt with {i} sentences",
                        end="\r",
                    )
                file_count += 1

        # Write any remaining sentences that haven't reached the 10k threshold
        if buffer:
            with open(f"book_{file_count}.txt", "wt", encoding="utf-8") as file:
                file.write("\n".join(buffer))
                print(f"Written remaining sentences to: book_{file_count}.txt")


@dataclass
class ScriptArguments:
    cleaned_files_dir_path: str = field(
        os.getenv("CLEANED_FILES_DIR_PATH"),
        metadata={"help": "The path where all the cleaned files are stored."},
    )

    sliding_window_size: int = field(
        default=8, metadata={"help": "Size of the sliding window for processing data."}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    reformatter = BookReformatter(
        script_args.cleaned_files_dir_path, script_args.sliding_window_size
    )
    reformatter.reformat_all_books()


if __name__ == "__main__":
    main()
