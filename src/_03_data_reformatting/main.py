import os
from glob import glob
from itertools import chain

import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from src.utils.common_utils import echo_with_color

nltk.download("punkt")


class BookReformatter:
    """
    Taken from
    https://gist.github.com/marrrcin/bcc115fbadf79eba9d9c8ca711da9e20
    """

    def __init__(self, sliding_window_size):
        cleaned_files_dir_path = os.path.join(
            curr_dir,
            "..",
            "_01_data_preprocessing",
            "_04_csv_to_txt_conversion",
            "cleaned_files",
        )

        self.book_paths = list(glob(os.path.join(cleaned_files_dir_path, "*.txt")))
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

        # Calculate the total for tqdm
        total = sum(
            1
            for _ in self.flatten(
                self.reformat_book(bpath) for bpath in self.book_paths
            )
        )

        for i, sentence in enumerate(
            tqdm(
                self.flatten(self.reformat_book(bpath) for bpath in self.book_paths),
                total=total,  # provide the total count of sentences for accurate progress bar
                desc=f"Reformatting sentences using a sliding window of {self.sw}",
            )
        ):
            buffer.append(sentence)

            if len(buffer) >= BUFFER_SIZE:
                file_path = os.path.join(curr_dir, f"book_{file_count}.txt")
                with open(file_path, "wt", encoding="utf-8") as file:
                    file.write("\n".join(buffer))
                    buffer.clear()

                    echo_with_color(
                        f"\nWritten to file: book_{file_count}.txt with {i} sentences",
                        color="magenta",
                    )
                file_count += 1

        # Write any remaining sentences that haven't reached the 10k threshold
        if buffer:
            file_path = os.path.join(curr_dir, f"book_{file_count}.txt")
            with open(file_path, "wt", encoding="utf-8") as file:
                file.write("\n".join(buffer))

                echo_with_color(
                    f"Written remaining sentences to: book_{file_count}.txt",
                    color="magenta",
                )


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(sliding_window_size):
    reformatter = BookReformatter(sliding_window_size)
    reformatter.reformat_all_books()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script parameters.")

    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=8,
        help="Size of the sliding window for processing data.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(args.sliding_window_size)
