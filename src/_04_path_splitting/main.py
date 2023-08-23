import os
from pathlib import Path

import typer
from sklearn.model_selection import train_test_split

from src.utils.common_utils import echo_with_color


def save_list_to_file(file_path, items_list):
    with open(file_path, "w") as f:
        f.write("\n".join(items_list))


def load_list_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()


class PathSplitter:
    def __init__(
        self,
        cleaned_files_dir_path: str,
        allpaths_file_path: str,
        trainpaths_file_path,
        testpaths_file_path,
    ):
        self.cleaned_files_dir_path = cleaned_files_dir_path

        self.allpaths_file_path = allpaths_file_path
        self.trainpaths_file_path = trainpaths_file_path
        self.testpaths_file_path = testpaths_file_path

        self.all_paths_list = []
        self.train_paths_list = []
        self.test_paths_list = []

    def split_paths(self):
        def extract_number(file_name: str):
            # Extract the numerical part from the file name
            return int(file_name.split("book_")[1].split(".txt")[0])

        # Get all the paths in the right order
        self.all_paths_list = [
            str(x) for x in Path(self.cleaned_files_dir_path).glob("*.txt")
        ]
        self.all_paths_list.sort(key=extract_number)

        # Split and shuffle the paths
        self.train_paths_list, self.test_paths_list = train_test_split(
            self.all_paths_list, test_size=0.2
        )

    def save_paths(self):
        echo_with_color("Saving the lists of file paths...", color="white")

        save_list_to_file(self.allpaths_file_path, self.all_paths_list)
        save_list_to_file(self.trainpaths_file_path, self.train_paths_list)
        save_list_to_file(self.testpaths_file_path, self.test_paths_list)

    def load_paths(self):
        try:
            echo_with_color("Loading the lists of file paths...", color="white")

            self.all_paths_list = load_list_from_file(self.allpaths_file_path)
            self.train_paths_list = load_list_from_file(self.trainpaths_file_path)
            self.test_paths_list = load_list_from_file(self.testpaths_file_path)

            return self.all_paths_list, self.train_paths_list, self.test_paths_list

        except FileNotFoundError:
            echo_with_color(
                "The file paths were not found.\nYou should run the script with --do_split_paths flag first.",
                color="white",
            )

            raise typer.Exit(code=1)  # Exits the script with exit code 1


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    reformatted_files_dir_path = os.path.normpath(
        os.path.join(curr_dir, "..", "_03_data_reformatting")
    )
    allpaths_file_path = os.path.join(curr_dir, "file_paths", "all_paths.txt")
    trainpaths_file_path = os.path.join(curr_dir, "file_paths", "train_paths.txt")
    testpaths_file_path = os.path.join(curr_dir, "file_paths", "test_paths.txt")

    path_splitter = PathSplitter(
        reformatted_files_dir_path,
        allpaths_file_path,
        trainpaths_file_path,
        testpaths_file_path,
    )
    path_splitter.split_paths()
    path_splitter.save_paths()


if __name__ == "__main__":

    main()
