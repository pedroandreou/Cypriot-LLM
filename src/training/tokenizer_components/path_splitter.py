import os
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())


def save_list_to_file(file_path, items_list):
    with open(file_path, "w") as f:
        f.write("\n".join(items_list))


def load_list_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()


class PathSplitter:
    def __init__(
        self,
        cleaned_files_dir_path: str = os.getenv("CLEANED_FILES_DIR_PATH"),
        allpaths_file_path: str = os.getenv("ALLPATHS_FILE_PATH"),
        trainpaths_file_path: str = os.getenv("TRAINPATHS_FILE_PATH"),
        testpaths_file_path: str = os.getenv("TESTPATHS_FILE_PATH"),
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
            return int(file_name.split("text_file")[1].split(".txt")[0])

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
        typer.echo("Saving the lists of file paths...")

        save_list_to_file(self.allpaths_file_path, self.all_paths_list)
        save_list_to_file(self.trainpaths_file_path, self.train_paths_list)
        save_list_to_file(self.testpaths_file_path, self.test_paths_list)

    def load_paths(self):
        try:
            typer.echo("Loading the file paths...")

            self.all_paths_list = load_list_from_file(self.allpaths_file_path)
            self.train_paths_list = load_list_from_file(self.trainpaths_file_path)
            self.test_paths_list = load_list_from_file(self.testpaths_file_path)

        except FileNotFoundError:
            typer.echo(
                f"The file paths were not found.\nYou should run the script with --should-split-train-test flag first."
            )

            raise typer.Exit()

    def get_paths(self):
        return self.all_paths_list, self.train_paths_list, self.test_paths_list
