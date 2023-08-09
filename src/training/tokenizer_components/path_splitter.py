import os
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())


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

        self.train_paths_list = []
        self.test_paths_list = []

    def split_paths(self):
        all_paths = [str(x) for x in Path(self.cleaned_files_dir_path).glob("*.txt")]
        self.train_paths_list, self.test_paths_list = train_test_split(
            all_paths, test_size=0.2
        )

        return self.train_paths_list, self.test_paths_list

    def save_paths(self):
        typer.echo("Saving the lists of file paths...")

        with open(self.trainpaths_file_path, "w") as f:
            for item in self.train_paths_list:
                f.write("%s\n" % item)

        with open(self.trainpaths_file_path, "w") as f:
            for item in self.train_paths_list:
                f.write("%s\n" % item)

        with open(self.testpaths_file_path, "w") as f:
            for item in self.test_paths_list:
                f.write("%s\n" % item)

    def load_paths(self):
        try:
            typer.echo("Loading the file paths...")

            with open(self.trainpaths_file_path, "r") as f:
                self.train_paths_list = f.read().splitlines()

            with open(self.testpaths_file_path, "r") as f:
                self.test_paths_list = f.read().splitlines()

            return self.train_paths_list, self.test_paths_list

        except FileNotFoundError:
            typer.echo(
                f"train_file_paths.txt and test_file_paths.txt not found.\nYou should run the script with --should-split-train-test flag first."
            )

            raise typer.Exit()
