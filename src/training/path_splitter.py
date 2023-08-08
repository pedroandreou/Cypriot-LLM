from pathlib import Path

import typer
from sklearn.model_selection import train_test_split


class PathSplitter:
    def __init__(self, cleaned_files_dir_path):
        self.cleaned_files_dir_path = cleaned_files_dir_path
        self.train_file_paths = []
        self.test_file_paths = []

    def split_paths(self):
        all_paths = [str(x) for x in Path(self.cleaned_files_dir_path).glob("*.txt")]
        self.train_file_paths, self.test_file_paths = train_test_split(
            all_paths, test_size=0.2
        )

        return self.train_file_paths, self.test_file_paths

    def save_paths(self):
        typer.echo("Saving the lists of file paths...")

        with open("training_testing_file_paths/train_file_paths.txt", "w") as f:
            for item in self.train_file_paths:
                f.write("%s\n" % item)

        with open("training_testing_file_paths/test_file_paths.txt", "w") as f:
            for item in self.test_file_paths:
                f.write("%s\n" % item)

    def load_paths(self):
        try:
            typer.echo("Loading the file paths...")

            with open("training_testing_file_paths/train_file_paths.txt", "r") as f:
                self.train_file_paths = f.read().splitlines()
                print("the train file paths are: ", self.train_file_paths)

            with open("training_testing_file_paths/test_file_paths.txt", "r") as f:
                self.test_file_paths = f.read().splitlines()
                print("the test file paths are: ", self.test_file_paths)

        except FileNotFoundError:
            typer.echo(
                f"train_file_paths.txt and test_file_paths.txt not found.\nYou should run the script with --should-split-train-test flag first."
            )

            raise typer.Exit()
