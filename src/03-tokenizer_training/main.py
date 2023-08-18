import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from transformers import HfArgumentParser

load_dotenv(find_dotenv())

from path_splitter import PathSplitter
from tokenizer import TokenizerWrapper


@dataclass
class ScriptArguments:
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    do_split_paths: bool = field(default=False)
    cleaned_files_dir_path: str = field(default=os.getenv("CLEANED_FILES_DIR_PATH"))
    allpaths_file_path: str = field(default="./file_paths/all_paths.txt")
    trainpaths_file_path: str = field(default="./file_paths/train_paths.txt")
    testpaths_file_path: str = field(default="./file_paths/test_paths.txt")

    do_train_tokenizer: bool = field(default=False)
    tokenizer_dir_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to where the tokenizer should be exported."},
    )
    do_push_tokenizer_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable pushing tokenizer to hub."}
    )

    def __post_init__(self):
        if self.tokenizer_dir_path is None:
            self.tokenizer_dir_path = f"./trained_tokenizer_bundle/cy{self.model_type}"


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.do_split_paths:
        print("Splitting all paths to train and test path sets...")

        path_splitter = PathSplitter(
            script_args.cleaned_files_dir_path,
            script_args.allpaths_file_path,
            script_args.trainpaths_file_path,
            script_args.testpaths_file_path,
        )
        path_splitter.split_paths()
        path_splitter.save_paths()

    else:
        print(
            "Skipping the split of all paths...\nWill try to load the train and test path sets from the files."
        )

        path_splitter = PathSplitter(
            script_args.cleaned_files_dir_path,
            script_args.allpaths_file_path,
            script_args.trainpaths_file_path,
            script_args.testpaths_file_path,
        )
        path_splitter.load_paths()

    # Get paths
    # Train and Test paths will be used for training and testing the model
    all_paths_list, _, _ = path_splitter.get_paths()

    if script_args.do_train_tokenizer:
        print("Training a tokenizer from scratch...")

        TokenizerWrapper(
            filepaths=all_paths_list,
            tokenizer_path=script_args.tokenizer_dir_path,
            model_type=script_args.model_type,
        )

    else:
        print("Skipping the training of a tokenizer...")


if __name__ == "__main__":
    main()
