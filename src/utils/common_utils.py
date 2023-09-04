import os

import torch
import typer


def get_new_subdirectory_path(base_dir_path, artifact_type):
    # Get a list of existing directories
    existing_dirs = [
        d
        for d in os.listdir(base_dir_path)
        if os.path.isdir(os.path.join(base_dir_path, d)) and d.startswith(f"tokenizer")
    ]

    # Find the next available counter
    count = 1
    while f"{artifact_type}_{count}" in existing_dirs:
        count += 1

    subdirectory = f"{artifact_type}_{count}"
    full_subdirectory_path = os.path.join(base_dir_path, subdirectory)

    # Create the directory
    os.mkdir(full_subdirectory_path)

    return full_subdirectory_path


def save_dataset(curr_dir, dataset, base_path, sub_dir, key):
    filename = os.path.join(curr_dir, base_path, f"{sub_dir}_{key}_dataset.pth")
    torch.save(dataset, filename)


def echo_with_color(text: str, color: str = "bright_cyan"):
    """Echos a text message with the specified color."""

    color_mapping = {
        "black": typer.colors.BLACK,
        "red": typer.colors.RED,
        "green": typer.colors.GREEN,
        "yellow": typer.colors.YELLOW,
        "blue": typer.colors.BLUE,
        "magenta": typer.colors.MAGENTA,
        "cyan": typer.colors.CYAN,
        "white": typer.colors.WHITE,
        "bright_black": typer.colors.BRIGHT_BLACK,
        "bright_red": typer.colors.BRIGHT_RED,
        "bright_green": typer.colors.BRIGHT_GREEN,
        "bright_yellow": typer.colors.BRIGHT_YELLOW,
        "bright_blue": typer.colors.BRIGHT_BLUE,
        "bright_magenta": typer.colors.BRIGHT_MAGENTA,
        "bright_cyan": typer.colors.BRIGHT_CYAN,
        "bright_white": typer.colors.BRIGHT_WHITE,
    }

    typer_color = color_mapping.get(color.lower(), typer.colors.BRIGHT_CYAN)
    typer.echo(typer.style(text, fg=typer_color))
