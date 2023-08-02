import typer
import pandas as pd


def main(
    input_file_name: str = "preprocessed_docs",
    output_dir_path: str = "/content/drive/MyDrive/Uni/Masters/Thesis/cleaned_files",
):
    df = pd.read_csv(f"{input_file_name}.csv")

    for num in range(len(df)):
        value = df.iloc[num, 1]
        # print(value)
        with open(f"{output_dir_path}/text_file{num}.txt", "w") as f:
            f.write(str(value))


if __name__ == "__main__":
    typer.run(main)
