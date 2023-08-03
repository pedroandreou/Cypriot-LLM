# :memo: Google Colab Notebooks
All of the following Notebooks are integral components of this repository. They come equipped with additional debugging functions designed to assist future developers in expanding upon them.<br>
- Preprocessing<br>
    https://colab.research.google.com/drive/15PowGyrbMmuWo7UHDszL79fqgvVenKMj#scrollTo=AxXFXqijO9yW
- Testing different use cases following various tutorials<br>
    https://colab.research.google.com/drive/1HctL2fAkkYQi-POhiISeXLrLn5rPNTJe#scrollTo=XiB3JcbY1WS4
- Training<br>
    https://colab.research.google.com/drive/1n0M7negqCKG--658g3hBdmpZ0IcgZgNe#scrollTo=Bzr8cCpGeqJO&uniqifier=3
<br>
<br>

## :building_construction: Environment

### You should create a virtualenv with the required dependencies by running
```
## Linux
make virtualenv


## Windows
python -m venv .venv
```


### How to activate the virtual environment to run the code
```
## Linux
source ./.env/bin/activate


## Windows
source ./.venv/Scripts/activate
pip install -r requirements.txt
```


### How to update the requirements
When a new requirement is needed you should add it to `unpinned_requirements.txt` and run
```
cmd.exe /C setup_new_environment.bat
```


# :crossed_flags: Source Code
## :hammer: Preprocessing
```
cd ./src/preprocessing
```
#### Create a CSV containing all the docs
```
python create_doc_csv.py --data-path="G:\My Drive\Uni\Masters\Thesis\dataset" --output-file-name="all_documents"
```
#### Preprocess the docs of the CSV
```
python clean_data.py --input-file-name="all_documents" --output-file-name="preprocessed_docs"
```
#### Export all docs to separate txt files as this would make our life easier when the tokenizer will need the paths to the files
```
python export_csv_docs_to_txt_files.py --input-file-name="preprocessed_docs" --output-dir-path="/content/drive/MyDrive/Uni/Masters/Thesis/cleaned_files"
```
#### Compare the tokens of files
```
python compare_token_counts.py "all_documents" "preprocessed_docs"
```
#### Calculate the file capacity
```
python calculate_file_capacity.py --input-file-name="preprocessed_docs"
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
