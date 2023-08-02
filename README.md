# :memo: Google Colab Notebooks
- Preprocessing<br>
    https://colab.research.google.com/drive/15PowGyrbMmuWo7UHDszL79fqgvVenKMj#scrollTo=AxXFXqijO9yW
- Testing different models<br>
    https://colab.research.google.com/drive/1HctL2fAkkYQi-POhiISeXLrLn5rPNTJe#scrollTo=XiB3JcbY1WS4
- This repo's code<br>
    https://colab.research.google.com/drive/1n0M7negqCKG--658g3hBdmpZ0IcgZgNe#scrollTo=Bzr8cCpGeqJO&uniqifier=3
<br>
<br>
The repository contains the code from the Notebooks. To amend or enhance the code, directly utilize the Notebooks, which contain debugging functions that are not part of this repo.
<br>
<br>
Despite executing identical code, the output dataframes from the Notebook and the repository differ in token count - 1,495,516 and 1,495,682 respectively. This discrepancy may be attributable to my laptop's hardware. Consequently, I'll proceed using the output CSV from the Notebook.


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
#### Create a CSV containg all the docs
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
python compare_token_counts.py --files='["all_documents.csv", "preprocessed_docs.csv"]'
```
#### Calculate the file capacity
```
python calculate-file-capacity.py --input-file-name="preprocessed_docs"
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
