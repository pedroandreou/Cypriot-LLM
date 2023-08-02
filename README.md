# :memo: Google Colab Notebooks
- Preprocessing
    https://colab.research.google.com/drive/15PowGyrbMmuWo7UHDszL79fqgvVenKMj#scrollTo=AxXFXqijO9yW
- Testing different models
    https://colab.research.google.com/drive/1HctL2fAkkYQi-POhiISeXLrLn5rPNTJe#scrollTo=XiB3JcbY1WS4
- CyBert and CyRoberta Code
    https://colab.research.google.com/drive/1n0M7negqCKG--658g3hBdmpZ0IcgZgNe#scrollTo=AfPCh_SskAKs
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
#### How to create a CSV containg all the docs
```
cd create_doc_csv.py --data_path=r"G:\My Drive\Uni\Masters\Thesis\dataset" --output_file_name="all_documents"
```
### How to preprocess the docs of the CSV
```
cd clean_data.py --input_file_name="all_documents" --output_file_name="preprocessed_docs"
```
### How to compare the tokens of files
```
python compare_token_counts.py --files=["first_file.csv", "another_file.csv", "yet_another_file.csv"]
```
### How to calculate the file capacity
```
python calculate_file_capacity.py --input_file_name="preprocessed_docs"
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
