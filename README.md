# Access the Google Colab notebook below for further preprocessing
    https://colab.research.google.com/drive/1cDvsdvFXVbQgoTl4_-t7XqY9D0-M91UU#scrollTo=QIjnWfk381H4


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
```


### How to update the requirements
```
cmd.exe /C setup_new_environment.bat
```


### How to create a Dataframe out of all the docs
```
python multiformat_document_reader.py "G:\My Drive\Uni\Masters\Thesis\dataset"
```


### How to compare the tokens of files
```
python compare_token_counts.py document_data_2023=07=05_12=28=20.xlsx document_data_2023=07=05_19=27=10.xlsx output.xlsx another_file.xlsx yet_another_file.xlsx
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
