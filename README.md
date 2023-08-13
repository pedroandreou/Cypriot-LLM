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


### Make a copy of the example environment variables file (this is not the virtual env; don't get confused; it's just for keeping your HuggingFace token secured)
```
cd src
xcopy .env.example .env
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
cd ./doc_merge_to_csv
python docs_to_csv.py \
    --merge_data
```

#### Preprocess the docs of the CSV
```
cd ./data_cleaner
```

If you want to clean the data and push it to the hub for the first time
```
python main.py \
    --do_clean_data \
    --do_push_to_hub \
    --first_time_login
```

If the data is already cleaned, you are already logged in to the huggingface-cli and you just want to push it to the hub
```
python main.py \
    --do_push_to_hub
```

#### Export all docs to separate txt files as this would make our life easier when the tokenizer needs the paths to the files
```
python export_csv_docs_to_txt_files.py
```


## :runner: Training
```
cd ./src/training
```
#### First time: Split Paths to 80% for training and 20% for testing, Train tokenizer, Create dataset masked encodings
```
python main.py \
    --do_split_paths \
    --do_train_tokenizer \
    --do_create_masked_encodings
```
#### Second time: Just push tokenizer to the HuggingFace Hub
```
python main.py \
    --do-push-tokenizer-to-hub
```
#### Third time: Just focus on Training the Model
```
python main.py \
    --do_train_model
```


## :trophy: Inference
```
cd ./src/inference
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
