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
cd ./src

python main.py \
    --do_merge_docs \
    --do_clean_data \
    --do_push_to_hub \
    --do_login_first_time \
    --do_export_csv_to_txt_files \
    --do_file_analysis
```


## :runner: Training Tokenizer
```
cd ./src/_02_tokenizer_training

python main.py \
    --model_type bert \
    --do_train_tokenizer \
    --do_push_tokenizer_to_hub False
```
If you want to add different arguments for training the tokenizer, just go to the `initial_configs` directory where you will find a config JSON file for the corresponding model. Change the values there and rerun the script.


## :books: Reformat all data (using 4 or 8 sliding window) for being able to train the model
```
cd ./src/_03_data_reformatting

python reformatter.py
```


## :books: Split Data
```
cd ./src/_04_path_splitting

python main.py \
    --do_split_paths
```


## :runner: Tokenize Data
```
cd ./src/_05_data_tokenizing

python main.py \
    --model_type bert \
    --do_tokenize_data True \
    --paths=train_test \
    --do_create_masked_encodings \
    --mlm_type=manual \
    --mlm_probability 0.15
```


## :runner: Training Model
```
cd ./src/_06_model_training

python main.py \
    --model_type bert \
    --do_train_model True \
    --trainer_type pytorch \
    --seed 42 \
    --vocab_size 30522 \
    --block_size 512 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --num_hidden_layers 6 \
    --type_vocab_size 1
```


## :trophy: Inference
```
cd ./src/_07_inference
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
