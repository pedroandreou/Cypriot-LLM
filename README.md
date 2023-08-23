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


### Make a copy of the example environment variables file (this is not the virtual env; don't get confused)
```
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
    --do_push_dataset_to_hub \


    --do_export_csv_to_txt_files \

    --do_file_analysis \

    --do_train_tokenizer \
    --model_type bert \
    --block_size 512 \
    --do_push_tokenizer_to_hub \

    --do_reformat_files \
    --sliding_window_size 8 \

    --do_split_paths \

    --do_tokenize_files \
    --paths train_test \

    --do_create_masked_encodings \
    --mlm_type manual \
    --mlm_probability 0.15 \

    --do_train_model
    --trainer_type pytorch \
    --seed 42 \
    --vocab_size 30522 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --num_hidden_layers 6  \
    --type_vocab_size 1 \
    --learning_rate 0.01 \
    --max_steps 1_000_000 \

    --do_login_first_time
```

If you want to add different arguments for training the tokenizer, just go to the `initial_configs` directory where you will find a config JSON file for the corresponding model. Change the values there and rerun the script.

As I am using the `tokenizers` library and not the `transfomers` one, I cannot just do `tokenizer.push_to_hub(huggingface_repo_name, private=True)`, but rather once training the tokenizer, I am cloning  the HuggingFace repo, moving the tokenizer files into the cloned repo, and pushing the tokenizer to HuggingFace. Don't worry though, as all there are done programmatically - look at `.src/hub_pusher.py`'s `push_tokenizer` function.


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
