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


# :runner: How to run the code
```
cd ./src
```

```
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

    --do_train_model \
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

# Cyclone

Cyclone is the name of the super computer at the Cyprus Institute. In order to run a job you need to request access by following
this [link](https://hpcfsupport.atlassian.net/servicedesk/customer/portal/3/create/29). Through the process of getting access,
you need to pass over your public key so that you can ssh into the super computer.

Each user is being given a separate user directory. In the home directory there is a project folder with the name `data_p105`, p105
is the project id. This folder is shared among all members of the project. The project folder also has generous quota for saving data
and models so all our work should go there.

If the github repo is not there already, clone and create a virtualenv but to do this, you need to follow below.


Before cloning the github repo, you need to do:
1. `cd ~/.ssh` on the supercomputer to copy the content of the `id_rsa.pub` file which is your public key
2. Go to your GitHub's Settings - `SSH and GPG keys`,
3. Click `New SSH key`
4. Add a descriptive title and paste the key into the "Key" field and click `Add SSH key`.
5. Try to clone your github repo in the SSH way; for example, `git clone git@github.com:MantisAI/cypriot_bert.git`, it will work.


Transfer the dataset from your local machine to the git repo on the HPC system by downloading [WinSCP](https://winscp.net/eng/download.php).
Follow this [documentation page](https://hpcf.cyi.ac.cy/documentation/data_transfer.html) for doing so.


Copy the environment variables as follows and change the paths to the location that uploaded the dataset:
```
cp .env.example .env
vim .env
```

For example in the root directory, add the dataset directory containing all the documents and also in the root directory, run `mkdir cleaned_files`
```
Cypriot-LLM/
├── cleaned_files/
├── dataset/
├── README.md
└── ...
```
and then in the `.env` file, the paths for dataset and cleaned files should be as follows:
```
DATASET_DIR_PATH="/nvme/h/cy22pa1/data_p156/Cypriot-LLM/dataset"
CLEANED_FILES_DIR_PATH="/nvme/h/cy22pa1/data_p156/Cypriot-LLM/cleaned_files"
```

To make paths work, go to the root directory and run:
```
pip install -e .
```

Then you need to run your batch script, but for doing so, you need to do this manually on the supercomputer:
```
source ./.venv/bin/activate
pip install -r requirements.xt
```

And lastly, you also need to login into your HuggingFace account:
```
huggingface-cli login
```
and enter your huggingface token by creating it with `write` access [here](https://huggingface.co/settings/tokens)

Then you will be able to run your batch job by doing:
```
sbatch job.sub
```

Where the job script is:
```
#!/bin/bash

#SBATCH --job-name=test-train
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=train.log
#SBATCH --error=error.log
#SBATCH -A p156

module load Python/3.9.6-GCCcore-11.2.0
source .venv/bin/activate
python src/main.py --do_merge_docs False \
                   --do_clean_data False \
                   --do_push_dataset_to_hub False \
                   --do_export_csv_to_txt_files False \
                   --do_file_analysis False \
                   --do_train_tokenizer \
                   --model_type bert \
                   --block_size 512 \
                   --do_push_tokenizer_to_hub False
```
you can see the output of the logs in `train.log` and the error in `error.log`


More information on how to run jobs here https://hpcf.cyi.ac.cy/documentation/running_jobs.html
