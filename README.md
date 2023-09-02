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
source ./.venv/bin/activate


## Windows
source ./.venv/Scripts/activate
pip install -r requirements.txt
```


### Make a copy of the example environment variables file (this is not the virtual env; don't get confused)
```
## Linux
cp .env.example .env


## Windows
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

    --do_file_analysis \

    --do_export_csv_to_txt_files \
    --do_load_dataset_from_hub False \

    --do_train_tokenizer \
    --model_type bert \
    --block_size 512 \
    --clean_text \
    --handle_chinese_chars False \
    --strip_accents False \
    --lowercase \
    --vocab_size 30522 \
    --limit_alphabet 1000 \
    --min_frequency 2 \
    --do_push_tokenizer_to_hub \

    --do_reformat_files \
    --sliding_window_size 8 \

    --do_split_paths \

    --do_tokenize_files \

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
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 0.01 \
    --num_train_epochs 2 \
    --num_eval_epochs 10
```

As I am using the `tokenizers` library and not the `transfomers` one, I cannot just do `tokenizer.push_to_hub(huggingface_repo_name, private=True)`, but rather once training the tokenizer, I am cloning  the HuggingFace repo, moving the tokenizer files into the cloned repo, and pushing the tokenizer to HuggingFace. Don't worry though, as all there are done programmatically - look at `.src/hub_pusher.py`'s `push_tokenizer` function.

# Cyclone

Cyclone is the name of the super computer at the Cyprus Institute. In order to run a job you need to request access by following
this [link](https://hpcfsupport.atlassian.net/servicedesk/customer/portal/3/create/29). Through the process of getting access,
you need to pass over your public key so that you can ssh into the super computer.

Each user is being given a separate user directory. In the home directory there is a project folder with the name `data_p156`, where p156 is the project id. This folder is shared among all members of the project. The project folder also has generous quota for saving data and models so all our work should go there.

If the github repo is not there already, clone and create a virtualenv but to do this, you need to follow below.


### Cloning GitHub repo on the supercomputer
Before cloning the github repo, you need to do:
1. `cd ~/.ssh` on the supercomputer to copy the content of the `id_rsa.pub` file which is your public key
2. Go to your GitHub's Settings - `SSH and GPG keys`,
3. Click `New SSH key`
4. Add a descriptive title and paste the key into the "Key" field and click `Add SSH key`.
5. Try to clone your github repo in the SSH way; for example, by doing `git clone git@github.com:pedroandreou/Cypriot-LLM.git`


### Transferring the raw dataset to the supercomputer
Transfer the dataset from your local machine to the cloned git repo on the supercomputer by downloading [WinSCP](https://winscp.net/eng/download.php) for Windows.
If your local OS is different, then follow this [documentation page](https://hpcf.cyi.ac.cy/documentation/data_transfer.html) for doing so.


### Create cleaned files dir
In the root directory of the git cloned repo, run `mkdir cleaned_files`
```
Cypriot-LLM/
├── dataset/ # from 'Transferring the raw dataset to the supercomputer' step above
├── cleaned_files/ # from this step
├── README.md
└── ...
```

### Modify `.env` file's paths for raw dataset and cleaned files dir
and then in the `.env` file, add the paths of dataset and cleaned files as follows:
```
DATASET_DIR_PATH="/nvme/h/cy22pa1/data_p156/Cypriot-LLM/dataset"
CLEANED_FILES_DIR_PATH="/nvme/h/cy22pa1/data_p156/Cypriot-LLM/cleaned_files"
```

### If paths do not work:
To make paths work, go to the root directory of the git cloned repo and run:
```
pip install -e .
```

### Login to your HuggingFace account
Login by doing:
```
huggingface-cli login
```
and enter your huggingface token by creating it with `write` access [here](https://huggingface.co/settings/tokens)


### Submitting batch jobs
Do not submit more than a single job at once as it will lead to errors.

<br>

The array mechanism was used for partitioning my tasks into distinct jobs, saving us from having multiple job scripts or continuously adjusting the flags.

<br>

Store the following job script in a `job.sub` file using `vim` or any other text editor of your preference:
```
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=train.log
#SBATCH --error=error.log
#SBATCH --time=24:00:00
#SBATCH -A p156

module load Python/3.9.6-GCCcore-11.2.0

# Check if the .venv directory does not exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate

cd src

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    python main.py \
            --do_merge_docs

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    python main.py \
            --do_clean_data
            # --do_push_dataset_to_hub

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    python main.py \
            --do_export_csv_to_txt_files
            --do_load_dataset_from_hub False \

elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    python main.py \
            --do_file_analysis

elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    python main.py \
            --do_train_tokenizer \
           --model_type bert \
           --block_size 512
           # --do_push_tokenizer_to_hub \

elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    python main.py \
            --do_reformat_files \
            --sliding_window_size 8

elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    python main.py \
            --do_split_paths

elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    python main.py \
            --do_tokenize_files \
            --model_type bert \
            --block_size 512

elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    python main.py \
            --do_create_masked_encodings \
            --mlm_type manual \
            --mlm_probability 0.15

elif [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    python main.py \
            --do_train_model \
            --model_type bert \
            --trainer_type pytorch \
            --seed 42 \
            --vocab_size 30522 \
            --block_size 512 \
            --hidden_size 768 \
            --num_attention_heads 12 \
            --num_hidden_layers 6  \
            --type_vocab_size 1 \
            --train_batch_size 8 \
            --eval_batch_size 8 \
            --learning_rate 0.01 \
            --num_train_epochs 2 \
            --num_eval_epochs 10

elif [ $SLURM_ARRAY_TASK_ID -eq 11 ]; then
    python main.py \
            --model_type bert \
            # --model_path \ # to be defined
            --block_size 512

fi
```
you can see the output of the logs in `train.log` and the error in `error.log`

<br>

Submit a single job as:
```
sbatch --array=<task> job.sub
```
where `<task>` can be any from 1 to 11


More information on how to run jobs here https://hpcf.cyi.ac.cy/documentation/running_jobs.html
