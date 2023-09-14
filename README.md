# Cypriot LLM

## :building_construction: Environment

### You should create a virtualenv with the required dependencies by running
```
python -m venv .venv
```


### How to activate the virtual environment to run the code
```
## Linux
source ./.venv/bin/activate
pip install -r requirements.txt


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
    --tokenizer_type WP \
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
    --tokenizer_version 1 \

    --do_mask_files \
    --encodings_version 1 \
    --mlm_type static \
    --mlm_probability 0.15 \

    --do_train_model \
    --model_type bert \
    --masked_encodings_version 1 \
    --seed 42 \
    --vocab_size 30522 \
    --num_hidden_layers 6  \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --type_vocab_size 1 \
    --trainer_type pytorch \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \

    --do_inference \
    --model_version 1 \
    --input_unmasked_sequence "είσαι"
    <!-- --input_masked_sequences "Θώρει τη [MASK]." "Η τηλεόραση, το [MASK], τα φώτα." "Μεν τον [MASK] κόρη μου." -->
```

As I am using the `tokenizers` library and not the `transfomers` one, I cannot just do `tokenizer.push_to_hub(huggingface_repo_name, private=True)`, but rather once training the tokenizer, I am cloning  the HuggingFace repo, moving the tokenizer files into the cloned repo, and pushing the tokenizer to HuggingFace. Don't worry though, as all there are done programmatically - look at `.src/hub_pusher.py`'s `push_tokenizer` function.


# :computer: Cyclone

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

```
Cypriot-LLM/
├── dataset/ # transfer it here
├── README.md
└── ...
```

### Modify environment variables file
Go to the `.env` file and add the path of the dataset as follows:
```
DATASET_DIR_PATH="/nvme/h/cy22pa1/data_p156/Cypriot-LLM/dataset"
```

### If the repo paths do not work:
Go to the root directory of the git cloned repo and run:
```
pip install -e .
```

### Login to your HuggingFace account
Login by doing:
```
huggingface-cli login
```
and enter your huggingface token by creating it with `write` access [here](https://huggingface.co/settings/tokens)


### Weight & Biases

The model's loss is logged in the W&B Platform.
Provide your API key when prompted; finding it [here](https://wandb.ai/settings).

<br>

We do not have sudo permissions for writing to `/tmp` on the supercomputer.

Thus, create the necessary directories, set and export the environment variables, adjust permissions, and add the environment variables to your .bashrc for persistence by doing:
<pre>
chmod +x setup_wandb.sh
./setup_wandb.sh
</pre>


### Submitting batch jobs

The array mechanism was used for partitioning my tasks into distinct jobs, saving us from having multiple job scripts or continuously adjusting the flags. Look into the `job.sub` script.

<br>

Typically, the job mechanism is utilized to submit multiple similar jobs that run concurrently. This might result in the second job completing faster than the first leading to errors. However, in our setup, each stage relies on the completion of the previous one.

<br>

In other words, do not submit more than a single job at once and expect the single submitted job to finish before you execute its successor job.

<br>

Submit a single job as:
```
sbatch --job-name=Stage<task> --array=<task> job.sub
```
where `<task>` can be any number from 1 to 11

<br>
<br>
Suggestion: when you inference, you will see many unicode characters in the output `train.log` file if you access it with a text editor as I am doing `Console(force_terminal=True)` for adding color to the token that replaced the mask token in the output Table that contains the predictions.
That said, just do `cat train.log`for seeing the file.
<br>
<br>

More information on how to run jobs here https://hpcf.cyi.ac.cy/documentation/running_jobs.html
