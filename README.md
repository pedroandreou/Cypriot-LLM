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
cmd.exe /C uninstall_install_requirements.bat
```


### How to run the Preprocessing stage
```
python count_tokens.py C:\\Users\\user\\Desktop\\dataset

python create_df.py C:\\Users\\user\\Desktop\\dataset
```


## 🛠 Initialization & Setup
    git clone https://github.com/pedroandreou/Cypriot-LLM.git
