@echo off
REM Remove the old virtual environment
rmdir /s /q .venv

REM Create a new virtual environment
python -m venv .venv

REM Activate the virtual environment
call .venv\Scripts\activate

REM Install the packages listed in unpinned_requirements.txt
pip install -r unpinned_requirements.txt

REM Write the resulting package list to requirements.txt
pip freeze | findstr /v "pkg-resources" > requirements.txt

REM Echo Done to indicate success
echo All done!
