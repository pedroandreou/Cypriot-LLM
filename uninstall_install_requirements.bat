@echo off

:: Uninstall all packages in the current virtual environment
for /f "delims=" %%i in ('pip freeze') do pip uninstall -y %%i

:: Install the packages listed in unpinned_requirements.txt
pip install -r unpinned_requirements.txt

:: Write the resulting package list to requirements.txt
pip freeze | findstr /v "pkg-resources" >> requirements.txt

echo All done!
