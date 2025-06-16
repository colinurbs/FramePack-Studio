@echo off
echo FramePack-Studio Setup Script
setlocal enabledelayedexpansion

REM Check if Python is installed (basic check)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH. Please install Python and try again.
    goto end
)

if exist "%cd%/venv" (
echo Virtual Environment already exists. 
set /p choice= "Do you want to reinstall packages?[Y/N]: "

if "!choice!" == "y" (goto checkgpu)
if "!choice!"=="Y" (goto checkgpu)

goto end
)

REM Check the python version
echo Python versions 3.10-3.12 have been confirmed to work. Other versions are currently not supported. You currently have:
python -V
set choice=
set /p choice= "Do you want to continue?[Y/N]: "


if "!choice!" == "y" (goto makevenv)
if "!choice!"=="Y" (goto makevenv)

goto end

:makevenv
REM This creates a virtual environment in the folder
echo Creating a Virtual Environment...
python -m venv venv
echo Upgrading pip in Virtual Environment to lower chance of error...
"%cd%/venv/Scripts/python.exe" -m pip install --upgrade pip

:checkgpu
REM ask Windows for GPU
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Nvidia GPU doesn't exist or drivers installed incorrectly. Please confirm your drivers are installed.
    goto end
)

echo Checking your GPU...

for /F "tokens=* skip=1" %%n in ('nvidia-smi --query-gpu=name') do set GPU_NAME=%%n && goto gpuchecked

:gpuchecked
echo Detected %GPU_NAME%
set "GPU_SERIES=%GPU_NAME:*RTX =%"
set "GPU_SERIES=%GPU_SERIES:~0,2%00"

REM This gets the shortened Python version for later use. e.g. 3.10.13 becomes 310.
for /f "delims=" %%A in ('python -V') do set "pyv=%%A"
for /f "tokens=2 delims= " %%A in ("%pyv%") do (
    set pyv=%%A
)
set pyv=%pyv:.=%
set pyv=%pyv:~0,3%

echo Installing torch...

if !GPU_SERIES! geq 5000 (
	goto torch270
) else (
	goto torch260
)

REM RTX 5000 Series
:torch270
"%cd%/venv/Scripts/pip.exe" install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
REM Check if pip installation was successful
if %errorlevel% neq 0 (
    echo Warning: Failed to install dependencies. You may need to install them manually.
    goto end
)
 
REM Ask if user wants Sage Attention
set choice=
echo Do you want to install any of the following? They speed up generation.
echo 1) Sage Attention
echo 2) Flash Attention 
echo 3) BOTH!
echo 4) No
set /p choice= "Input Selection: "

set both="N"

if "!choice!" == "1" (goto triton270)
if "!choice!"== "2" (goto flash270)
if "!choice!"== "3" (set both="Y"
goto triton270
)

goto requirements

:triton270
REM Sage Attention and Triton for Torch 2.7.0
"%cd%/venv/Scripts/pip.exe" install "triton-windows<3.4" --force-reinstall
"%cd%/venv/Scripts/pip.exe" install "https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp%pyv%-cp%pyv%-win_amd64.whl" --force-reinstall
echo Finishing up installing triton-windows. This requires extraction of libraries into Python Folder...

REM Check for python version and download the triton-windows required libs accordingly
if %pyv% == 310 (
	powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.10.11_include_libs.zip', 'triton-lib.zip')"
)

if %pyv% == 311 (
	powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.11.9_include_libs.zip', 'triton-lib.zip')"
)

if %pyv% == 312 (
	powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip', 'triton-lib.zip')"
)
	
REM Extract the zip into the Python Folder and Delete zip
powershell Expand-Archive -Path '%cd%\triton-lib.zip' -DestinationPath '%cd%\venv\Scripts\' -force
del triton-lib.zip
if %both% == "Y" (goto flash270)

goto requirements

:flash270
REM Install flash-attn. 
"%cd%/venv/Scripts/pip.exe" install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%%2Bcu128torch2.7.0cxx11abiFALSE-cp%pyv%-cp%pyv%-win_amd64.whl?download=true" 
goto requirements


REM RTX 4000 Series and below
:torch260 
"%cd%/venv/Scripts/pip.exe" install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
REM Check if pip installation was successful
if %errorlevel% neq 0 (
    echo Warning: Failed to install dependencies. You may need to install them manually.
    goto end
)

REM Ask if user wants Sage Attention
set choice=
echo Do you want to install any of the following? They speed up generation.
echo 1) Sage Attention
echo 2) Flash Attention 
echo 3) BOTH!
echo 4) No
set /p choice= "Input Selection: "

set both="N"

if "!choice!" == "1" (goto triton260)
if "!choice!"== "2" (goto flash260)
if "!choice!"== "3" (set both="Y"
goto triton260)

goto requirements

:triton260
REM Sage Attention and Triton for Torch 2.6.0
"%cd%/venv/Scripts/pip.exe" install "triton-windows<3.3.0" --force-reinstall
"%cd%/venv/Scripts/pip.exe" install  https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp%pyv%-cp%pyv%-win_amd64.whl --force-reinstall

echo Finishing up installing triton-windows. This requires extraction of libraries into Python Folder...

REM Check for python version and download the triton-windows required libs accordingly
if %pyv% == 310 (
	powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.10.11_include_libs.zip', 'triton-lib.zip')"
)

if %pyv% == 311 (
	powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.11.9_include_libs.zip', 'triton-lib.zip')"
)

if %pyv% == 312 (
	powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip', 'triton-lib.zip')"
)
	
REM Extract the zip into the Python Folder and Delete zip
powershell Expand-Archive -Path '%cd%\triton-lib.zip' -DestinationPath '%cd%\venv\Scripts\' -force
del triton-lib.zip

if %both% == "Y" (goto flash260)

goto requirements

:flash260
REM Install flash-attn. 
"%cd%/venv/Scripts/pip.exe" install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4%%2Bcu126torch2.6.0cxx11abiFALSE-cp%pyv%-cp%pyv%-win_amd64.whl?download=true" 

:requirements
echo Installing remaining required packages through pip...
REM This assumes there's a requirements.txt file in the root
"%cd%/venv/Scripts/pip.exe" install -r requirements.txt 

REM Check if pip installation was successful
if %errorlevel% neq 0 (
    echo Warning: Failed to install dependencies. You may need to install them manually.
    goto end
)

echo Setup complete.

:end
echo Exiting setup script.
pause
