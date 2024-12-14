
@echo off
echo Setting up PDF AI Training System...

:: Create Python virtual environment
python -m venv venv
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

:: Create necessary directories
mkdir data\raw_pdfs
mkdir data\processed
mkdir models
mkdir logs

echo Setup complete!
echo Please place your PDF files in the data\raw_pdfs directory
echo Run main.py to start training
pause