
##create conda environment with python3.8
conda create --name speech-prompting python=3.8
conda activate speech-prompting

##clone this repo somewhere in the file system. Don't forget to add the path in the files
https://github.com/facebookresearch/textlesslib

##follow the instructions on this link to install dependencies
https://github.com/facebookresearch/textlesslib/tree/main/examples/twist

##after installing dependencies above, download models in the models models directory. Download the following 4 models to start. The complete list of models is in models/twist_models.py
cd models

wget https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/mhubert_base_25hz_cp_mls_cv_sp_fisher.pt
wget https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin
wget https://dl.fbaipublicfiles.com/textless_nlp/twist/lms/TWIST-350M.zip
unzip -r TWIST-350M.zip

##define the appropriate locations in evaluat file


##additional installations
pip install accelerate


