import os
#os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ORT_LOGGING_LEVEL'] = '3'
from .main import Model 
from .main import supported_models 