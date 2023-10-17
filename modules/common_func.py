from pathlib import Path
import clip_interrogator
from PIL import Image
import uuid
import os
import toml
import subprocess, sys

def create_train_project(name=""):
    train_project_folder = r"%s\train_project" % Path().resolve()
    project_folder = r"%s\%s" % (train_project_folder, (name+"_"+uuid.uuid1().hex), )
    img_folder = r"%s\img" % project_folder
    data_set_folder = r"%s\10_dataset" % img_folder
    model_folder = r"%s\model" % project_folder
    log_folder = r"%s\log" % project_folder
    os.makedirs(project_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(data_set_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    
    return project_folder, img_folder, data_set_folder, model_folder, log_folder
    
def generate_lora_toml(project_folder):
    toml_file = "%s\moonshot_train.toml" % project_folder
    data = toml.load("moonshot_train.toml")
    # Modify field
    data['dataset']['train_data_dir']="%s\\img" % project_folder 
    data['saving']['output_dir']="%s\\model" % project_folder 
    data['saving']['logging_dir']="%s\\log" % project_folder 

    # To use the dump function, you need to open the file in 'write' mode
    # It did not work if I just specify file location like in load
    f = open(toml_file,'w')
    toml.dump(data, f)
    f.close()

def start_train_lora(lora_trainer_folder, project_folder):
    config_file = "%s\moonshot_train.toml" % project_folder
    ps_command = "cd %s; .\\train_by_toml_msai.ps1 -config_file %s" % (lora_trainer_folder, config_file)
    command = 'powershell.exe -command "%s"' % ps_command
    subprocess.call(command, shell=True)
    model_path = "%s\\model\\custom_lora.safetensors" % project_folder
    return model_path