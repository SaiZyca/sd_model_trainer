from pathlib import Path
from clip_interrogator import Config, Interrogator
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

def image_analysis(image):
    caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
    clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]
    config = Config()
    config.clip_model_name = clip_model_name
    config.caption_model_name = caption_model_name
    ci = Interrogator(config)
    
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def image_to_prompt(image, mode):
    caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
    clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]
    config = Config()
    config.clip_model_name = clip_model_name
    config.caption_model_name = caption_model_name
    ci = Interrogator(config)
    
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 16 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert("RGB")
    if mode == 'best':
        prompt = ci.interrogate(image)
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image)
    elif mode == 'fast':
        prompt =  ci.interrogate_fast(image)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    
    with open('readme.txt', 'w') as f:
        f.write(prompt)

def interrogate_folder(image_folder, prefix_caption="", clip_mode="best", caption_model="blip-large", clip_model="ViT-L-14/openai"):
    # setup interrogate
    config = Config()
    config.clip_model_name = clip_model #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]
    config.caption_model_name = caption_model #@param ["blip-base", "blip-large", "git-large-coco"]
    ci = Interrogator(config)
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 16 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    
    
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith((".png", ".jpg")):
                image_file = os.path.join(root, file)
                image = Image.open(image_file).convert("RGB")
                prompt = prefix_caption
                file_name, ext = os.path.splitext(image_file)
                caption_path = file_name+".txt"
                if clip_mode == 'best':
                    prompt += ci.interrogate(image)
                elif clip_mode == 'classic':
                    prompt += ci.interrogate_classic(image)
                elif clip_mode == 'fast':
                    prompt +=  ci.interrogate_fast(image)
                elif clip_mode == 'negative':
                    prompt += ci.interrogate_negative(image)
                
                with open(caption_path, 'w') as f:
                    f.write(prompt)
    
    print ("interrogate finish")  
    
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