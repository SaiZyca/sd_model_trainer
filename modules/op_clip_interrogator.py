
import clip_interrogator
from . import op_torch
import os

ci = None
low_vram = False

# load default clip config
config = clip_interrogator.Config(
    device=op_torch.cuda_device.device,
    cache_path = 'cache',
    clip_model_name ='ViT-L-14/openai',
    caption_model_name ='blip-large',
    )

if low_vram:
    config.apply_low_vram_defaults()
    
ci = clip_interrogator.Interrogator(config)

def clip_interrogator_version():
    '''
    return clip_interrogator version
    '''
    return clip_interrogator.__version__

def list_caption_models():
    '''
    list clip_interrogator caption models
    '''
    return clip_interrogator.list_caption_models()

def list_clip_models():
    '''
    list clip_interrogator clip models
    '''
    return clip_interrogator.list_clip_models()

def load(clip_model_name):
    global ci
    if ci is None:
        print(f"Loading CLIP Interrogator {clip_interrogator_version()}...")

        config = clip_interrogator.Config(
            device=op_torch.cuda_device.device,
            cache_path = 'cache',
            clip_model_name=clip_model_name,
        )
        if low_vram:
            config.apply_low_vram_defaults()
        ci = clip_interrogator.Interrogator(config)

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

def unload():
    global ci
    if ci is not None:
        print("Offloading CLIP Interrogator...")
        ci.caption_model = ci.caption_model.to(op_torch.cuda_device.cpu)
        ci.clip_model = ci.clip_model.to(op_torch.cuda_device.cpu)
        ci.caption_offloaded = True
        ci.clip_offloaded = True
        op_torch.cuda_device.torch_gc()

def image_analysis(image, clip_model_name):
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
    
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

def image_to_prompt(image, mode, clip_model_name, blip_model_name):
    if blip_model_name != ci.config.caption_model_name:
        ci.config.caption_model_name = blip_model_name
        ci.load_caption_model()

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
    
    # ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    # ci.config.flavor_intermediate_count = 16 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    
    image = image.convert("RGB")
    if mode == 'best':
        prompt = ci.interrogate(image)
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image)
    elif mode == 'fast':
        prompt =  ci.interrogate_fast(image)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    
    with open('readme.txt', 'w', encoding='UTF-8') as f:
        f.write(prompt)
    
    return prompt

def batch_process(image_folder, mode, clip_model_name, blip_model_name, prefix_text, postfix_text):
    # setup interrogate
    if blip_model_name != ci.config.caption_model_name:
        ci.config.caption_model_name = blip_model_name
        ci.load_caption_model()

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

def interrogate_folder(image_folder, prefix_caption="", mode="best", blip_model_name="blip-large", clip_model_name="ViT-L-14/openai"):
    # setup interrogate
    if blip_model_name != ci.config.caption_model_name:
        ci.config.caption_model_name = blip_model_name
        ci.load_caption_model()

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
        
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
                if mode == 'best':
                    prompt += ci.interrogate(image)
                elif mode == 'classic':
                    prompt += ci.interrogate_classic(image)
                elif mode == 'fast':
                    prompt +=  ci.interrogate_fast(image)
                elif mode == 'negative':
                    prompt += ci.interrogate_negative(image)
                
                with open(caption_path, 'w', encoding='UTF-8') as f:
                    f.write(prompt)
    
    print ("interrogate finish")  