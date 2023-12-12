#!/usr/bin/env python3
import clip_interrogator
from . import op_torch
import os
from PIL import Image

ci = None
low_vram = False
cancel_status = False

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

    cuda_info=device_info()
    
    return cuda_info

def cancel_batch():
    global cancel_status
    cancel_status = True

def setup_interrogator(blip_model, clip_model, caption_max_length, chunk_size, flavor_intermediate_count):
    if blip_model != ci.config.caption_model_name:
        ci.config.caption_model_name = blip_model
        ci.load_caption_model()

    if clip_model != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model
        ci.load_clip_model()
    
    # config
    ci.config.caption_max_length = int(caption_max_length)
    ci.config.chunk_size = int(chunk_size)
    ci.config.flavor_intermediate_count = int(flavor_intermediate_count)

def device_info():
    cuda_info = op_torch.cuda_device.info()
    info = ("%s:%s GB | Free: %s GB | Allocated: %s GB | Cached: %s GB " % (\
                cuda_info['device'],\
                cuda_info['Total Vram'],\
                cuda_info['Free Vram'],\
                cuda_info['Vram Allocated'],\
                cuda_info['Vram Cached'],)\
                    )
    return info

def image_analysis(image, clip_model, num_top_classes):
    if clip_model != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model
        ci.load_clip_model()
    
    num_top_classes = int(num_top_classes)
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ", ".join(ci.mediums.rank(image_features, num_top_classes))
    top_artists = ", ".join(ci.artists.rank(image_features, num_top_classes))
    top_movements = ", ".join(ci.movements.rank(image_features, num_top_classes))
    top_trendings = ", ".join(ci.trendings.rank(image_features, num_top_classes))
    top_flavors = ", ".join(ci.flavors.rank(image_features, num_top_classes))
    
    return top_mediums, top_artists, top_movements, top_trendings, top_flavors

def image_to_prompt(image, mode):
    image = image.convert("RGB")
    if mode == 'best':
        prompt = ci.interrogate(image)
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image)
    elif mode == 'fast':
        prompt =  ci.interrogate_fast(image)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    elif mode == 'caption':
        prompt = ci.generate_caption(image)
    elif mode == 'Manual':
        prompt = ""

    return prompt

def caption_image(image, mode, clip_model, blip_model, caption_max_length, chunk_size, flavor_intermediate_count, num_top_classes, prefix_text, postfix_text, filter_text):
    
    setup_interrogator(blip_model, clip_model, caption_max_length, chunk_size, flavor_intermediate_count)
    
    top_mediums, top_artists, top_movements, top_trendings, top_flavors = image_analysis(image, clip_model, num_top_classes)

    prompt = "%s%s%s" % (prefix_text, image_to_prompt(image, mode), postfix_text)
    
    filter_text_list = filter_text.replace(' ','').split(',')
    for text in filter_text_list:
        prompt = prompt.replace(text, '')
    
    with open('last_prompt.txt', 'w', encoding='UTF-8') as f:
        f.write(prompt)
        
    cuda_info=device_info()
    
    return prompt, cuda_info, top_mediums, top_artists, top_movements, top_trendings, top_flavors

def batch_caption_images(batch_folder, mode, clip_model, blip_model, 
                         caption_max_length, chunk_size, flavor_intermediate_count, 
                         prefix_text, postfix_text, img_exts, caption_ext, filter_text, filename_filter):
    setup_interrogator(blip_model, clip_model, caption_max_length, chunk_size, flavor_intermediate_count)
    
    new_list = list()
    for ext in list(img_exts.split(",")):
        if ext.startswith("."):
            ext = ext
        else:
            ext = ".%s" % ext
        new_list.append(ext)
        
    if caption_ext == "":
        caption_ext = ".txt" 
    if caption_ext.startswith(".") is False:
        caption_ext = ".%s" % caption_ext
    
    global cancel_status
    
    for root, dirs, files in os.walk(batch_folder):
        for file in files:
            try:
                if cancel_status:
                    print ("interrogate Interrupt")
                    break
                if file.endswith(tuple(new_list)) and filename_filter in file:
                    image_file = os.path.join(root, file)
                    image = Image.open(image_file).convert("RGB")
                    file_name, ext = os.path.splitext(image_file)
                    caption_path = "%s%s" % (file_name, caption_ext)
                        
                    prompt = "%s%s%s" % (prefix_text, image_to_prompt(image, mode), postfix_text)
                    
                    filter_text_list = filter_text.replace(' ','').split(',')
                    for text in filter_text_list:
                        prompt = prompt.replace(text, '')
                    
                    with open(caption_path, 'w', encoding='UTF-8') as f:
                        f.write(prompt)
                    
                    print ("%s prompt sucess" % caption_path)
                        
            except OSError as e:
                print(f"{e}; continuing")

    cancel_status = False
    print ("interrogate finish")
    
    cuda_info=device_info()
    
    return cuda_info
