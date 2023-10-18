#!/usr/bin/env python3
import gradio as gr
from . import op_torch, symbols
from . import op_clip_interrogator as Ci_op
from . import op_tinker



def prompt_tab():
    with gr.Group():
        with gr.Accordion("clip interrogator Config", open=True):
            with gr.Row():
                caption_max_length = gr.Number(value=32, label="caption max length", minimum=1,step=1, interactive=True,)  
                chunk_size = gr.Number(value=1024, label="chunk/batch size", minimum=1,step=1, interactive=True,)
                flavor_intermediate_count = gr.Number(value=2048, label="flavor count", minimum=1,step=1, interactive=True,)
            with gr.Row():
                mode = gr.Dropdown(['best', 'fast', 'classic', 'negative', 'caption'], value='caption', label='Mode')
                clip_model = gr.Dropdown(Ci_op.list_clip_models(), value=Ci_op.ci.config.clip_model_name, label='CLIP Model')
                blip_model = gr.Dropdown(Ci_op.list_caption_models(), value=Ci_op.ci.config.caption_model_name, label='Caption Model')
    with gr.Group():
        with gr.Row(equal_height=True):
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=6)
                prompt_button = gr.Button("Generate prompt")
    with gr.Group():
        data_type = gr.Radio(choices=["Files", "Folder"], value="Folder", label="Offline data type",visible=False)
        with gr.Row():
            batch_folder = gr.Text(label="Images folder to caption", value="", interactive=True, placeholder="Images folder", scale=4)
            open_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
            img_exts = gr.Text(label="Image file type (separated by ,)", value="jpg,png,webp", interactive=True,)
        with gr.Row():
            prefix_text = gr.Text(placeholder="Prefix to add to caption (Optional)", show_label=False)
            postfix_text = gr.Text(placeholder="Postfix to add to caption (Optional)", show_label=False)
            caption_ext = gr.Text(placeholder="Caption file extension, default .txt", show_label=False)
        with gr.Row():
            batch_button = gr.Button("Caption images")
            interrupt_button = gr.Button('Interrupt', visible=True)
            
    with gr.Group():
        with gr.Row():
            info = Ci_op.device_info()
            device_info_box = gr.Textbox(label="Device Info:",interactive=False,value=info, scale=3)
            unload_button = gr.Button("Unload")
            
    with gr.Group():
        with gr.Row():
            analyze_button = gr.Button("Analyze features")
        with gr.Row():
            medium = gr.Label(label="Medium", num_top_classes=5)
            artist = gr.Label(label="Artist", num_top_classes=5)        
            movement = gr.Label(label="Movement", num_top_classes=5)
            trending = gr.Label(label="Trending", num_top_classes=5)
            flavor = gr.Label(label="Flavor", num_top_classes=5)
            
    with gr.Accordion("About Note", open=False):
        gr.Markdown(
            "CLIP models:\n"
            "* For best prompts with Stable Diffusion 1.* choose the **ViT-L-14/openai** model.\n"
            "* For best prompts with Stable Diffusion 2.* choose the **ViT-H-14/laion2b_s32b_b79k** model.\n"
            "* For best prompts with Stable Diffusion XL choose **ViT-L-14/openai** or **ViT-bigG-14/laion2b_s39b** model.\n"
            "\nOther:\n"
            "* When you are done click the **Unload** button to free up memory."
        )    

    # execute
    prompt_button.click(Ci_op.image_to_prompt, 
                        inputs=[image, mode, clip_model, blip_model, caption_max_length, chunk_size, flavor_intermediate_count, prefix_text, postfix_text], 
                        outputs=[prompt, device_info_box],)
    
    open_folder_button.click(op_tinker.file_browser, 
                             inputs=data_type, 
                             outputs=batch_folder, 
                             show_progress="hidden")
    
    batch_button.click(Ci_op.batch_process, 
                        inputs=[batch_folder, mode, clip_model, blip_model, caption_max_length, chunk_size, flavor_intermediate_count, prefix_text, postfix_text, img_exts, caption_ext],
                        outputs=device_info_box,
                        )
    
    unload_button.click(Ci_op.unload,
                        inputs=None,
                        outputs=device_info_box,
                        )
    
    analyze_button.click(Ci_op.image_analysis, 
                         inputs=[image, clip_model], 
                         outputs=[medium, artist, movement, trending, flavor],)
    
    
def ui():
    with gr.Blocks() as ui:
        prompt_tab()
        # with gr.Tab('Prompt'):
        #     prompt_tab()
        # with gr.Tab('Analyze'):
        #     analyze_tab()
        # with gr.Tab('About'):
        #     about_tab()


               
    return ui