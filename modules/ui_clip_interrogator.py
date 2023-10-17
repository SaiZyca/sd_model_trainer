#!/usr/bin/env python3
import gradio as gr
from . import op_torch, symbols
from . import op_clip_interrogator as Ci_op
from . import op_tinker



def prompt_tab():
    with gr.Row():
        mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Mode', value='classic')
        clip_model = gr.Dropdown(Ci_op.list_clip_models(), value=Ci_op.ci.config.clip_model_name, label='CLIP Model')
        blip_model = gr.Dropdown(Ci_op.list_caption_models(), value=Ci_op.ci.config.caption_model_name, label='Caption Model')
        # clip_model = gr.Dropdown(Ci_op.list_clip_models(), value='ViT-L-14/openai', label='CLIP Model')
        # blip_model = gr.Dropdown(Ci_op.list_caption_models(), value='blip-large', label='Caption Model')
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
            batch_size = gr.Number(value=1, label="Batch size", minimum=1,step=1, interactive=True,)
            img_ext = gr.Text(label="Image file type (separated by ,)", value="jpg,png,webp", interactive=True,)
            caption_ext = gr.Text(label="Caption file extension", value=".txt", interactive=True)
        with gr.Row():
            prefix_text = gr.Text(placeholder="Prefix to add to caption (Optional)", show_label=False)
            postfix_text = gr.Text(placeholder="Postfix to add to caption (Optional)", show_label=False)
        with gr.Row():
            batch_button = gr.Button("Caption images")
            interrupt_button = gr.Button('Interrupt', visible=True)
    with gr.Group():
        with gr.Row():
            device_info = op_torch.cuda_device.info()
            info = ("%s:%s GB | Vram Allocated: %s GB | Vram Cached: %s GB " % (\
                        device_info['device'],\
                        device_info['Total Vram'],\
                        device_info['Vram Allocated'],\
                        device_info['Vram Cached'],)\
                            )
            gr.Textbox(label="Device Info:",interactive=False,value=info, scale=3)
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
    prompt_button.click(Ci_op.image_to_prompt, inputs=[image, mode, clip_model, blip_model], outputs=prompt)
    
    open_folder_button.click(op_tinker.file_browser, inputs=data_type, outputs=batch_folder, show_progress="hidden")
    batch_button.click(Ci_op.batch_process, 
                        inputs=[batch_folder, mode, clip_model, blip_model, prefix_text, postfix_text],
                        outputs=None,
                        )
    unload_button.click(Ci_op.unload)
    analyze_button.click(Ci_op.image_analysis, inputs=[image, clip_model], outputs=[medium, artist, movement, trending, flavor])
    
    
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