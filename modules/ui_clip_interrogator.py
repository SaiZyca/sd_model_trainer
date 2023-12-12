#!/usr/bin/env python3
import gradio as gr
from . import op_torch, symbols
from . import op_clip_interrogator as Ci_op
from . import op_tinker



def caption_image_ui():
    with gr.Group():
        with gr.Accordion("clip interrogator Config", open=True):
            with gr.Row():
                caption_max_length = gr.Number(value=32, label="caption max length", minimum=1,step=1, interactive=True,)  
                chunk_size = gr.Number(value=1024, label="chunk/batch size", minimum=1,step=1, interactive=True,)
                flavor_intermediate_count = gr.Number(value=2048, label="flavor count", minimum=1,step=1, interactive=True,)
                num_top_classes = gr.Number(label="top features count", value = 5, maximum=100, minimum=1, step=1)
            with gr.Row():
                mode = gr.Dropdown(['best', 'fast', 'classic', 'negative', 'caption', 'Manual'], value='caption', label='Mode')
                clip_model = gr.Dropdown(Ci_op.list_clip_models(), value=Ci_op.ci.config.clip_model_name, label='CLIP Model')
                blip_model = gr.Dropdown(Ci_op.list_caption_models(), value=Ci_op.ci.config.caption_model_name, label='Caption Model')
    with gr.Group():
        with gr.Row(equal_height=True):
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=6)
                caption_image_button = gr.Button("Caption Image")

    with gr.Group():
        with gr.Row():
            batch_folder = gr.Textbox(label="Images folder to caption", value="", interactive=True, placeholder="Images folder", scale=4)
            open_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
            img_exts = gr.Textbox(label="Image file ext (separated by ,)", value="jpg,png,webp", interactive=True,)
            filename_filter = gr.Textbox(label="filename filter", value="", interactive=True,)
        with gr.Row():
            prefix_text = gr.Textbox(placeholder="Prefix to add to caption (Optional)", show_label=False)
            postfix_text = gr.Textbox(placeholder="Postfix to add to caption (Optional)", show_label=False)
            filter_text = gr.Textbox(placeholder="filter caption if exist (Optional, separated by ,)", show_label=False)
            caption_ext = gr.Textbox(placeholder="Caption file extension, default .txt", show_label=False)
        with gr.Row():
            batch_caption_images_button = gr.Button("Batch Caption images in folder")
            interrupt_button = gr.Button('Interrupt', visible=True)

    with gr.Group():
        with gr.Row():
            analyze_button = gr.Button("Analyze features", visible=False)
        with gr.Row():
            top_mediums = gr.Textbox(label="Top Mediums", lines=6)
            top_artists = gr.Textbox(label="Top Artists", lines=6)        
            top_movements = gr.Textbox(label="Top Movements", lines=6)
            top_trendings = gr.Textbox(label="Top Trendings", lines=6)
            top_flavors = gr.Textbox(label="Top Flavors", lines=6)

    with gr.Group():
        with gr.Row():
            info = Ci_op.device_info()
            device_info_box = gr.Textbox(label="Device Info:",interactive=False,value=info, scale=3, show_label=False)
            unload_button = gr.Button("Unload Clip Model")
            

            
    with gr.Accordion("About Note", open=False):
        gr.Markdown(
            "CLIP models:\n"
            "* For best prompts with Stable Diffusion 1.* choose the **ViT-L-14/openai** model.\n"
            "* For best prompts with Stable Diffusion 2.* choose the **ViT-H-14/laion2b_s32b_b79k** model.\n"
            "* For best prompts with Stable Diffusion XL choose **ViT-L-14/openai** or **ViT-bigG-14/laion2b_s39b** model.\n"
        )    

    # execute
    caption_image_button.click(Ci_op.caption_image, 
                        inputs=[image, mode, clip_model, blip_model, 
                                caption_max_length, chunk_size, flavor_intermediate_count, num_top_classes, 
                                prefix_text, postfix_text, filter_text ], 
                        outputs=[prompt, device_info_box, top_mediums, top_artists, top_movements, top_trendings, top_flavors],
                        api_name="caption-image",)
    
    open_folder_button.click(op_tinker.folder_browser, 
                             inputs=[], 
                             outputs=batch_folder, 
                             show_progress="hidden")
    
    batch_caption_images_button.click(Ci_op.batch_caption_images, 
                        inputs=[batch_folder, mode, clip_model, blip_model, 
                                caption_max_length, chunk_size, flavor_intermediate_count, 
                                prefix_text, postfix_text, img_exts, caption_ext, filter_text, filename_filter],
                        outputs=device_info_box,
                        )
    
    interrupt_button.click(Ci_op.cancel_batch,
                           inputs=[],
                           outputs=[])
    
    unload_button.click(Ci_op.unload,
                        inputs=[],
                        outputs=device_info_box,
                        )
    
    analyze_button.click(Ci_op.image_analysis, 
                         inputs=[image, clip_model, num_top_classes], 
                         outputs=[top_mediums, top_artists, top_movements, top_trendings, top_flavors],)
    
    
def ui():
    with gr.Blocks() as ui:
        caption_image_ui()
        # with gr.Tab('Analyze'):
        #     analyze_tab()
        # with gr.Tab('About'):
        #     about_tab()


               
    return ui