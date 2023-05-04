import gradio as gr
import shutil
import os
from . import common_func

def setup():
    os.environ.get('TORCH_COMMAND', "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")

def train_avatar_tab(setting_path=""):
    n_inputs = 5
    img_inps = list()
    text_inpt = list()
    
    with gr.Blocks() as ui:
        with gr.Row():
            gr.Markdown(
            """
            # 個性化描述
            輸入你的英文名字、年齡、特徵等等.
            """)
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    name_caption = gr.Textbox(label="Name", placeholder="Name",)
                    age_caption = gr.Number(value=20, label="Age", precision=0,interactive=True,)
                    gender_caption = gr.Radio(choices=["male", "female"], value="male")
            with gr.Column(scale=4):
                with gr.Row():
                    feature_caption = gr.Textbox(label="Feature", placeholder="Features, use , to div prompt",)
                    
            text_inpts = [age_caption, gender_caption, name_caption, feature_caption]
        with gr.Row():
            gr.Markdown(
            """
            # 頭部照片
            請上傳不同角度與表情的大頭照.
            """)
        with gr.Row():
            for i in range(n_inputs):
                with gr.Box():
                    with gr.Column():
                        img = gr.Image(label="Image", interactive=True, visible=True, type="pil")
                        img_inps.append(img)
        with gr.Row():
            gr.Markdown(
            """
            # 半身照
            請上傳不同角度腰部以上包含頭部的照片.
            """)
        with gr.Row():
            for i in range(n_inputs):
                with gr.Box():
                    with gr.Column():
                        img = gr.Image(label="Image", interactive=True, visible=True, type="pil")
                        img_inps.append(img)
        with gr.Row():
            gr.Markdown(
            """
            # 全身照
            請上傳不同角度與姿勢包含腳的全身照片.
            """)
        with gr.Row():
            for i in range(n_inputs):
                with gr.Box():
                    with gr.Column():
                        img = gr.Image(label="Image", interactive=True, visible=True, type="pil")
                        img_inps.append(img)
        with gr.Row() as clip_interrogator:
            clip_mode = gr.Radio(choices=["best", "fast", "classic", "negative"], value="best", interactive=True)
            caption_model = gr.Dropdown(choices=["blip-base", "blip-large", "git-large-coco"], value="blip-base", interactive=True)
            clip_model = gr.Dropdown(choices=["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"], value="ViT-L-14/openai", interactive=True)
            caption_settings = [clip_mode, caption_model, clip_model]
        with gr.Row():
            train_avatar = gr.Button(label="Train Model")
        with gr.Row():
            export_model = gr.File(label="Trained Model", interactive=False)
        with gr.Row():
            log_output = gr.HTML(value="")

    def execute(*args):
        project_folder, image_folder, data_set_folder, model_folder, log_folder = common_func.create_train_project(name=args[17])
        prefix_caption = "a %s-ages %s named %s, %s" % (args[15], args[16], args[17], args[18])
        for i in range(0, 5):
            image_path = r"%s\head_shot_%s.png" % (data_set_folder, i)
            if args[i] is not None:
                args[i].save(image_path)
        for i in range(5, 10):
            image_path = r"%s\upper_%s.png" % (data_set_folder, i)
            if args[i] is not None:
                args[i].save(image_path)
        for i in range(10, 15):
            image_path = r"%s\full_%s.png" % (data_set_folder, i)
            if args[i] is not None:
                args[i].save(image_path)
        
        common_func.interrogate_folder(image_folder, prefix_caption, args[19], args[20], args[21])
        common_func.generate_lora_toml(project_folder)
        lora_trainer_folder = r"C:\_Dev\Repository\lora-scripts"
        model_path = common_func.start_train_lora(lora_trainer_folder, project_folder)

        return prefix_caption, model_path

    train_avatar.click(fn=execute, inputs=(img_inps+text_inpts+caption_settings), outputs=[log_output, export_model], api_name="train")
    
    return ui