import gradio as gr
from . import op_tinker, symbols


def training_picker_tabs():
    with gr.Accordion("Movie Extractor", open=True):
        filetypes = gr.JSON(value={'video files':[("video files","*.avi;*.mp4"),]}, visible=False)
        with gr.Group():
            with gr.Row():
                movie_file_path = gr.Textbox(label="Movie file for extract", value="", interactive=True, placeholder="Movie file for extract", scale=1, show_label=False)
                movie_file_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
                target_folder_path = gr.Textbox(label="extract files to", value="", interactive=True, placeholder="extract files to", scale=1, show_label=False)
                target_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
            with gr.Row():
                extract_keyframe_type = gr.Radio(choices=["Full", "I","B","P"], label="Keyframe Type ( pict_type )", interactive=True,value="Full")
                scene_sensitivity_input = gr.Slider(0, 1, value=0.3, label="scene sensitivity", info="(0 = Disable)", interactive=True, step=0.01)
                blur_sensitivity_input = gr.Slider(0, 16, value=6, label="blur sensitivity", info="(if < 6 will very slow)", interactive=True, step=1)
                frame_step_input = gr.Slider(1, 100, value=1, label="Extract every nth frame", info="(1 = Perframe)", interactive=True, step=1)
            with gr.Row():
                overwrite_exist_checkbox = gr.Checkbox(value=True, label="Overwrite exist files",interactive=True)
            with gr.Row():
                extract_frames_button = gr.Button(value="Extract Frames", variant="primary")

        with gr.Group():
            with gr.Row():
                extracted_frame_Sets = gr.Dropdown(choices=["a","b"], elem_id="frameset_dropdown", label="Extracted Frame Set", interactive=True)
            
    movie_file_button.click(op_tinker.file_browser, 
                             inputs=[filetypes], 
                             outputs=movie_file_path, 
                             show_progress="hidden")
    
    target_folder_button.click(op_tinker.folder_browser, 
                            inputs=[], 
                            outputs=target_folder_path, 
                            show_progress="hidden")
def ui():
    with gr.Blocks() as ui:
        training_picker_tabs()