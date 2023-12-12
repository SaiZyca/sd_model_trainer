import gradio as gr
from . import op_tinker, op_ffmpeg, symbols


def movie_extractor_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("Movie Extractor", open=True):
            
            with gr.Group():
                with gr.Row():
                    ffmpeg_filter = gr.JSON(value={'extension filter':[("ffmpeg binary","*.exe"),]}, visible=False)
                    ffmpeg_bin_path = gr.Textbox(label="ffmpeg binary path", value=r"./ffmpeg.exe", interactive=True, placeholder="Movie file for extract", scale=1, show_label=True)
                    ffmpeg_bin_button = gr.Button(symbols.document_symbol, elem_id='open_folder_small')
                with gr.Row():
                    ext_filter = gr.JSON(value={'extension filter':[("video files","*.avi;*.mp4;*.mov;*.mkv"),]}, visible=False)
                    movie_file_path = gr.Textbox(label="Movie file for extract", value="", interactive=True, placeholder="Movie file for extract", scale=1, show_label=False)
                    movie_file_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
                    target_folder_path = gr.Textbox(label="extract files to", value="", interactive=True, placeholder="extract files to", scale=1, show_label=False)
                    target_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
                with gr.Row():
                    keyframe_type = gr.Radio(choices=["Full", "I","B","P"], label="Keyframe Type ( pict_type )", interactive=True,value="Full")
                    extract_format = gr.Radio(value=".png", choices=[".jpg",".png",".webp"], label="extract format",)
                with gr.Row():
                    scene_sensitivity = gr.Slider(0, 1, value=0.3, label="scene sensitivity", info="(0 = Disable)", interactive=True, step=0.01,)
                    blur_sensitivity = gr.Slider(0, 16, value=6, label="blur sensitivity", info="(if < 6 will very slow)", interactive=True, step=1,)
                    frame_step = gr.Slider(1, 100, value=1, label="Extract every nth frame", info="(1 = Perframe)", interactive=True, step=1,)
                with gr.Row():
                    overwrite_exist = gr.Checkbox(value=True, label="Overwrite exist files",interactive=True)
                    
                extract_frames_button = gr.Button(value="Extract Frames")

        ffmpeg_bin_button.click(op_tinker.file_browser, 
                                inputs=[ffmpeg_filter], 
                                outputs=ffmpeg_bin_path, 
                                show_progress="hidden")

        movie_file_button.click(op_tinker.file_browser, 
                                inputs=[ext_filter], 
                                outputs=movie_file_path, 
                                show_progress="hidden")
        
        target_folder_button.click(op_tinker.folder_browser, 
                                inputs=[], 
                                outputs=target_folder_path, 
                                show_progress="hidden")

        def test_fun(movie_file_path, target_folder_path, keyframe_type, scene_sensitivity, blur_sensitivity, frame_step, overwrite_exist, extract_format):
            print (movie_file_path)

        extract_frames_button.click(op_ffmpeg.extract_mov_frames,
                                    inputs=[movie_file_path, target_folder_path, keyframe_type, scene_sensitivity, blur_sensitivity, frame_step, overwrite_exist, extract_format],
                                    outputs=[],)

    return ui

def batch_process_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("Batch Process", open=True):
            with gr.Group():
                with gr.Row():
                    target_folder_path = gr.Textbox(label="image files folder", value="", interactive=True, placeholder="extract files to", scale=4, show_label=True)
                    target_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
                    img_exts = gr.Textbox(label="Image file ext (separated by ,)", value="jpg,png,webp", interactive=True,)
                    file_count = gr.Number(value=0, label="file count", minimum=1,step=1, interactive=False,) 
            with gr.Group():
                image_process = gr.CheckboxGroup(["Resize", "Crop", "Upscale"], label="Image Process")
                image_process_button = gr.Button(value="Process") #variant="primary"
                        
                        
        target_folder_button.click(op_tinker.get_image_folder, 
                                inputs=[img_exts], 
                                outputs=[target_folder_path, file_count], 
                                show_progress="hidden")
    
    return ui


def ui():
    with gr.Blocks() as ui:
        with gr.Row():
            movie_extractor_ui()
        with gr.Row():
            batch_process_ui()
        # with gr.Tab('Prompt'):
        #     prompt_tab()
               
    return ui