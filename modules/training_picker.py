import subprocess
import platform
import math
import json
import sys
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter


def on_ui_tabs():

    # fixed_size = int(opts.training_picker_fixed_size)
    # videos_path = Path(opts.training_picker_videos_path)
    # framesets_path = Path(opts.training_picker_framesets_path)
    # default_output_path = Path(opts.training_picker_default_output_path)

    # for p in [videos_path, framesets_path]:
    #     os.makedirs(p, exist_ok=True)

    # def get_videos_list():
    #     return list(v.name for v in videos_path.iterdir() if v.suffix in [".mp4", ".mkv"])

    # def get_framesets_list():
    #     return list(v.name for v in framesets_path.iterdir() if v.is_dir())

    with gr.Blocks(analytics_enabled=False) as training_picker:
        # videos_list = get_videos_list()
        # framesets_list = get_framesets_list()
        # structure
        with gr.Row():
            with gr.Column():
                with gr.Row().style(equal_height=True):
                    gr.File(file_count="multiple", file_types=["text", ".json", ".csv"]),
                with gr.Row():
                    extract_keyframe_type = gr.Radio(choices=["Full", "I","B","P"], 
                                                  label="Keyframe Type ( pict_type )", 
                                                  interactive=True,
                                                  value="Full")
                with gr.Row():
                    scene_sensitivity_input = gr.Slider(0, 1, value=0.3, label="scene sensitivity", info="(0 = Disable)", interactive=True, step=0.01)
                    blur_sensitivity_input = gr.Slider(0, 16, value=6, label="blur sensitivity", info="(if < 6 will very slow)", interactive=True, step=1)
                    frame_step_input = gr.Slider(1, 100, value=1, label="Extract every nth frame", info="(1 = Perframe)", interactive=True, step=1)
                with gr.Row():
                    extract_frames_button = gr.Button(value="Extract Frames", variant="primary")
                    overwrite_exist_checkbox = gr.Checkbox(value=True, label="Overwrite exist files",interactive=True)
                    
                log_output = gr.HTML(value="")
            with gr.Column():
                with gr.Row():
                    frameset_dropdown = gr.Dropdown(choices=["a","b"], elem_id="frameset_dropdown", label="Extracted Frame Set", interactive=True)
        
        with gr.Row():
            with gr.Column():
                crop_preview = gr.Image(interactive=False, elem_id="crop_preview", show_label=False)
            with gr.Column():
                frame_browser = gr.Image(interactive=False, elem_id="frame_browser", show_label=False)
                with gr.Row():
                    prev_button = gr.Button(value="<", elem_id="prev_button")
                    with gr.Row():
                        frame_number = gr.Number(value=0, elem_id="frame_number", live=True, show_label=False)
                        frame_max = gr.HTML(value="", elem_id="frame_max")
                    next_button = gr.Button(value=">", elem_id="next_button")
        
        # invisible elements
        crop_button = gr.Button(elem_id="crop_button", visible=False)
        crop_parameters = gr.Text(elem_id="crop_parameters", visible=False)
        

        # events
        def extract_frames_button_click(video_file, extract_keyframe, frame_step, scene_sensitivity, blur_sensitivity, overwrite_exist):
            input_path = videos_path / video_file
            output_path = framesets_path / Path(video_file).stem
            if not overwrite_exist:
                if output_path.is_dir():
                    print("Directory already exists!")
                    return gr.update(), f"Frame set already exists at {output_path}! Delete the folder first if you would like to recreate it."
            os.makedirs(output_path, exist_ok=True)
            output_name_fmat = r"%s\%s_%s" % (output_path, Path(video_file).stem, "%06d.png")
            command = ""
            v_filter = "-vf "
            v_select = "select=not(mod(n\,%s))" % frame_step
            ffmpeg_bin = "%s/ffmpeg.exe" % os.path.dirname(__file__)
            if os.path.isfile(ffmpeg_bin):
                if extract_keyframe != "Full":
                    v_select += "*eq(pict_type\,%s)" % extract_keyframe
                if scene_sensitivity != 0:
                    v_select += "*gt(scene\,%s)" % scene_sensitivity
                if blur_sensitivity != 0:
                    v_select += ",blurdetect=block_width=64:block_height=64:block_pct=80,metadata=select:key=lavfi.blur:value=%s:function=less" % blur_sensitivity
                    
            # command = r'%s -i %s -vf "%s" -vsync 0 %s\%s_%s' % (ffmpeg_bin, input_path, v_select, output_path, Path(video_file).stem, "%06d.png")
            
            subprocess.run([ffmpeg_bin, "-i", input_path, "-vf", v_select, "-vsync", "vfr", output_name_fmat])
            
            return gr.update(), command
                


            
        # collect inputs/outputs
        inputs = [video_dropdown, extract_keyframe_type, frame_step_input, scene_sensitivity_input, blur_sensitivity_input, overwrite_exist_checkbox]
        extract_frames_button.click(fn=extract_frames_button_click, inputs=inputs, outputs=[frameset_dropdown, log_output])

        def get_image_update():
            global current_frame_set_index
            global current_frame_set
            return gr.Image.update(value=current_frame_set[current_frame_set_index].get()), current_frame_set_index+1, f"/{len(current_frame_set)}"

        def null_image_update():
            return gr.update(), 0, ""

        def frameset_dropdown_change(frameset):
            global current_frame_set_index
            global current_frame_set
            current_frame_set_index = 0
            full_path = framesets_path / frameset
            current_frame_set = [CachedImage(impath) for impath in full_path.iterdir() if impath.suffix in [".png", ".jpg"]]
            try: current_frame_set = sorted(current_frame_set, key=lambda f:int(re.match(r"^(\d+).*", f.path.name).group(1)))
            except Exception as e: print(f"Unable to sort frames: {e}")
            return get_image_update()
        frameset_dropdown.change(fn=frameset_dropdown_change, inputs=[frameset_dropdown], outputs=[frame_browser, frame_number, frame_max])

        def prev_button_click():
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != []:
                current_frame_set_index = (current_frame_set_index - 1) % len(current_frame_set)
                return get_image_update()
            return null_image_update()
        prev_button.click(fn=prev_button_click, inputs=[], outputs=[frame_browser, frame_number, frame_max])

        def next_button_click():
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != []:
                current_frame_set_index = (current_frame_set_index + 1) % len(current_frame_set)
                return get_image_update()
            return null_image_update()
        next_button.click(fn=next_button_click, inputs=[], outputs=[frame_browser, frame_number, frame_max])

        def frame_number_change(frame_number):
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != []:
                current_frame_set_index = int(min(max(0, frame_number - 1), len(current_frame_set) - 1))
                return get_image_update()
            return null_image_update()
        frame_number.change(fn=frame_number_change, inputs=[frame_number], outputs=[frame_browser, frame_number, frame_max])

        def process_image(image, should_resize, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters, square_original):
            w, h = image.size
            if should_resize:
                ratio = fixed_size / max(w, h)
                image = image.resize((math.ceil(w * ratio), math.ceil(h * ratio)))
                if square_original:
                    square_original = square_original.resize((fixed_size - 1, fixed_size - 1)) # i would prefer to resize to the exact fixed size but a sliver of unblurred image appears otherwise in the final result :/
            if outfill_setting != "Don't outfill":
                image = outfill_methods[outfill_setting](image, color=outfill_color, blur=outfill_border_blur, n_clusters=outfill_n_clusters, original=square_original)
            return image

        def get_squared_original(full_im, bounds, outfill_method):
            x1, y1, x2, y2 = bounds
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max(w, h) // 2
            iw, ih = full_im.size
            outrad = max(iw, ih)
            dim_override = (int(outrad*2), int(outrad*2))
            ox, oy = (0, 0) if outfill_method == "Black outfill" else (outrad // 2 + (outrad - iw) // 2, outrad // 2 + (outrad - ih) // 2)
            new_bounds = (cx - r + ox, cy - r + oy, cx + r + ox, cy + r + oy)
            if outfill_method == "Stretch pixels at border":
                full_im = border_stretch(full_im, blur=0, dim_override=dim_override, axis_override=0)
                full_im = border_stretch(full_im, blur=0, dim_override=dim_override, axis_override=1)
            elif outfill_method == "Reflect image around border":
                full_im = reflect(full_im, blur=0, dim_override=dim_override, axis_override=0)
                full_im = reflect(full_im, blur=0, dim_override=dim_override, axis_override=1)
            return full_im.crop(new_bounds)

        def crop_button_click(raw_params, frame_browser, should_resize, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters, outfill_original_image_outfill_setting):
            params = json.loads(raw_params)
            im = Image.fromarray(frame_browser)
            crop_boundary = (params['x1'], params['y1'], params['x2'], params['y2'])
            cropped = im.crop(crop_boundary)
            if outfill_setting == "Reuse original image":
                square_original = get_squared_original(im, crop_boundary, outfill_original_image_outfill_setting)
            else:
                square_original = None
            cropped = process_image(cropped, should_resize, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters, square_original)
            save_path = Path(output_dir)
            os.makedirs(str(save_path.resolve()), exist_ok=True)
            current_images = [r for r in (re.match(r"(\d+).png", f.name) for f in save_path.iterdir()) if r]
            if current_images == []:
                next_image_num = 0
            else:
                next_image_num = 1 + max(int(r.group(1)) for r in current_images)
            filename = save_path / f"{next_image_num}.png"
            cropped.save(filename)
            return gr.Image.update(value=cropped), f"Saved to {filename}"
        crop_button.click(fn=crop_button_click, inputs=[crop_parameters, frame_browser, resize_checkbox, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters, outfill_original_image_outfill_setting], outputs=[crop_preview, log_output])

        def bulk_process_button_click(frameset, should_resize, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters):
            if outfill_setting == "Reuse original image":
                return gr.Image.update(value="https://user-images.githubusercontent.com/2313721/200725535-d2aca52a-497f-4424-a2dd-200118f5ab66.png"), "what did you expect would happen with that outfill method"
            for frame in tqdm(list((framesets_path / frameset).iterdir())):
                if frame.suffix in [".png", ".jpg"]:
                    with Image.open(frame) as img:
                        img = process_image(img, should_resize, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters, None)
                        save_path = Path(output_dir)
                        os.makedirs(str(save_path.resolve()), exist_ok=True)
                        img.save(Path(output_dir) / frame.name)
            return gr.update(), f'Processed images saved to "{output_dir}"!'
        bulk_process_button.click(fn=bulk_process_button_click, inputs=[frameset_dropdown, resize_checkbox, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters], outputs=[crop_preview, log_output])

        def outfill_setting_change(outfill_setting): 
            outfill_outputs = [
                "outfill_setting_options",
                "color_container",
                "border_blur_container",
                "n_clusters_container",
                "original_image_outfill_setting_container"
            ]
            visibility_pairs = {
                "Solid color": [
                    "outfill_setting_options",
                    "color_container"
                ],
                "Blurred & stretched overlay" : [
                    "outfill_setting_options",
                    "border_blur_container"
                ],
                "Dominant image color": [
                    "outfill_setting_options",
                    "n_clusters_container"
                ],
                "Stretch pixels at border": [
                    "outfill_setting_options",
                    "border_blur_container"
                ],
                "Reflect image around border": [
                    "outfill_setting_options",
                    "border_blur_container"
                ],
                "Reuse original image": [
                    "outfill_setting_options",
                    "border_blur_container",
                    "original_image_outfill_setting_container"
                ]
            }
            return [gr.update(visible=(outfill_setting in visibility_pairs and o in visibility_pairs[outfill_setting])) for o in outfill_outputs]
        outfill_setting.change(fn=outfill_setting_change, inputs=[outfill_setting], outputs=[outfill_setting_options, color_container, border_blur_container, n_clusters_container, original_image_outfill_setting_container])

        reset_aspect_ratio_button.click(fn=None, _js="resetAspectRatio", inputs=[], outputs=[])

    return (training_picker, "Training Picker", "training_picker"),
