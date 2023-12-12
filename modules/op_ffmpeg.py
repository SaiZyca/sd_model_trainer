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


def extract_mov_frames(movie_file_path, target_folder_path, keyframe_type, scene_sensitivity, blur_sensitivity, frame_step, overwrite_exist, extract_format):
    
    # def ffmpeg
    ffmpeg_bin = "%s\\ffmpeg.exe" % os.path.dirname(__file__)
    v_filter = "-vf "
    v_select = "select=not(mod(n\,%s))" % frame_step
    
    for movie_file in list(movie_file_path.split(",")):
        command = None
    # def filename and path
        filename = Path(movie_file).stem
        output_folder = r"%s\%s" % (target_folder_path, filename)
        full_output_filename = r"%s\%s_%s%s" % (output_folder, filename, "%06d", extract_format)
    
        if not overwrite_exist:
            if output_folder.is_dir():
                print("Directory already exists!")
                return gr.update(), f"Frame set already exists at {output_folder}! Delete the folder first if you would like to recreate it."
        os.makedirs(output_folder, exist_ok=True)

        
        if os.path.isfile(ffmpeg_bin):
            if keyframe_type != "Full":
                v_select += "*eq(pict_type\,%s)" % keyframe_type
            if scene_sensitivity != 0:
                v_select += "*gt(scene\,%s)" % scene_sensitivity
            if blur_sensitivity != 0:
                v_select += ",blurdetect=block_width=64:block_height=64:block_pct=80,metadata=select:key=lavfi.blur:value=%s:function=less" % blur_sensitivity
        
        movie_file = os.path.normpath(movie_file)   
        full_output_filename = os.path.normpath(full_output_filename) 
        
        command = [ffmpeg_bin, "-i", movie_file, "-vf", v_select, "-vsync", "vfr", full_output_filename]
        
        
        subprocess.run(command)
        
        # print ("---- %s extracted finish ----" % movie_file)
    

