import modules.scripts as scripts
import modules.images
from pathlib import Path
from modules import script_callbacks
import gradio as gr
from PIL import Image
import torch
import cv2
import scripts.makeframeutils as mfu
import numpy as np

mfdir = Path.joinpath(Path.cwd(), "Outputs", "MakeFrame")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
types_vid = ['.mp4', '.mkv', '.avi', '.ogv', '.ogg', '.webm']
types_gif = ['.gif', '.webp', '.apng', '.tiff']
types_all = types_vid+types_gif
types_out = ['.mp4', '.mkv', '.avi', '.gif', '.webp', '.apng', '.tiff']

def on_ui_tabs():
    with gr.Blocks() as makeframe_ui:
        # Top button row
        mf_working_path = gr.Textbox(value=mfdir, label="Working directory")
        with gr.Row(): 
            with gr.Box():
                with gr.Column():
                    gr.HTML(value="Break animation to frames")
                    upload_anim = gr.File(label="Upload Animation", file_types = types_all, live=True, file_count = "single")
                    break_anim = gr.Button("Break", visible= False)
                    break_status = gr.HTML(None, visible=False)
                    with gr.Box():
                        loaded_width = gr.Number(value=0, interactive = False, label = "Width")
                        loaded_height = gr.Number(value=0, interactive = False, label = "Height")
                        loaded_fps = gr.Number(value=0, interactive = False, label = "FPS")
                        loaded_runtime = gr.Number(value=0, interactive = False, label = "Runtime")
                        loaded_frames = gr.Number(value=0, interactive = False, label = "Total frames")
            with gr.Box():
                with gr.Column():
                    gr.HTML(value="Keyframe sheet operations")
                    with gr.Tabs():
                        with gr.Tab("Make Keyframe Grid"):
                            with gr.Box():
                                makegrid_input_path = gr.Textbox(label="Input directory")
                                with gr.Row():
                                    with gr.Column(min_width=32):
                                        makegrid_rows = gr.Slider(2, 20, 8, step=1, label="Grid rows", interactive=True)
                                        makegrid_cols = gr.Slider(2, 20, 8, step=1, label="Grid columns", interactive=True)
                                    with gr.Column(min_width=32):
                                        makegrid_maxwidth = gr.Slider(64, 4096, 2048, step=8, label="Maximum generation width", interactive=True, elem_id="maxwidth")
                                        makegrid_maxheight = gr.Slider(64, 4096, 2048, step=8, label="Maximum generation height", interactive=True, elem_id="maxheight")
                                with gr.Row():
                                    makegrid_button = gr.Button("Make grid", visible= True)
                                makegrid_output_gallery = gr.Gallery()
                                makegrid_status = gr.HTML(None, visible=True)
                        with gr.Tab("Break Keyframe grid"):
                            with gr.Box():
                                breakgrid_input_image = gr.File(label="Upload grid", file_types = ['.png', '.jpg'], live=True, file_count = "single")
                                with gr.Row():
                                    with gr.Column():
                                        breakgrid_rows = gr.Slider(2, 20, 8, step=1, label="Grid rows", interactive=True)
                                        breakgrid_cols = gr.Slider(2, 20, 8, step=1, label="Grid columns", interactive=True)
                                break_grid = gr.Button("Break", visible= True)
                                break_grid_status = gr.HTML(None, visible=False)
                        with gr.Tab("Options"):
                            with gr.Box():
                                with gr.Row():
                                    label_size = gr.Number(72, precision=1, label="Grid label font size")
                                    label_color = gr.ColorPicker("#ffffff", label="Grid label font color")
                                    use_histogram = gr.Checkbox(False, label="Use histograms for scene changes")
                                
            with gr.Column():
                gr.HTML(value="Make animation from frames")
                with gr.Box():
                    makeanim_input_path = gr.Textbox(label="Input directory")
                    makeanim_width = gr.Number(128, precision=1, label="Output width")
                    makeanim_height = gr.Number(128, precision=1, label="Output height")
                    makeanim_fps = gr.Number(15.0, label="Output FPS")
                    makeanim_ext = gr.Dropdown(types_out, value='.gif', label="Extension")
                    makeanim_button = gr.Button("Make Animation")
                    makeanim_status = gr.HTML(None, visible=True)

        #funcs
        def BreakFrames(filepath, output_dir):
            # Check or create root directory:
            try:
                if not Path.exists(Path(output_dir)):
                    Path.mkdir(Path(output_dir))
            except:
                st_out = f"Error: could not locate nor create directory {output_dir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Create the output subdirectory if it doesn't exist
            output_breaksdir = Path.joinpath(Path(output_dir), "Breaks")
            output_subdir = Path.joinpath(output_breaksdir, Path(filepath.name).stem)
            try:
                if not Path.exists(output_breaksdir):
                    Path.mkdir(output_breaksdir)
                if not Path.exists(output_subdir):
                    Path.mkdir(output_subdir)
            except:
                st_out = f"Error: could not create directory {output_subdir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Open the video file
            try:
                video_capture = cv2.VideoCapture(filepath.name)
                # Check if the anim was loaded
                if not video_capture.isOpened():
                    st_out = f"Error: Could not open video file {filepath.name}."
                    print(st_out)
                    return gr.update(value = st_out, visible = True)
            except:
                st_out = f"Error: Could not open video file {filepath.name}."
                print(st_out)
                return gr.update(value = st_out, visible = True)

            # Read each frame from anim
            frame_count = 0
            while True:
                ret, frame = video_capture.read()
                if ret:
                    output_file_name = f"frame_{str(frame_count).zfill(5)}.png"  # e.g., frame_00001.png
                    output_file_path = str(Path.joinpath(output_subdir, output_file_name))
                    cv2.imwrite(output_file_path, frame)
                    frame_count += 1
                else:
                    break
            video_capture.release()
            st_out = f"{frame_count} frames written to {output_subdir}."
            print(st_out)
            return gr.update(value = st_out, visible = True)
        
        def ProcessAnimUpload(filepath):
            filepath = filepath.name
            cap = cv2.VideoCapture(filepath)

            if not cap.isOpened():
                print(f"Error: Could not open video file {filepath}.")
                return None, None, None, None, None, gr.update(visible = False), gr.update(visible = False)

            # Get the properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps > 0:  # no /0
                runtime_seconds = frame_count / fps
            else:
                runtime_seconds = 0

            cap.release()

            return width, height, fps, runtime_seconds, frame_count, gr.update(visible = True), gr.update(value = "Ready to break.", visible = True)
        
        def ClearAnimUpload():
            return None, None, None, None, None, gr.update(visible = False), gr.update(visible = False)
        
        def MakeKeyFrameGrid(input_path, output_dir, rows, cols, maxw, maxh, fontsize, fontcolor, usehistogram):
            # Check input path
            try:
                input_path = Path(input_path)
            except:
                st_out = f"Could not locate path {input_path}."
                print(st_out)
                return st_out, None
            
            # Try gather files
            try:
                filepaths = list(input_path.iterdir())
            except:
                st_out = f"Could not locate files in {input_path}."
                print(st_out)
                return st_out, None
            
            # Check or create root directory:
            try:
                if not Path.exists(Path(output_dir)):
                    Path.mkdir(Path(output_dir))
            except:
                st_out = f"Error: could not locate nor create directory {output_dir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Create the output subdirectory if it doesn't exist
            output_subdir = Path.joinpath(Path(output_dir), "Grids")
            try:
                if not Path.exists(output_subdir):
                    Path.mkdir(output_subdir)
            except:
                st_out = f"Error: could not create directory {output_subdir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Make tensors
            image_tensors = [mfu.load_and_preprocess(path) for path in filepaths]
            N = np.clip((rows*cols), 2, len(image_tensors)-1) # Number of scene changes to find. Rows X Columns with extra space for first frame.
            if usehistogram:
                histograms = [mfu.compute_histogram(tensor) for tensor in image_tensors]
                differences = [torch.norm(histograms[i + 1] - histograms[i], p=1) for i in range(len(histograms) - 1)]
            else:
                differences = [torch.norm(image_tensors[i+1] - image_tensors[i], p=2) for i in range(len(image_tensors)-1)]
            _, top_indices = torch.topk(torch.tensor(differences), k=N, largest=True)
            keyframe_indices = sorted([index.item() + 1 for index in top_indices])
            keyframe_indices.insert(0, 0)

            # Pad the list with extras; black space frustrates generation
            if len(keyframe_indices) < rows*cols:
                keyframe_indices = mfu.padlist(keyframe_indices, rows*cols)

            grids_out = []
            # Size needs to be divisible by 8 AND the grid dimension
            if not rows == 8: maxh = mfu.closest_lcm(maxh, 8, rows)
            if not cols == 8: maxw = mfu.closest_lcm(maxw, 8, cols)
            # Build base grid
            keyframes = [Image.open(filepaths[i]) for i in keyframe_indices]
            keyframes = mfu.normalize_size(keyframes) #normalize sizes to /8
            grid = mfu.constrain_image(mfu.MakeGrid(keyframes, rows, cols), maxw, maxh)
            gridpath = mfu.get_iterated_path(output_subdir, input_path.stem, extension='.png')
            grid.save(gridpath)
            grids_out.append(gridpath)

            # Build labeled grid
            keyframes_labeled = [mfu.ImgLabeler(frame, str(idx), size=fontsize, color=fontcolor) for frame, idx in zip(keyframes, keyframe_indices)]
            grid_labeled = mfu.constrain_image(mfu.MakeGrid(keyframes_labeled, rows, cols), maxw, maxh)
            grid_labeled_path = mfu.get_iterated_path(output_subdir, input_path.stem, extension='.jpg')
            grid_labeled.save(grid_labeled_path)
            grids_out.append(grid_labeled_path)
            
            st_out = "Success."
            return st_out, grids_out

        def BreakKeyFrameGrid(input_img, output_dir, rows, cols):
            # Check or create root directory:
            try:
                if not Path.exists(Path(output_dir)):
                    Path.mkdir(Path(output_dir))
            except:
                st_out = f"Error: could not locate nor create directory {output_dir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Create the output subdirectory if it doesn't exist
            output_breaksdir = Path.joinpath(Path(output_dir), "Keyframes")
            output_subdir = Path.joinpath(output_breaksdir, Path(input_img.name).stem)
            try:
                if not Path.exists(output_breaksdir):
                    Path.mkdir(output_breaksdir)
                if not Path.exists(output_subdir):
                    Path.mkdir(output_subdir)
            except:
                st_out = f"Error: could not create directory {output_subdir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Break
            brokenlist = mfu.BreakGrid(Image.open(input_img.name), rows=rows, cols=cols)
            for i, frame in enumerate(brokenlist):
                output_file_name = f"frame_{str(i).zfill(5)}.png"  # e.g., frame_00001.png
                output_file_path = str(Path.joinpath(output_subdir, output_file_name))
                frame.save(output_file_path)
            st_out = f"{len(brokenlist)+1} frames written to {output_subdir}."
            return st_out

        def MakeAnimation(input_path, output_dir, width, height, fps, ext):
            try:
                input_path = Path(input_path)
            except:
                st_out = f"Could not locate path {input_path}."
                print(st_out)
                return st_out, None
            
            # Try gather files
            try:
                filepaths = list(input_path.iterdir())
            except:
                st_out = f"Could not locate files in {input_path}."
                print(st_out)
                return st_out, None
            
            # Check or create root directory:
            try:
                if not Path.exists(Path(output_dir)):
                    Path.mkdir(Path(output_dir))
            except:
                st_out = f"Error: could not locate nor create directory {output_dir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # Create the output subdirectory if it doesn't exist
            output_subdir = Path.joinpath(Path(output_dir), "Animations")
            try:
                if not Path.exists(output_subdir):
                    Path.mkdir(output_subdir)
            except:
                st_out = f"Error: could not create directory {output_subdir}."
                print(st_out)
                return gr.update(value = st_out, visible = True)
            
            # If animated image format..
            if ext in types_gif:
                sized_imgs = [Image.open(i).resize((width, height), Image.Resampling.LANCZOS) for i in filepaths]
                output_path = mfu.get_iterated_path(output_subdir, input_path.stem, extension=ext)
                sized_imgs[0].save(output_path,
                    save_all = True, append_images = sized_imgs[1:], loop = 0,
                    optimize = False, duration = 1000/fps)
                st_out = f"Success: animation written to {output_path}."
                return st_out
            # If video format..
            else:
                output_path = mfu.get_iterated_path(output_subdir, input_path.stem, extension=ext)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # TODO might need to dynamically adjust
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for img_path in filepaths:
                    img_path = str(img_path)
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (width, height))

                    # Write
                    video_writer.write(img_resized)
                
                video_writer.release()
                st_out = f"Success: animation written to {output_path}."
                return st_out
    #triggers
        upload_anim.upload(fn=ProcessAnimUpload, inputs=[upload_anim], outputs=[loaded_width, loaded_height, loaded_fps, loaded_runtime, loaded_frames, break_anim, break_status])
        upload_anim.clear(fn=ClearAnimUpload, inputs=[], outputs=[loaded_width, loaded_height, loaded_fps, loaded_runtime, loaded_frames, break_anim, break_status])
        break_anim.click(fn=BreakFrames, inputs=[upload_anim, mf_working_path], outputs=[break_status])
        makegrid_button.click(fn=MakeKeyFrameGrid, inputs=[makegrid_input_path, mf_working_path, makegrid_rows, makegrid_cols, makegrid_maxwidth, makegrid_maxheight, label_size, label_color, use_histogram], outputs = [makegrid_status, makegrid_output_gallery])
        break_grid.click(fn=BreakKeyFrameGrid, inputs=[breakgrid_input_image, mf_working_path, breakgrid_rows, breakgrid_cols], outputs=[break_grid_status])
        makeanim_button.click(fn=MakeAnimation, inputs=[makeanim_input_path, mf_working_path, makeanim_width, makeanim_height, makeanim_fps, makeanim_ext], outputs=makeanim_status)
        return ((makeframe_ui, "MakeFrame", "makeframe_ui"),)

script_callbacks.on_ui_tabs(on_ui_tabs)