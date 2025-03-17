# Imports the necessary modules
import gradio as gr
import subprocess
import os
import torch  # Import torch to check for GPU availability

# Global variable to save the actual progress (this is for stop button)
current_process = None

def toggle_language_visibility(task):
    # Show language dropdown only if the task is "transcribe"
    return gr.update(visible=task == "translate")

def run_whisper(input_file, output_dir, output_format, model_source, model, custom_model_path, language, device, w_time_stamp, max_line_width, max_line_count):
    global current_process
    os.makedirs(output_dir, exist_ok=True)

    # Validate custom model path
    if model_source == "Custom":
        if not custom_model_path:
            return "Error: Please provide a custom model path when using Custom model source."
        if not os.path.isfile(custom_model_path):
            return f"Error: Custom model file '{custom_model_path}' not found."

    # Determine model argument
    model_arg = custom_model_path if model_source == "Custom" else model

    # Gets textbox value
    max_line_width = max_line_width or "25"  # If value is empty, uses "25"
    max_line_count = max_line_count or "2"  # If value is empty, uses "2"

    # Commands used for CMD (background)
    command = [
        "whisper",
        input_file,
        "--model", model_arg,
        "--output_format", output_format,
        "--language", language, #when is None it just transcribe literally
        "--word_timestamps", "True" if w_time_stamp else "False", 
        "--max_line_width", max_line_width if w_time_stamp else "None",
        "--max_line_count", max_line_count if w_time_stamp else "None",
        "--output_dir", output_dir
    ]

    # Check if device have GPU or not
    if device == "GPU":
        command.extend(["--device", "cuda"])
    elif device == "CPU":
        command.extend(["--device", "cpu"])

    # This is for process
    try:
        # Popen is used to be able stop (or cancel) the process
        current_process = subprocess.Popen(command)
        current_process.wait()  # Waits until process ends

        # Manages the output file and its base name
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.{output_format}")
        return output_file if os.path.exists(output_file) else "File not generated or canceled."
    
    # Runs where there's an error with Whisper
    except subprocess.CalledProcessError as e:
        return f"Whisper error: {e}"
    # Runs when Whisper is not found
    except FileNotFoundError:
        return "Whisper not found. Make sure it's installed."
    # Cleans the process after it finished
    finally:
        current_process = None

def stop_whisper():
    global current_process
    if current_process:
        current_process.terminate()  # Stops the process
        return "Process stopped."
    return "There's no process in execution"

def open_directory(output_dir):
    os.system(f'open {output_dir}' if os.name == 'posix' else f'explorer {output_dir}')
    return "Open folder"

def toggle_timestamp_settings(w_time_stamp):
    return gr.update(visible=w_time_stamp)

def toggle_model_source(model_source):
    if model_source == "Predefined":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def check_gpu_availability():
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        # If GPU is available, show the device input and hide the GPU message
        return gr.update(visible=True), gr.update(visible=False)
    else:
        # If no GPU is available, hide the device input and show the GPU message
        return gr.update(visible=False), gr.update(value="No GPU found, using CPU instead", visible=True)

custom_css = """
.container {
    max-width: 65%;
    margin: 0 auto;
}
.title {
    text-align: center;
    display:block;
}
.desc {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=custom_css) as whisper_cli_iface:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Whisper CLI", elem_classes="title")
        gr.Markdown("Transcribe everything you want!", elem_classes="desc")

        audio_input = gr.Audio(type="filepath", label="Upload your audio file")
        output_file = gr.File(label="File generated")

        with gr.Tab("Settings"):
            with gr.Row():
                model_source = gr.Radio(["Predefined", "Custom"], label="Model Source", value="Predefined")
            with gr.Row():
                model_input = gr.Radio(["tiny", "base", "small", "medium", "large", "turbo"], value="turbo", label="Model", visible=True)
                custom_model_input = gr.Textbox(label="Custom Model Path", visible=False, placeholder="Path to .pt model file")

            with gr.Row():
                output_dir_input = gr.Textbox(value="output", label="Output folder")
                output_format_input = gr.Radio(["txt", "vtt", "srt", "tsv", "json", "all"], value="srt", label="Output format")
                device_input = gr.Radio(["GPU", "CPU"], value="GPU", label="Device", visible=False)
                gpu_message = gr.Textbox(label="GPU Status", visible=False)
                language_input = gr.Dropdown(["af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","yue","zh"], value="en", label="Language")

        with gr.Tab("Extra settings"):
            with gr.Row():
                w_time_stamp = gr.Checkbox(label="Use timestamps", value=True)
            with gr.Row(visible=True) as timestamp_options:
                max_line_width = gr.Textbox(label="Max line width", placeholder="25", value="25", interactive=True)
                max_line_count = gr.Textbox(label="Max line count", placeholder="2", value="2", interactive=True)

        with gr.Row():
            generate_button = gr.Button("Generate", scale=4)
            stop_button = gr.Button("Stop", scale=1)
            folder_button = gr.Button("Open Folder", scale=1)

        w_time_stamp.change(fn=toggle_timestamp_settings, inputs=w_time_stamp, outputs=timestamp_options)
        model_source.change(fn=toggle_model_source, inputs=model_source, outputs=[model_input, custom_model_input])

        # Check GPU availability on load
        whisper_cli_iface.load(fn=check_gpu_availability, outputs=[device_input, gpu_message])

        generate_button.click(
            fn=run_whisper,
            inputs=[
                audio_input, 
                output_dir_input, 
                output_format_input,
                model_source,
                model_input,
                custom_model_input,
                language_input,
                device_input,
                w_time_stamp, 
                max_line_width, 
                max_line_count
            ],
            outputs=output_file
        )
        
        stop_button.click(
            fn=stop_whisper,
            outputs=None
        )
        
        folder_button.click(
            fn=open_directory,
            inputs=output_dir_input,
        )

whisper_cli_iface.launch()