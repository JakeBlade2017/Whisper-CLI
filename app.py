# Obviously here goes the library
import gradio as gr
import subprocess
import os

# Here we check if GPU is present
import torch

# This function runs the commands as you do in cmd: whisper audio.mp3 --model turbo
def run_whisper(audio_file, model, device, output_format, output_dir, verbose, fp16, word_timestamps, max_line_count):
    os.makedirs(output_dir, exist_ok=True)

    #Commands that program will puse like you do in cmd: whisper audio_file --model turbo --device cuda, etc, etc
    command = [
        "whisper",
        audio_file,
        "--model", model,
        "--device", device,
        "--output_format", output_format,
        "--output_dir", output_dir,
        "--verbose", verbose,
        "--fp16", fp16,
        "--word_timestamps", word_timestamps,
        "--max_line_count", str(max_line_count),
    ]

    #Handles error handling
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return f"Transcription successful! Files saved in: {output_dir}\n\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error occurred:\n{e.stderr}"

# Here goes the gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Whisper")

    with gr.Row():
        audio_file = gr.Audio(sources='upload', label='Upload your audio file', show_download_button=False, show_share_button=False, type="filepath")

    #Transcribe settings
    with gr.Tab("Transcription"):
        model = gr.Dropdown(
            choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
            value='turbo',
            label='Model',
            info='Choose what model will use whisper, if is not present Whisper will download it'
        )

        #This detects if devices has GPU present, if not it will use the CPU instead
        def get_device_options():
            if torch.cuda.is_available():
                return gr.update(visible=True), gr.update(visible=False)#, gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=True)#, gr.update(visible=False)

        #Based on last lines, this is intended if user choices CPU even if user has GPU available
        def toggle_fp16(device_choice):
            if device_choice == "GPU":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        # This is now a unified device selector
        device = gr.Radio(
            choices=['cpu', 'cuda'],
            value='cuda' if torch.cuda.is_available() else 'CPU',
            label='Device',
            info='Choose what device you will use. GPU is faster than CPU.',
        )

        output_format = gr.CheckboxGroup(
            choices=['txt', 'vtt', 'srt', 'tsv', 'json', 'all'],
            value='all',
            label='Output format',
            info='Choose what file format do you want to be exported the transcription, choose "all" if you want to export into all options available',
        )

        output_dir = gr.Textbox(
            placeholder="Enter output directory path (e.g., ./output)",
            label="Output Directory",
            value="./output"
        )

    #Transcribe settings
    with gr.Tab("Settings"):
        verbose = gr.Radio(
            choices=['True', 'False'],
            value='True',
            label='Verbose',
            info='This is for debugging or view the progress (default is: True)'
        )

        fp16 = gr.Radio(
            choices=["True", "False"],
            value="True",
            label="Use fp16",
            info="Wether to perform inference in fp16 (only GPU)",
            visible=False
        )

        #Runs on startup
        demo.load(fn=get_device_options, outputs=[device, fp16])

        #Trigger on device change
        device.change(fn=toggle_fp16, inputs=device, outputs=fp16)

        #This toggles the max line count, only is visible when word_timestamps is True
        def toggle_max_line_count(enabled):
            return gr.update(visible=True)

        word_timestamps = gr.Radio(
            choices=["True", "False"],
            value="False",
            label="Word-level timestamps",
            info="Extract word-level timestamps and refine the results based on them"
        )

        max_line_count = gr.Slider(
            label="Max line count",
            minimum=0,
            maximum=10,
            value=0,
            step=1,
            visible=False,
            info="The maximum number of lines in a segment"
        )

        #Triggers when word_timestamps (input) changes
        word_timestamps.change(fn=toggle_max_line_count, inputs=word_timestamps, outputs=max_line_count)

    #Output parameters
    with gr.Tab("Output"):
        output_text = gr.Textbox(label="Result", lines=10)

    submit_btn = gr.Button("Transcribe")

    submit_btn.click(
        run_whisper,
        inputs=[
            audio_file,
            model,
            device,
            output_format,
            output_dir,
            verbose,
            fp16,
            word_timestamps,
            max_line_count
        ],
        outputs=output_text
    )

demo.launch()