import gradio as gr
import subprocess
import os

def run_whisper(input_file, output_dir, output_format, model, language, device):
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        "whisper",
        input_file,
        "--model", model,
        "--output_format", output_format,
        "--language", language,
        "--word_timestamps", "True",
        "--max_line_width", "25",
        "--max_line_count", "2",
        "--output_dir", output_dir
    ]

    # Agregar opción de dispositivo correctamente
    if device == "GPU":
        command.extend(["--device", "cuda"])
    elif device == "CPU":
        command.extend(["--device", "cpu"])  # O simplemente omitir este parámetro

    try:
        subprocess.run(command, check=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.{output_format}")
        return output_file if os.path.exists(output_file) else "Error: Archivo no generado."
    except subprocess.CalledProcessError as e:
        return f"Error en Whisper: {e}"
    except FileNotFoundError:
        return "Whisper no encontrado. Asegúrate de que está instalado y en el PATH."

whsiper_cli_iface = gr.Interface(
    fn=run_whisper,
    inputs=[
        gr.Audio(type="filepath", label="Sube tu archivo de audio"),
        gr.Textbox(value="outputs", label="Carpeta de salida"),
        gr.Dropdown(["txt", "vtt", "srt", "tsv", "json"], value="srt", label="Formato de salida"),
        gr.Dropdown(["large-v3", "turbo", "small", "tiny"], value="turbo", label="Modelo a usar"),
        gr.Dropdown(["af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh"], value="en", label="Idioma del audio"),
        gr.Radio(["GPU", "CPU"], value="GPU", label="Dispositivo a usar")
    ],
    outputs=gr.File(label="Archivo generado"),
    title="Whisper CLI",
    description="Sube un archivo de audio y obtén la transcripción.",
    flagging_mode="never"
)

whsiper_cli_iface.launch()
