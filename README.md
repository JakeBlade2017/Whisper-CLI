# Whisper-CLI

A lightweight web-based interface for OpenAI’s Whisper (Python version), built with Gradio.
This UI provides essential controls for transcription all accessible through a web interface.

<img width="1913" height="888" alt="image" src="https://github.com/user-attachments/assets/fcd4ac5d-4555-4416-84c5-61dbbf52d604" />

# Instalation Guide
Before we get started, you wil need to install torch first, next you will se how: 

**1. Install torch with CUDA support** (NVIDIA only) \
`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118`

*Not sure if you installed torch+cuda correctly?* check down the section: [**Verify your torch+cuda installation**](#Verify-your-torch+cuda-installation).

 - **If you want to install the cpu version then simply run** \
   `pip install torch==2.6.0`

**2. Run this command to install everything else** \
`pip install -r requirements.txt`

**3. Once everything is installed, just run** \
`python app.py`

# Verify your torch+cuda installation
Here we will make sure that you installed correctly torch with CUDA support
1. Open your terminal and start python \
  `python`
2. Import torch module \
  `import torch`
3. Check if torch can use your GPU with this code\
  `torch.cuda.is_available()`

If the result is `True`, your installation is correct. If it is `False`, check that you have the NVIDIA and CUDA drivers installed correctly.
