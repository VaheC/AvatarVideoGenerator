# AvatarVideoGenerator
The aim of the repository is to create an app which will allow to generate talking avatar videos.

The general pipeline of the app is represented in the image below.

![image](avatar_video_generation_pipeline.jpg)

## Websites providing similar service

Before working on the project I visited some websites that provide the similar service. Here are them: [Runway](https://runwayml.com/), [Nighcafe Creator](https://creator.nightcafe.studio/), and [Visper](https://visper.tech/en).

Of course, all of the websites provide a great content and services. Not surprisingly the options available for free users are not plausible in terms of the generated video's duration. Specifically, Runway allows to generate videos with duration of 4 seconds if you are a free user. Though Visper allows free users to create videos that last up to 20 seconds, it is also not a great deal. Finally Nightcafe Creator doesn't provide video generation service. But they have a service which generates image from text. 

## Detailed description of the pipeline

At first, when a user visits the app, she/he needs to provide an image of an avatar that will be talking in the video. There are 3 options to do this. The first one is to expand a bar called "Available avatars" to see the default options available in the app and select a avatar using "Please select an avatar" dropdown list. The second option is to upload an avatar of her/his choice using "Please upload an avatar" upload button. PLEASE REMEMBER TO REMOVE THE UPLOADED IMAGE IF YOU ARE GOING TO USE THE OTHER OPTIONS AFTER TRYING THE SECOND OPTION (there will be "cross" sign available next to the uploaded image in the app, you can use it to remove the image). Finally the third option is to generate an avatar using stable-diffusion model by stabilityai which is available publicly on huggingface. She/he needs to provide a text for prompt in the "Please type a prompt to generate an image for the avatar" text area and to click on "generate_avatar" button. It takes around 5 minutes to generate an avatar using the last option. PLEASE REMEMBER THAT AN AVATAR IMAGE SHOULD CONTAIN A FACE IN IT, OTHERWISE THE APP WILL FAIL.
<br>
In the second step the user should create an audio file that will be reproduced in the video. There are 2 ways to do this. The first one is to select a voice audio and to provide text, then to generate the audio using "Generate audio from the text" button ([coqui TTS](https://github.com/coqui-ai/TTS) model is used). There are 2 options to provide the voice audio: the user can either select it from the default available voices by expanding "Available voice audio" and selecting one from "Please select a voice to clone" dropdown list or upload an audio that contains the desired voice (PLEASE REMEMBER TO REMOVE THE UPLOADED AUDIO IF YOU ARE GOING TO USE THE OTHER OPTION AFTER TRYING THIS ONE: there will be "cross" sign available next to the uploaded audio in the app, you can use it to remove the audio). The time of audio generation depends on the length of the text provided: approximately it takes 0.7 seconds to generate 1 second of audio. The second option is to upload the audio, that will be reproduced in the video, using "Please, upload an audio" upload button. PLEASE REMEMBER TO REMOVE THE UPLOADED AUDIO IF YOU ARE GOING TO USE THE OTHER OPTION AFTER TRYING THIS ONE (there will be "cross" sign available next to the uploaded audio in the app, you can use it to remove the audio).
<br>
The third step is to generate the video using "Generate video" button ([Wav2Lip](https://github.com/Rudrabha/Wav2Lip) model is used). Depending on the length of the audio to reproduce the video generation time varies: it takes around 4 seconds to generate 1 second of video.
The all steps above are shown the following [video]().

## Models used 

There are 3 open source models used in the app: [stable-diffusion by stabilityai](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), [coqui TTS](https://github.com/coqui-ai/TTS), and [Wav2Lip](https://github.com/Rudrabha/Wav2Lip). 
<br>
TTS can be installed only using python versions >= 3.9 and Wav2Lip works with python 3.6: the packages are completely incompatible. This forced me to download the models from their repositories and save them as a local package. Moreover I downloaded the pretrained weights and saved them too in order to avoid downloading them everytime the app starts: this is crucial especially for deployment as for local use the weights are saved automatically in case of TTS and diffusion model. Below I will show how to save the weights for each model.
<br>
For TTS use the following
```python
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
```

Then go to the directory where "tts_models--multilingual--multi-dataset--your_tts" folder has been saved: it contains the all necessary files. Copy the files to the folder called tts_model.

<br>
For Wav2Lip visit the its [git repository](https://github.com/Rudrabha/Wav2Lip) and follow the instructions on how to download the pretrained weigths for face detection. Then rename the file to s3fd.pth and save it inside "Wav2Lip\face_detection\detection\sfd" folder. In the repository the link to the Wav2Lip pretrained model's weights is given too. So download it and paste inside wav2lip_model folder.

<br>
For diffusion model use the following.
```python
import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
```

In my case it created "C:\Users\vchar\.cache\huggingface\hub" inside which there is another folder containing all the files related to the model. Copy all the files from that folder and paste inside the diffusion_model folder.

