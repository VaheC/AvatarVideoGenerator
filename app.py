import streamlit as st 
import os
from glob import glob
from pathlib import Path

# from TTS.TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
from Wav2Lip.video_generator import create_video

gpu = False
model_path = Path(r"tts_model/model_file.pth")
config_path = Path(r"tts_model/config.json")
vocoder_path = None
vocoder_config_path = None
model_dir = None
language="en"
file_path="generated_audio.wav"
speaker = None
split_sentences = True
pipe_out = None

# def get_synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, model_dir, gpu):

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=None,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint=None,
    encoder_config=None,
    model_dir=model_dir,
    use_cuda=gpu,
)

# return synthesizer

# synthesizer = get_synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, model_dir, gpu)

def get_audio(synthesizer, speaker, language, speaker_wav, split_sentences, text):
    
    wav = synthesizer.tts(
        text=text,
        speaker_name=speaker,
        language_name=language,
        speaker_wav=speaker_wav,
        reference_wav=None,
        style_wav=None,
        style_text=None,
        reference_speaker_name=None,
        split_sentences=split_sentences
    )

    synthesizer.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)

# avatar_images_dir = Path('avatar_images')
avatar_images_list = os.listdir('avatar_images')
avatar_names_list = list(map(lambda x: x.split('.')[0], avatar_images_list))

n_cols_avatars = 3
n_rows_avatars = int((len(avatar_images_list) - len(avatar_images_list) % n_cols_avatars) / n_cols_avatars)
if len(avatar_images_list) % n_cols_avatars != 0:
    n_rows_avatars += 1

voice_audio_list = os.listdir('voice_audios')
voice_names_list = list(map(lambda x: x.split('.')[0], voice_audio_list))

n_cols_voices = 3
n_rows_voices = int((len(voice_audio_list) - len(voice_audio_list) % n_cols_voices) / n_cols_voices)
if len(voice_audio_list) % n_cols_voices != 0:
    n_rows_voices += 1

st.set_page_config(
    page_title='Avatar service',
    layout='wide'
)

st.markdown("<h1 style='text-align: center; color: white;'>Avatar video generation</h1>", unsafe_allow_html=True)

# st.title('Avatar video generation')

st.subheader('Step 1: Avatar Selection')

with st.expander('Available avatars'):
    n_images_shown = 0
    for i in range(n_rows_avatars):
        avatar_cols_list = st.columns(n_cols_avatars)
        for j in range(n_cols_avatars):
            avatar_cols_list[j].image(
                os.path.join('avatar_images', avatar_images_list[j+i*3]), 
                width=150,
                caption=avatar_names_list[j+i*3]
            )
            n_images_shown += 1
            if n_images_shown == len(avatar_images_list):
                break

def avatar_callback():
    if st.session_state.avatar_image:
        st.session_state.selected_avatar = st.session_state.avatar_image

def uploaded_avatar_callback():
    if st.session_state.uploaded_avatar_image is None:
        pass
    else:
        image_path = "uploaded_avatar_image" + \
            os.path.splitext(st.session_state.uploaded_avatar_image.name)[-1]
        with open(image_path, "wb") as f:
            f.write(st.session_state.uploaded_avatar_image.getvalue())

step1_col1, step1_col2 = st.columns(2)

with step1_col1:
    selected_avatar = st.selectbox(
        label='Please select an avatar',
        options=avatar_names_list,
        key='avatar_image',
        on_change=avatar_callback
    )

    st.write('or')

    uploaded_image = st.file_uploader(
        label='Please upload an avatar',
        type=['png', 'jpg', 'jpeg'],
        on_change=uploaded_avatar_callback,
        key='uploaded_avatar_image'
    )

with step1_col2:
    if uploaded_image is None:
        st.image(os.path.join('avatar_images', avatar_images_list[avatar_names_list.index(selected_avatar)]), width=300)
    else:
        uploaded_avatar_image_path = glob('uploaded_avatar_image.*')[0]
        st.image(uploaded_avatar_image_path, width=300)

st.subheader('Step 2: Audio Selection')
# st.markdown("<div title='Opa'>Option 1</div>", unsafe_allow_html=True)
option1_expander = st.expander('Option 1')
option1_expander.write(
    '''Please select or upload an audio with a voice you want to be used in the video.
    Then provide a text that will be used in the video. Afterwards click on 
    <Generate audio from text> button to get the audio which will be used in the video:
    please, take into account that depending on the size of the text it may take some time.
    '''
)

with st.expander('Available voice audio'):
    n_voices_shown = 0
    for i in range(n_rows_voices):
        voice_cols_list = st.columns(n_cols_voices)
        for j in range(n_cols_avatars):
            voice_cols_list[j].audio(
                os.path.join('voice_audios', voice_audio_list[j+i*3])
            )
            voice_cols_list[j].write(voice_names_list[j+i*3])
            n_voices_shown += 1
            if n_voices_shown == len(voice_audio_list):
                break

def voice_callback():
    if st.session_state.voice_audio:
        st.session_state.selected_voice = st.session_state.voice_audio

def uploaded_voice_callback():
    if st.session_state.uploaded_voice_audio is None:
        pass
    else:
        audio_path = "uploaded_voice_audio" + \
            os.path.splitext(st.session_state.uploaded_voice_audio.name)[-1]
        with open(audio_path, "wb") as f:
            f.write(st.session_state.uploaded_voice_audio.getvalue())

step21_col1, step21_col2 = st.columns(2)

with step21_col1:
    selected_voice = st.selectbox(
        label='Please select a voice to clone',
        options=voice_names_list,
        key='voice_audio',
        on_change=voice_callback
    )

    st.write('or')

    uploaded_voice = st.file_uploader(
        "Upload a voice to clone", 
        type=['mp3', 'wav'],
        key='uploaded_voice_audio',
        on_change=uploaded_voice_callback
    )

with step21_col2:
    st.markdown('<br>', unsafe_allow_html=True)
    if uploaded_voice is None:
        st.audio(os.path.join('voice_audios', voice_audio_list[voice_names_list.index(selected_voice)]))
    else:
        uploaded_voice_audio_path = glob('uploaded_voice_audio.*')[0]
        st.audio(uploaded_voice_audio_path)

step21txt_col1, step21txt_col2 = st.columns(2)

with step21txt_col1:
    uploaded_txt = st.text_area(
        label='Please input text for avatar',
        key='txt4audio'
    )

def generate_audio():
    if st.session_state.audio_button:

        if uploaded_voice is None:
            speaker_wav = os.path.join('voice_audios', voice_audio_list[voice_names_list.index(selected_voice)])
        else:
            speaker_wav = "uploaded_voice_audio.mp3"

        get_audio(
            synthesizer, speaker, language, 
            speaker_wav, split_sentences, 
            text=st.session_state.txt4audio
        )

with step21txt_col2:
    st.markdown('<br>', unsafe_allow_html=True)
    st.button(
        label='Generate audio from text',
        key='audio_button',
        on_click=generate_audio
    )

if st.session_state.audio_button:
    gen_audio_col1, _ = st.columns(2)
    gen_audio_col1.audio("generated_audio.wav")

# st.subheader('Step 2 - Option 2')

option1_expander = st.expander('Option 2')
option1_expander.write(
    '''Please, just upload an audio that will reproduced in the video.
    '''
)

def uploaded_audio_callback():
    if st.session_state.uploaded_audio is None:
        pass
    else:
        audio_path = "uploaded_audio" + \
            os.path.splitext(st.session_state.uploaded_audio.name)[-1]
        with open(audio_path, "wb") as f:
            f.write(st.session_state.uploaded_audio.getvalue())

step22_col1, step22_col2 = st.columns(2)

with step22_col1:
    uploaded_audio = st.file_uploader(
        "Please, upload an audio", 
        type=['mp3', 'wav'],
        key='uploaded_audio',
        on_change=uploaded_audio_callback
    )

with step22_col2:
    st.markdown('<br>', unsafe_allow_html=True)
    if uploaded_audio is None:
        pass
    else:
        st.audio(glob('uploaded_audio.*')[0])

st.subheader('Step 3')

def generate_video():
    if st.session_state.video_button:

        if uploaded_audio is None:
            voice_audio = glob('generated_audio.*')[0]
        else:
            voice_audio = glob('uploaded_audio.*')[0]

        # if st.session_state.audio_button:
        #     voice_audio = glob('generated_audio.*')[0]
        # else:
        #     voice_audio = os.path.join('voice_audios', voice_audio_list[voice_names_list.index(selected_voice)])

        if uploaded_image is None:
            face = os.path.join('avatar_images', avatar_images_list[avatar_names_list.index(selected_avatar)])
        else:
            face = glob('uploaded_avatar_image.*')[0]

        create_video(voice_audio, face)

step3_button_col1, _, _ = st.columns([3, 4, 5])

with step3_button_col1:
    st.button(
        label='Generate video',
        key='video_button',
        on_click=generate_video
    )

if st.session_state.video_button:

    step3_col1, _, _ = st.columns([4, 3, 5])

    with step3_col1:
        st.video(
            # os.path.join('avatar_videos', 'generated_video.mp4')
            'generated_video.mp4'
        )

    # with step3_col2:
    #     # st.markdown('<br>', unsafe_allow_html=True)
    #     # with open(os.path.join('avatar_videos', 'generated_video.mp4'), 'rb') as file:
    #     with open('generated_video.mp4', 'rb') as file:
    #         st.download_button(
    #             label='Download generated video',
    #             data=file,
    #             file_name='avatar_video.mp4',
    #             mime='video/mp4'
    #         )