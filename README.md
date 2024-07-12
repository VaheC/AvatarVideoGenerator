# AvatarVideoGenerator
The aim of the repository is to create an app which will allow to generate talking avatar videos.

The general pipeline of the app is represented in the image below.

![image](avatar_video_generation_pipeline.jpg)

## Websites providing similar service

Before working on the project I visited some websites that provide the similar service. Here are them: [Runway](https://runwayml.com/), [Nighcafe Creator](https://creator.nightcafe.studio/), and [Visper](https://visper.tech/en).

Of course, all of the websites provide a great content and services. Not surprisingly the options available for free users are not plausible in terms of the generated video's duration. Specifically, Runway allows to generate videos with duration of 4 seconds if you are a free user. Though Visper allows free users to create videos that last up to 20 seconds, it is also not a great deal. Finally Nightcafe Creator doesn't provide video generation service. But they have a service which generates image from text. 

## Detailed description of the pipeline

At first, when a user visits the app, she/he needs to provide an image of an avatar that will be talking in the video. There are 3 options to do this. The first one is to expand a bar called "Available avatars" to see the default options available in the app and select a avatar using "Please select an avatar" dropdown list. The second option is to upload an avatar of her/his choice using "Please upload an avatar" upload button. PLEASE REMEMBER TO REMOVE AN UPLOADED IMAGE IF YOU ARE GOING TO USE THE OTHER OPTIONS AFTER TRYING THE SECOND OPTION (there will be "cross" sign available next to the uploaded image in the app, you can use it to remove the image). Finally the third option is to generate an avatar using stable-diffusion model by stabilityai which is available publicly on huggingface. She/he needs to provide a text for prompt in the "Please type a prompt to generate an image for the avatar" text area and to click on "generate_avatar" button. It takes around 5 minutes to generate an avatar using the last option. PLEASE REMEMBER THAT AN AVATAR IMAGE SHOULD CONTAIN A FACE IN IT, OTHERWISE THE APP WILL FAIL.
