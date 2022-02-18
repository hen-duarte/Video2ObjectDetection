# Video2ObjectDetection
This repo contains a beginner version of the "Computer Vision Component" prototype that allows the extraction of events in the turnaround process of the airport. 

The code is inspired by the [YOLO](https://github.com/pjreddie/darknet) repository. 
 
## Requirements
Tested with Python 3.9. Install the necessary packages with
```bash
pip install -r requirements.txt
```

## Download of the weights file
The weights file is the neural network model trained specifically for the turnaround process, with airport objects. 
The model can be downloaded in the following link: 
<https://drive.google.com/drive/folders/1MjeCrXIQclk0y9LlL--hWrUvUXPTHaEx>

## Download examples of turnaround videos
There are a short list of turnaround airport videos available online. 
A list of youtube videos can be downloaded using the following:
<https://drive.google.com/drive/folders/1ZuBrpAuvPmQ4P5a-26dgpZMdPjBPwRvD?usp=sharing>

## Solution configuration
There are 2 configuration needed to be made before running the code:
* Video Stream HTTP Connection 
* Siddhi Stream HTTP Connection

## Domain 
This solution describes a process in real-world environment by acquiring data from environments where wouldn't be possible. 
The domain of this solution could be given by:
![solution_domain](https://user-images.githubusercontent.com/99749820/154712825-cf3bb91c-bd7a-491b-a35e-cd41ce65f19e.png){width="100px"}


# Prototype Running Printscreen
![printscreen-prototype](https://user-images.githubusercontent.com/99749820/154713987-1f305130-7f4b-464a-ae03-94e9959433f6.png){width="100px"}
