# Real-time sign language translator
 
Real-time American Sign Language translator. Powered by artificial intelligence leveraging Support Vector Machines (SVM), OpenCV library and MediaPipe hands module. 

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![scikit-learn Badge](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=fff&style=flat)
![MediaPipe Badge](https://img.shields.io/badge/MediaPipe-0097A7?logo=mediapipe&logoColor=fff&style=flat)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project description

### Training dataset

The dataset used for this project is the [ASL (American Sign Language) Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data), which contains a collection of images for training on each sign of the _ASL_ alphabet. These images depict hands from various individuals. Each letter of the alphabet is represented by an average of 7,500 images, which exhibit a variety of skin tones and lighting conditions.
        
To ensure the quality of the data that will feed into the model, the following exclusion criteria were established: 
- **Incomplete images:** Images in which the hand is not fully visible will be removed. 
- **Low-quality images:** Images with excessive digital noise resulting in a "pixelated" appearance will be discarded. 
- **Irrelevant images:** Images in which no hand is present will be excluded.

Additionally, image collections corresponding to the letters "J" and "Z" will be completely discarded, as their representation requires a series of specific movements that cannot be captured in static images. Similarly, images representing deletion and space operations will be removed, as they are not relevant to the proposed recognizer. These measures are aimed at minimizing confusion and maximizing the accuracy of sign identification within the _ASL_.

### Architecture

![System architecture blocks diagram](assets/architecture.jpg)

### Model validation
Cross-validation was chosen as the model validation strategy. This method involves dividing the entire dataset into percentages, where 75\% of the specimens are used for model training, and the remaining 25\% are reserved for validation. The split ratio was selected as it constitutes a standard proportion for dataset division.

## Results

## Result analysis

## Conclusion

## Demo

## Author

**Andrés Montero Gamboa**<br>
Computing Engineering Undergraduate<br>
Instituto Tecnológico de Costa Rica (TEC)<br>
[LinkedIn](https://www.linkedin.com/in/andres-montero-gamboa) | [GitHub](https://github.com/andresmg07)

## License

MIT License

Copyright (c) 2024 Andrés Montero Gamboa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.