# Segmentation for Wildscenes Dataset using DeepLabV3

## Intro
This Project is aimed at using DeepLabV3 pretrained model to ahchieve segmentation for WildScenes Dataset (https://csiro-robotics.github.io/WildScenes/). The dataset contains cloud points data and 2D .png data. The 2D .png data from directory WildScenes2d is only used in the project.

## Repository Structure
```
├── demo.ipynb               # Get the Color & idx mapper, Plot the train logs and Show the prediction result
├── color2id.json                 # Color & idx mapper
├── utils.py                       # Functions about resize, encode RGB into class idx, augmentation operations ...
├── dataset.py                        # Create pytorch dataset
├── train.py                        # Train and test
├── predict.py                          # Predict for a single image
```
