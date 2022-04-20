# 02506 Mini Project -  CNN for segmentation
Mini project in the course Advanced Image Analysis Spring 22 @ The Technical University of Denmark.
<br>
Using a Convolutional Neural Network for image segmentation.

## Setup
To run the code in this project do a
```
pip install -r requirements.txt
```
in the root folder to install the dependencies to your environment.
<br>
If new pip dependencies are added to the project update the `requirements.txt` file with the dependencies **and the dependency version** to make sure everyone is using the same pip packages and versions to avoid errors.

## Data
The image data is ignored through the `.gitignore`, but can be downloaded automatically by running 
```
python make_data.py
```
using the script in the folder `src/data`.
<br><br>
Otherwise the data can manually be downloaded from [here](http://www2.imm.dtu.dk/courses/02506/data/EM_ISBI_Challenge.zip).
<br>
Then unzip the data and place the contents in the `data/raw` folder.

### Creating Image Patches
To generate image patches and patch labels run the script
```
python make_image_patches.py
```
in the folder `src/features`.

## Repository Structure
The structure of this repo follows the [Cookiecutter Data Science Template](https://drivendata.github.io/cookiecutter-data-science/#directory-structure).

### data
All raw data should be saved to the `data/raw` folder. When data is preprocessed through code it should be saved (if necessary) to the `data/preprocessed` folder.

### models
If models are saved after training for later use, they should be saved to the `models` folder.

### notebooks
All notebooks should be saved to the `notebooks` folder.

### reports
If analysis are saved, they should be saved to the `reports` folder and specifically figures/plots needs to be saved to the `reports/figures` folder.

### src
The `src` folder should contain **all** code for the project except that in the `notebooks`.

#### src/data
The `src/data` folder should contain code that generates or downloads data to the `data/raw` folder.

#### src/features
The `src/features` folder should contain code that generates data features from the raw data in `data/raw` and saves it to the `data/preprocessed` folder.

#### src/models
The `src/models` folder should contain code for training a model and saving it to `models` in the root folder if necessary or loading a model from `models` and perform predictions.

#### src/visualization
The `src/visualization` folder should contain code for generating figures/plots and saving them to `reports/figures`.

## Authors
August Semrau Andersen, Gustav Gamst Larsen, and William Diedrichsen Marstrand.