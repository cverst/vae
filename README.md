# VAE
A Variational Autoencoder (VAE) training on images of Animal Crossing villagers.

## Description
This repository contains the code to train a VAE on a toy dataset of Animal Crossing villagers. If you do not know the principles underlying VAEs I recommend reading [this blog post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73). The blog gives a nice introduction to and explanation of VAEs and helped me a lot during development of this repo. Two other links that were of great use to me are [this blog post](https://towardsdatascience.com/vae-variational-autoencoders-how-to-employ-neural-networks-to-generate-new-images-bdeb216ed2c0) and [this tutorial](https://www.tensorflow.org/tutorials/generative/cvae). All code should be run from the command line.

## Getting Started

### Software
The following software needs installation before continuing
* Python: version 3.10 was used to develop this repo, but earlier versions are likely to work as well
* An environment manager such as conda or pipenv

### Clone repo
Clone the repository with
```
git clone https://github.com/cverst/vae
```

### Download data
Two Kaggle-datasets need downloading
1. [A JSON file with metadata and annotations for villagers](https://www.kaggle.com/datasets/nookipedia/animal-crossing-series-villagers)
2. [Image files for 392 villagers](https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers)

Store these data in a location you remember, e.g., a folder named `./data` in the root of this repo.

### Virtual environment
Create a virtual environment. You can use your environment manager of choice. The example below uses a combination of conda and pip to set up an environment.

First create an environment
```
conda create -n minicomp_env python=3.8
```
and activate the environment
```
conda activate minicomp_env
```
Next, access the project folder
```
cd vae
```
and install the required python packages in your environment
```
pip install -r requirements.txt
```

## Running the code

### Data processing
Before we can start to train a model we need to preprocess our data. We will merge the two separate datasets and create a tfrecord for downstream use.

Start by changing directories.
```
cd extract_data
```
Next we preprocess the JSON file containing data annotations. the `--target` argument is optional and will default to the file path `../data/animal-crossing-villagers-parsed.json`.
```
python parse_json.py <location of animal-crossing-villagers.json> --target <location for animal-crossing-villagers-parsed.json>
```
We follow up by creating the tfrecord file. The `--annotations` argument corresponds to the `--target` file path from the `parse_json.py` call and is optional, defaulting to the default of the `parse_json.py` call. The `--target` argument is optional and defaults to the file path`../data/villagers.tfrecord`.
```
python create_dataset.py <location of .../images/ folder> --annotations <location of animal-crossing-villagers-parsed.json> --target <location for villagers.tfrecord>
```
We can check if our tfrecord creation was succesful by looking at a sample from the tfrecord. Both arguments are optional, where `--source` defaults to the `--target` of the `create_dataset.py` call, and `n_shuffle` defaults to 1 (no shuffling).
```
python show_sample.py --source <location of .../images/ folder> --n_shuffle <size of shuffle buffer>
```

### Running the model
With our dataset in place we can train our model. First, change back to the root of the repository.
```
cd ..
```
There we can train our model using default settings with the following command.
```
python run.py
```
If we changed the tfrecord location from its default, we need to change the command to the following.
```
python run.py --tfrecord_path <path to villagers.tfrecord>
```
The following command would do the exact same as the command above, because we are using all default arguments. These can be changed for more control over how the model trains.
```
python run.py --tfrecord_path <path to villagers.tfrecord> --image_shape 64 64 --n_channels 4 --latent_dim 1 --n_epochs 500 --epoch_step 1
```
* `--image_shape` specifies the resolution of our data as it goes into the model
* `--n_channels` sets the number of color channels to use, which must be one of 1, 3, or 4
* `--latent_dim` sets the number of latent distributions to use
* `--n_epochs` sets the number of epochs to train our model
* `--epoch_step` specifies sets the interval in epochs after which an image should be saved to a video of training results 

### Done!
That's it! You are all set to experiment with the code.





## Notes for GitHub pages


The dataset is too small to encode any colors. We therefore use a greyscale version of the images.
The dataset is small, which means we must limit the number of laten variables. Two is too much in most of the latent space, one has a more continuous laten space.
ReLU in final layer works better than sigmoid

subclassing a tf.keras model is nicer but chose this option for simplicity

used three convolutional layers. though other models might have better performance, this worked good enough

overfitting for 2d latent space, see eamples and real images
1d latent space worked better
