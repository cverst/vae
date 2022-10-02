# VAE
A Variational Autoencoder (VAE) training on images of Animal Crossing villagers.

# Table of contents
1. [Description](#description)
2. [Getting started](#getting-started)
3. [Running the code](#running-the-code)
4. [Results from experiments](#results-from-experiments)

<a id="description" /></a>
## 1. Description
This repository contains the code to train a VAE on a toy dataset of Animal Crossing villagers. If you do not know the principles underlying VAEs I recommend reading [this blog post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73). The blog gives a nice introduction to and explanation of VAEs and helped me a lot during development of this repo. Two other links that were of great use to me are [this blog post](https://towardsdatascience.com/vae-variational-autoencoders-how-to-employ-neural-networks-to-generate-new-images-bdeb216ed2c0) and [this tutorial](https://www.tensorflow.org/tutorials/generative/cvae). All code should be run from the command line.

Sections [2. Getting started](#getting-started) and [3. Running the code](#running-the-code) contain information on how to install and run the code. In section [4. Results from experiments](#results-from-experiments) you can find some results from experiments using this code to train VAEs on the Animal Crossing dataset.

<a id="getting-started" /></a>
## 2. Getting Started

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
conda create -n vae_env python=3.10
```
and activate the environment
```
conda activate vae_env
```
Next, access the project folder
```
cd vae
```
and install the required python packages in your environment
```
pip install -r requirements.txt
```

<a id="running-the-code" /></a>
## 3. Running the code

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

<a id="results-from-experiments" /></a>
## 4. Results from experiments
With the information of the blogs mentioned in [1. Description](#description) in mind, we can carry out a few experiments. The outputs from such experiments are discussed below.

### Example data
We will first take a look at a sample of our dataset. We see colorful animals of different species, gender, and personality. The shape of a villagar, mostly associated with its species, seems the most prominent feature. This prominence will be emphasized by the results of our modeling as shown in the sections below.

![Samples from the Animal Crossing villager dataset](/readme_images/samples.jpg)

### Note on model architecture
In this repository we opt for convolutional layers because we work with image data with multiple color channels. Moreover we try to use settings that create a somewhat gradual decrease in features towards the bottleneck. In contrast, many other VAE examples use only fully connected layers. As a consequence of the convolutional layers, our example may increase the risk of overfitting.

This code base was never meant to be used for finding an optimal model architecture. In particular, the peformance of our VAEs will mostly be limited by the small size of our dataset. Our model works good enough and we can make some observations following simple changes to our hyperparameters and discuss these results below.

### VAE with 2D latent space
A commonly used datset for VAE experiments is the MNIST dataset. Compared to our dataset, the MNIST dataset has 150x the number of images and the MNIST images are simpler. The MNIST dataset is commonly embedded into a latent space of two latent distributions. We take this as a starting point.

We start with training a VAE with the following parameters:

    latent_dim = 2
    image_shape = [64, 64]
    n_channels = 4
    n_epochs = 2000

As a first result, we take a look at training and validation loss.

![Training and validation loss for a 2D latent space](/readme_images/loss_2d.jpg)

As we keep training our model we can see that the training loss keeps decreasing, whereas the validation loss seems to level off. So here we see a first hint of overtraining.

With our model trained we can take a look at how well defined and regularized (see [here](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)) our latent space actually is. We plot the 2D embeddings of our validation set in the latent space. You obtain this representation by processing our images with the encoder only.

![Validation images on a 2D manifold](/readme_images/species_manifold_2d.jpg)

As you can see in the legend we used different colors for different species. In general, animals from the same species were relatively close to one another, so that requirement on the latent space was met. However, the overall spread of our embeddings is quite high, up to about 15 standard deviations from the mean. We can take a final look at this first model by sampling from our latent space. We first make a raster of coordinates of our latent variables across our latent space and then process each coordinate with our decoder. This way we can create new data and at the same time see how our latent space behaves.

![Images generated from 2D latent space](/readme_images/reconstructed_images_2d.jpg)

We can clearly see how different species would map to our 2D latent space. At the same time we see some nonsense data, in the middle on the right, an indication of poor regularization. Moreover, we see some images that strongly resemble some of our training images. For instance, the bottom row contains two examples that are nearly identical to the following two training images.

![Agent S](/readme_images/Agent_S.png)
![Peanut](/readme_images/Peanut.png)

Clearly, our model is overfitting. With a 2D latent space our model has too much capacity for a dataset as small as ours. The number of latent distributions should be smaller for smaller datasets and can only be bigger for bigger datasets!

### VAE with 1D latent space
In an attempt to remove overfitting we reduced the latent space to a single latent distribution. Looking at the training and validation loss we do not see clear overfitting as with a model with a 2D latent space.

![Training and validation loss for a 1D latent space](/readme_images/loss_1d.jpg)

We are skipping the look at the embedded validations set here and immediately take a look at the decoded latent space.

![Images generated from 1D latent space](/readme_images/reconstructed_images_1d.jpg)

From left to right we can see a rabbit, flamingo, bear, kangaroo, horse, pig, dog, deer, unknown/hamster/tiger, duck/frog, mouse, and koala. Although we lost some of the color of the 2D representation, the latent space is continuous and seemingly denser information. We also do not see samples from our training set in our generated data. These changes indicate better regularization of the latent space.

Finally we create a movie where we walk through the latent space "from left to right." This movie shows a seamless transition from one generated villager to the next. It looks like we will go from species to species, but only because the villager shape defined by their species is such a dominant feature in the dataset.

![Walk across 1D latent space](/readme_images/walk_across_latent_space_1d.gif)
