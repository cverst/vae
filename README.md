# vae

We need to download two datasets from Kaggle:
1. [A JSON file with metadata and annotations for villagers](https://www.kaggle.com/datasets/nookipedia/animal-crossing-series-villagers)
2. [Image files for 492 villagers](https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers)

The dataset is too small to encode any colors. We therefore use a greyscale version of the images.
The dataset is small, which means we must limit the number of laten variables. Two is too much in most of the latent space, one has a more continuous laten space.
ReLU in final layer works better than sigmoid
