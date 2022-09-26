# vae

install python 3.9
install requirements
use virtual environment of choice, example here using conda

We need to download two datasets from Kaggle:
1. [A JSON file with metadata and annotations for villagers](https://www.kaggle.com/datasets/nookipedia/animal-crossing-series-villagers)
2. [Image files for 492 villagers](https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers)

The dataset is too small to encode any colors. We therefore use a greyscale version of the images.
The dataset is small, which means we must limit the number of laten variables. Two is too much in most of the latent space, one has a more continuous laten space.
ReLU in final layer works better than sigmoid


python parse_json.py ../data/annotations/animal-crossing-villagers.json --target ../data/annotations/animal-crossing-villagers-parsed.json

python create_dataset.py ../data/images/ --annotations ../data/annotations/animal-crossing-villagers-parsed.json --target ../data/villagers.tfrecord

python show_sample.py --source ../data/villagers.tfrecord --n_shuffle 1


python run.py --tfrecord_path ./data/villagers.tfrecord --image_shape 64 64 --n_channels 1 --latent_dim 4 --n_epochs 2 --epoch_step 1


used three convolutional layers. though other models might have better performance, this worked good enough

overfitting for 2d latent space, see eamples and real images
1d latent space worked better