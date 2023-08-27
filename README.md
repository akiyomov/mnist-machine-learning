# MNIST Machine Learning Experiments

This project demonstrates different machine learning models on the MNIST dataset using PyTorch and scikit-learn. The models include Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Transformers, Decision Trees, and k-Nearest Neighbors.

## Project Structure

The project is organized into three main components:

1. `models.py`: Defines the different machine learning models using PyTorch.
2. `dataset.py`: Loads and preprocesses the MNIST dataset using torchvision.
3. `train.py`: Conducts experiments with the models, trains them, and evaluates their performance.

## Setup


## Clone the repository
```sh
git clone https://github.com/your-username/mnist-machine-learning.git
```

## Install dependencies
```sh
pip install torch torchvision scikit-learn
```

## Navigate to the project directory
```sh
cd mnist-machine-learning
```

## Experiment with different models
```sh
python train.py --model <model_name>
```

Replace <model_name> with one of the following choices: ann, cnn, rnn, transformer, dt, knn.

For example, to run experiments using an Artificial Neural Network, execute:
```sh
python train.py --model ann
```

## Notes
- For the Decision Tree and k-Nearest Neighbors models (dt and knn), you'll need to adjust the code in train.py to load and preprocess the data appropriately.
- Feel free to modify the hyperparameters, architecture, and other settings in the code to further experiment and improve results.
- The project is designed for educational purposes and can serve as a starting point for experimenting with different ML models on the MNIST dataset.

## License
This project is licensed under the MIT License.