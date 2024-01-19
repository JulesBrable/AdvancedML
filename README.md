# AdvancedML
Repository for the final project of the Advanced Machine Learning course (taught by Austin Stromme during the 1st Semester of the final year at ENSAE Paris).

## Contents

* `notebooks` folder contains the notebook `descriptives.ipynb`, which briefly shows how to reproduce the empirical results we obtained in our paper (the underlying approach followed in this notebooks is described in our paper). This notebook contains 3 sections : Data, K-Fold Cross Validation and Training with best hyper-paramters. The `algorithms.ipynb` notebook documents our theoretical experiment, where we sought optimal settings to minimize different types of functions using Adam and other stochastic algorithms. It includes visualizations of convergence paths and tables summarizing the hyperparameters for replicating the results.
* `src` folder holds two modules (`etl` and `model`) utilized in the `descriptives.ipynb` notebook, and another (`algorithms`) used in `algorithms.ipynb`. The scripts in the first two modules handle data extraction, preprocessing, model construction, training, and evaluation. The latter module contains our Adam algorithm implementation (`adam.py`) and utility functions for optimization, hyperparameter tuning, and visualization.
* `app` folder contains the source code of a [`Streamlit`](https://streamlit.io/) web app (see below for a more detailed description of this app).

_**Note:** this repository does not contain the final report associated to this study (in which we present more in depth both theoretical and empirical aspects of our work)._

## Setup Instructions

From the command line, you will have to follow the following steps to set this project up:

1. Clone this repository:

```bash
git clone https://github.com/JulesBrable/AdvancedML.git
```

2. Go to the project folder:
```bash
cd AvdancedML
```

3. Install the listed dependencies:
   
```bash
pip install -r requirements.txt
```

4. Install the project as a package:
   
```bash
pip install -e .
```

5. Get the data:

```python
python src/etl/utils.py
```

_**Note:** the data comes from the public [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), and can also be directly downloaded from this site. We have also added the data in this repository in the `data` folder for back-up, although we are aware it is not a best practice._

## Web application

In this project, we also built a simple [`Streamlit`](https://streamlit.io/) web app. Its purpose is to visualize intuitively how the Adam algorithm is performing on minimizing a mathematical function, compared to other optimization algorithms. The user is encouraged to test different combination of hyper-paramters, to see their impact on the optimization path.

To access the app, one can simply click [here](https://advancedml-optimization.streamlit.app/) (it is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)). You just have to authenticate yourself before (creating a Streamlit account is completely free of charge).

On the other hand, you can also run this app locally. To do so, once you have followed the set-up instruction described above, you will have to run the following commands:

1. Go to the app folder:

```bash
cd app
```

2. Run this app locally:

```bash
streamlit run app.py --server.port=8000
```

By default, we are using port 8000, so once you have run the last two commands, you will be able to access the app with the following link: [http://localhost:8000/](http://localhost:8000/).

<br>

<p align="center">
  <img src="https://github.com/JulesBrable/AdvancedML/blob/main/assets/img/app1.png" width="400" style="margin-right: 10px;"/>
  <img src="https://github.com/JulesBrable/AdvancedML/blob/main/assets/img/app2.png" width="400" />
</p>

<br>

## Contact

* [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
* [Yedidia Agnimo](https://github.com/Yedson54) - yedidia.agnimo@ensae.fr
* [Ayman Limae](https://github.com/Liaym) - ayman.limane@ensae.fr
