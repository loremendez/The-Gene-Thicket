# The Gene Thicket
Inference of gene regulatory networks (GRN's) using temporal convolutional neural networks (CNN's).

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">

<h3 align="center">GRN through causal CNN's</h3>

***  <a href="https://github.com/loremendez/Gemstones">
    <p align="center">
      <img src="Resources/Images/sample1.jpg" width="600">
    </p>
  </a>
</p> ***


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Some nice explanation of the project here.

### Built With

* [Anaconda 4.10.1](https://www.anaconda.com/)
* [Python 3.9](https://www.python.org/downloads/release/python-380/)
* [TensorFlow 2.5](https://www.tensorflow.org/tutorials/quickstart/beginner)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

A running installation of Anaconda. If you haven't installed Anaconda yet, you can follow the next tutorial: <br>
[Anaconda Installation](https://docs.anaconda.com/anaconda/install/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/loremendez/Gemstones.git
   ```
2. Install the environment <br>
    You can do it either by loading the [`YML`](https://github.com/loremendez/Gemstones/blob/main/conda_environment.yml) file
    ```sh
    conda env create -f conda_environment.yml
    ```
    or step by step
    1. Create and activate the environment
        ```sh
        conda create -n causal_cnn_env python=3.9
        conda activate causal_cnn_env
        ```
    2. Install the required packages
        ```sh
        pip install --upgrade pip
        pip list  # show packages installed within the virtual environment

        pip install torch #neural network
        pip install numpy pandas matplotlib seaborn networkx cdlib argparse #network analysis
        pip install scikit-learn statsmodels numba python-igraph leidenalg scanpy #scanpy
        pip install scvelo #scvelo

        pip install jupyterlab
        ```

<!-- USAGE EXAMPLES -->
## Usage

Open Jupyter-lab and open the notebook [`Gemstones.ipynb`](https://github.com/loremendez/Gemstones/blob/main/Gemstones.ipynb)
```sh
jupyter-lab
```

<!-- References -->
## References
<a id="1">[1]</a>
Based on TCDF algorithm by [`Nauta et. al, 2019`](https://www.mdpi.com/2504-4990/1/1/19).
Link: [`TCDF github`](https://github.com/M-Nauta/TCDF)


<!-- CONTACT -->
## Contact

Lorena Mendez - [LinkedIn](https://www.linkedin.com/in/lorena-mendezg/?originalSubdomain=de) - lorena.mendez@tum.de

Take a look into my [other](https://github.com/loremendez) projects!
