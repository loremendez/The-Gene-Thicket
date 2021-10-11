# The Gene Thicket
Inference of gene regulatory networks (GRN's) using temporal convolutional neural networks (CNN's).

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
-->

<!-- PROJECT LOGO -->

![Screenshot from 2021-08-21 18-29-16](https://user-images.githubusercontent.com/62608007/136803016-a54437b9-6268-4d24-b92a-148d8533827e.png)


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
## About The Gene Thicket

The Gene Thicket is my Master's thesis project. The gene thicket is a deep learning-based model, which goal is to build gene regulatory networks that are capable to predict future gene expressions.    

**What is so amazing about of gene regulatory networks?**

Gene Regulatory Networks are graphs that describe the regulatory process of a system. Their nodes are genes and their edges represent the regulatory relationship between two genes. The edges have a direction (which gene regulates which other gene), a sign (activation or repression) and a weight (strength of the relationship). If we understand how a system works, then we can develop better products to help people, animals and every other living organism.

The inference of gene regulatory networks (GRNs) has been an area of research for more than twenty years! GRNs are an amazing challenge in Computational Biology since they relate to many different things, such as perturbations, trajectory inference, RNA velocity and much more! We can also be creative and incorporate different types of data.

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
