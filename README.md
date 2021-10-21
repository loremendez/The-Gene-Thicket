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
      <a href="#about-the-project">About The Gene Thicket</a>
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
* [Pytorch 1.9](https://pytorch.org/blog/pytorch-1.9-released/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

A running installation of Anaconda. If you haven't installed Anaconda yet, you can follow the next tutorial: <br>
[Anaconda Installation](https://docs.anaconda.com/anaconda/install/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/loremendez/The-Gene-Thicket.git
   ```
2. Install the environment <br>
    You can do it either by loading the [`YML`](https://github.com/loremendez/The-Gene-Thicket/blob/main/gene_thicket_env.yml) file
    ```sh
    conda env create -f gene_thicket_env.yml
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

        pip install jupyterlab
        ```

<!-- USAGE EXAMPLES -->
## Usage

The first thing to do is download the curated and synthetic data from [`Pratapa et. al, 2020`](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7098173/)

Open Jupyter-lab and open the notebook [`Run_Gene_Thicket.ipynb`](https://github.com/loremendez/The-Gene-Thicket/blob/main/Analysis_beeline_data/Run_Gene_Thicket.ipynb)
```sh
jupyter-lab
```
To benchmark our method, we ran [`Genie3`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0012776), [`GRNBoost2`](https://academic.oup.com/bioinformatics/article/35/12/2159/5184284?login=true), [`TCDF`](https://www.mdpi.com/2504-4990/1/1/19) and [`Sincerities`](https://academic.oup.com/bioinformatics/article/34/2/258/4158033?login=true).

To run Genie3 and GRNBoost, you need to install the environment: [`arboreto_env.yml`](https://github.com/loremendez/The-Gene-Thicket/blob/main/arboreto_env.yml). To run Sincerities, you need to install the environment: [`sincerities_env.yml`](https://github.com/loremendez/The-Gene-Thicket/blob/main/sincerities_env.yml). TCDF runs with gene_thicket_env, but you need to download the py files available on [`TCDF github`](https://github.com/M-Nauta/TCDF).

Run the models first, then the Analysis that shows a comparison between the models. You can also take a look at particular examples in the rest of the notebooks.

To test The Gene Thicket on real data, we selected the Pancreas dataset that it is integrated in [`scvelo`](https://scvelo.readthedocs.io/scvelo.datasets.pancreas/). To run this analysis, you need to follow the notebooks for the Pancreas Analysis and download the [`SCENIC`](https://www.nature.com/articles/s41596-020-0336-2) files and cis-target databases that are indicated in the notebooks.

<!-- References -->
## References
<a id="1">[1]</a>
Based on TCDF algorithm by [`Nauta et. al, 2019`](https://www.mdpi.com/2504-4990/1/1/19).
Link: [`TCDF github`](https://github.com/M-Nauta/TCDF)


<!-- CONTACT -->
## Contact

Lorena Mendez - [LinkedIn](https://www.linkedin.com/in/lorena-mendezg/?originalSubdomain=de) - lorena.mendez@tum.de

Take a look into my [other](https://github.com/loremendez) projects!
