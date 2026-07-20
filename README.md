<p align="center">
	<a href='https://www.sciencedirect.com/science/article/pii/S1093968726031117/pdfft?md5=e938bb1a4863999573698c8be612eb7a&pid=1-s2.0-S1093968726031117-main.pdf'><img src='https://img.shields.io/badge/Paper-Elsevier-FF551D?logo=elsevier' alt='Paper' /></a>
	<a href='https://nevf-cfd.github.io/'><img src='https://img.shields.io/badge/Project-Page-green?logo=safari&logoColor=fff' alt='Project Page' /></a>
	<a href="https://colab.research.google.com/drive/17gbHzfWKct0r8XctirKBpJhCVFpNQNHL?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	<a><img src='https://img.shields.io/badge/python-3.10%2B-blueviolet' alt='Python' /></a>
	<a><img src='https://img.shields.io/badge/code%20style-black-black' /></a>
	<a href='https://opensource.org/license/lgpl-2-1'><img src='https://img.shields.io/badge/license-LGPLv2+-blue' /></a>
</p>

# 🧊 NeVF

[**NeVF: Representing CFD simulations as neural flow volume fields for efficient compression, reconstruction, and analysis**](https://doi.org/10.1016/j.cacaie.2026.100125)<br/>
[Omar A. Mures](https://omaralv.com/), [Miguel Cid Montoya](https://mcidmontoya.com/)

## 📢 Latest News

#### 🔥 **[2026.08]** [Colab](https://colab.research.google.com/drive/17gbHzfWKct0r8XctirKBpJhCVFpNQNHL?usp=sharing) example released! 💡
#### 🔥 **[2026.07]** Pip package released! 📦
#### 🔥 **[2026.07]** Code Released - Get Started Now! 🚀
#### 🔥 **[2026.02]** Code Coming Soon! 👀

## Usage

### Installation

The `nevf` library is available using pip. We recommend using a virtual environment to avoid conflicts with other software on your machine.

``` bash
pip install nevf
```

To run the demo:

```bash
python train.py -c configs/NeVF_S_LES.json
```

To create a conda env for development:

```bash
conda env create -f environment.yaml
```

## 🎯 TODO

The repo is still under construction, thanks for your patience. 

- [x] Release Colab example.
- [x] Release pip package.
- [x] Release of the neural compression code.

## 📜 Citation

```
@Article{mures2026nevf,
  author = {Mures, Omar A. and Cid Montoya, Miguel},
  title = {{NeVF: Representing CFD} simulations as neural flow volume fields for efficient compression, reconstruction, and analysis},
  journal = {Computer-Aided Civil and Infrastructure Engineering},
  volume = {49},
  pages = {100125},
  year = {2026},
  doi = {10.1016/j.cacaie.2026.100125},
  url = {https://doi.org/10.1016/j.cacaie.2026.100125},
}
```
