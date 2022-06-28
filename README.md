# Debiasing Deep Chest X-Ray Classifiers using Intra- and Post-processing Methods

This repository holds the official code for the paper "*Debiasing Deep Chest X-Ray Classifiers using Intra- and Post-processing Methods*" presented at the [ICLR 2022 Workshop on Socially Responsible Machine Learning](https://iclrsrml.github.io/) and accepted at the [the 7<sup>th</sup> Machine Learning for Healtcare Conference (MLHC), 2022](https://www.mlforhc.org/). A short explanation of the method is provided in this [contributed talk](https://youtu.be/Kw3Cf7XxzNs); poster can be viewed [here](documents/DiffBiasProxies_poster_SRML_ICLR_2022.pdf).

<p align="center">
  <img align="middle" src="figures/setting_summary.png" alt="Intra-processing scenario" width="350"/>
</p>
<center>
  <i>The <b>intra-processing setting</b>: a classification model is trained on centre <b>1</b>, and debiased on centres <b>2</b>, <b>3</b>, and <b>4</b> that might have different protected attributes and fairness constraints, denoted by $A$.</i>
</center>

### 🦴 Motivation

<img align="right" src="figures/debiasing_workflow.png" width="400" />

Deep neural networks for image-based screening and computer-aided diagnosis have achieved expert-level performance on various medical imaging modalities, including chest radiographs. Recently, several works have indicated that these state-of-the-art classifiers can be biased with respect to sensitive patient attributes, such as race or gender, leading to growing concerns about demographic disparities and discrimination resulting from algorithmic and model-based decision-making in healthcare. A practical scenario of mitigating bias w.r.t. protected attributes could be as follows: consider deploying a predictive neural-network-based model in several clinical centres with different demographics (see the figure above). The constraints on the bias and protected attribute of interest might vary across clinical centres due to different population demographics. Therefore, it might be more practical to debias the original model based on the *local* data, following an **intra-** or **post-processing** approach.

### ✂️ Pruning and Gradient Descent/Ascent for Debiasing

This repository implements two novel intra-processing techniques based on **fine-tuning** and **pruning** an already-trained neural network. These methods are simple yet effective and can be readily applied *post hoc* in a setting where the protected attribute $A$ is unknown during the model development and test time. The general debiasing procedure is schematically summarised in the figure to the right: an already-trained network $f$<sub><img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\theta}"></sub>$(\cdot)$ is debiased on held-out validation data <img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}"><sub>valid</sub>, <img src="https://render.githubusercontent.com/render/math?math=\mathcal{Y}"><sub>valid</sub>, using differentiable proxy functions for the classification parity, and can produce unbiased predictions without the protected attribute $A$ at *test time*.
  
### 📝 Requirements
All the libraries required are in the conda environment [`environment.yml`](environment.yml). To install it, follow the instructions below:
```
conda env create -f environment.yml   # install dependencies
conda activate DiffBiasProxies        # activate environment
```

### ⚙️ Usage
[`/bin`](bin/) folder contains shell scripts for the experiments described in the paper:
- **Tabular**: [`run_adult`](bin/run_adult), [`run_bank`](bin/run_bank), [`run_compas`](bin/run_compas), [`run_mimic_iii`](bin/run_mimic_iii)
- **MIMIC-CXR**: [`run_mimic_cxr`](bin/run_mimic_cxr)

To run the MIMIC-III experiment, first execute the [code](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII) by Sanjay Purushotham, Chuizheng Meng, Zhengping Che, and Yan Liu, to get the pre-processed files.

Further details are documented within the code.

### 🙏 Acknowledgements
The code structure is based on the repository by Yash Savani, Colin White, and Naveen Sundar Govindarajulu available [here](https://github.com/abacusai/intraprocessing_debiasing).

### 📧 Maintainers

### 📕 References

### 🏆 Citation
