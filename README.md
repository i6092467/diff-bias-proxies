## Debiasing Deep Chest X-Ray Classifiers using Intra- and Post-processing Methods

### Abstract
Deep neural networks for image-based screening and computer-aided diagnosis have achieved expert-level performance on various medical imaging modalities, including chest radiographs. Recently, several works have indicated that these state-of-the-art classifiers can be biased with respect to sensitive patient attributes, such as race or gender, leading to growing concerns about demographic disparities and discrimination resulting from algorithmic and model-based decision-making in healthcare. Fair machine learning has focused on mitigating such biases against disadvantaged or marginalised groups, mainly concentrating on tabular data or natural images. This work presents two novel intra-processing techniques based on fine-tuning and pruning an already-trained neural network. These methods are simple yet effective and can be readily applied *post hoc* in a setting where the protected attribute is unknown during the model development and test time. In addition, we compare several intra- and post-processing approaches applied to debiasing deep chest X-ray classifiers. To the best of our knowledge, this is one of the first efforts studying debiasing methods on chest radiographs. Our results suggest that the considered approaches successfully mitigate biases in fully connected and convolutional neural networks offering stable performance under various settings. The discussed methods can help achieve group fairness of deep medical image classifiers when deploying them in domains with different fairness considerations and constraints.

### Requirements
All the libraries required are in the conda environment [`environment.yml`](environment.yml). To install it, follow the instructions below:
```
conda env create -f environment.yml   # install dependencies
conda activate DiffBiasProxies        # activate environment
```

### Experiments
[`/bin`](bin/) folder contains shell scripts for the experiments described in the paper:
- **Tabular**: [`run_adult`](bin/run_adult), [`run_bank`](bin/run_bank), [`run_compas`](bin/run_compas), [`run_mimic_iii`](bin/run_mimic_iii)
- **MIMIC-CXR**: [`run_mimic_cxr`](bin/run_mimic_cxr)

To run the MIMIC-III experiment, first execute the [code](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII) by Sanjay Purushotham, Chuizheng Meng, Zhengping Che, and Yan Liu, to get the pre-processed files.

Further details are documented within the code.

### Acknowledgements
The code structure is based on the repository by Yash Savani, Colin White, and Naveen Sundar Govindarajulu available [here](https://github.com/abacusai/intraprocessing_debiasing).
