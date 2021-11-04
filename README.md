# Integrating Expert Knowledge with Domain Adaptation for Unsupervised Fault Diagnosis
To appear in IEEE Transactions on Instrumentation and Measurement. [PDF](https://arxiv.org/abs/2107.01849)


## Introduction
In this repository, we provide the readers with the synthetic CWRU data set. We generate the synthetic faults by injecting experts' understanding on fault patterns to the healthy signals. Details can be found in the paper.

### Download Links
+ [Dataset](https://github.com/qinenergy/syn2real/releases)
  + [Our Synthetic CWRU](https://github.com/qinenergy/syn2real/releases/download/data/cwru_synthetic.parquet)
  + [Real CWRU](https://github.com/qinenergy/syn2real/releases/download/data/cwru.parquet) | [Original author's link](https://engineering.case.edu/bearingdatacenter/download-data-file)
+ [Google Colab](https://colab.research.google.com/drive/1o-8ETOG-ej3HxVl4lvNJN8D0G6734bu7?usp=sharing)

### Structure
```
├── data
│   ├── cwru.parquet            # CWRU dataset in parquet format, provided in the download link
│   ├── cwru_synthetic.parquet  # Synthetic CWRU dataset we generated, provided in the download link
│   ├── preprocessed            # Preprocessed data we used for our experiments
│   │   ├── XreallDEenv.npy
│   │   ├── XsynallDEenv.npy
│   │   ├── yreallDEenv.npy
│   │   └── ysynallDEenv.npy
│   └── preprocess.py           # Preprocessing code
├── code
│   └── Syn2real_CWRU.ipynb     # Notebook to reproduce our CWRU synthetic to real adaptation results. Identical to the Google Colab we provide.
└── README.md
```

Example code to read and preprocess the parquet files are provided in data/preprocess.py 

## Code
We provide a google colab research notebook for readers to better under our method. You can play with it [here](https://colab.research.google.com/drive/1o-8ETOG-ej3HxVl4lvNJN8D0G6734bu7?usp=sharing).

## Reference
If you use our dataset or find our code helpful, please consider citing our paper:
```
@article{wang2021integrating,
  title={Integrating Expert Knowledge with Domain Adaptation for Unsupervised Fault Diagnosis},
  author={Wang, Qin and Taal, Cees and Fink, Olga},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021}
}
```
