# CoGANPPIS_package
## Overview
This is the python package of CoGANPPIS, a coevolution-enhanced global attention neural network for protein-protein interaction site prediction. It contains all the necessary components from feature generation to final prediction so that you only need to input raw protein sequences to obtain the prediction results.

## Installation
```bash
git lfs clone git@github.com:Slam1423/CoGANPPIS_package.git
```

## Requirements:
- Linux/MacOS
- python 3.6.9+
- biopython 1.79
- pybiolib 1.1.988
- pytorch 1.10.0
- numpy 1.19.5
- pandas 1.1.5
- scipy 1.5.4

## Hyperparameters:
`-f`: The dataset name.
usage example: `-f example`

`-d`: Protein database for Blast.
usage example: `-d nr`

`-n`: Number of sequences in multiple sequence alignments. (preferably >= 1000)
usage example: `-n 1000`

## How to run
First, you have to copy your dataset into `CoGANPPIS_package/raw_input_sequences/` with the name of `datasetname + _seq.txt` in the form of fasta. For example, your dataset name is `example`, then you should save the sequences into `example_seq.txt` in the form of fasta as follows:

```bash
>1acb_I
KSFPEVVGKTVDQAREYFTLHYPQYDVYFLPEGSPVTLDLRYNRVRVFYNPGTNVVNHVPHVG
>1ay7_A
DVSGTVCLSALPPEATDTLNLIASDGPFPYSQDGVVFQNRESVLPTQSYGYYHEYTVITPGARTRGTRRIITGEATQEDYYTGDHYATFSLIDQTC
>1c1y_B
SNTIRVFLPNKQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVFRLLHEHKGKKARLDWNTDAASLIGEELQVDFL

```

Now you can run the package with the following codes:

```bash
cd CoGANPPIS_package/
python3 main.py -f example -d nr -n 1000
```

The predicted labels will be saved to `predict_result_dir/example_predict_result.pkl` in the form of list in python, whose elements sequentially refer to the predicted labels of the residues and length is equal to the total number of residues of the input sequences. 

