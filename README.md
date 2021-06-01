# Multilingual Negation Scope Resolution for Clinical Text
This repository contains code for identifying negation scopes in multilingual clinical data using a transformer-based multilingual language model, as described in the paper [Multilingual Negation Scope Resolution for Clinical Text](https://www.aclweb.org/anthology/2021.louhi-1.2.pdf) (LOUHI 2021).

The code in this repository is a re-implementation of the original code that was used for the experiments described in the paper. The original code is based on the [mt-dnn](https://github.com/namisan/mt-dnn) framework, and it was re-implemented in order to make it less complex. 
The results that these models achieve are comparable to the results reported in the paper, and can be found [here](docs/results.md) If you would like to obtain the code that was used for the original experiments in the paper, please contact the authors at ```mrkhartmann4@gmail.com``` 
### Installing required packages
Install the conda package manager following the [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Create a virtual env and install the required packages using
```
conda create --name multinegscope 
conda activate multinegscope
pip install -r requirements.txt 
```
## Using trained models to predict negation scopes
You can use our pretrained model to predict negation scopes in your data of interest. Negation scopes are predicted in two stages: First, negation cues are identified based on a pre-defined negation cue list. Second, negation scopes are marked given the identified negation cues. In the following, we describe the steps you need to follow.


### 1) Identifying negation cues using a pre-defined cue list
```
python tag_negation_cues.py \
--cue_list ./data/cues/cues_danish.txt \
--input_file ./examples/example.jsonl
```

The script creates two files ```./examples/example#cues.jsonl``` and ```./examples/example#cues.html```. The first can be used as input file for the negation scope resolution model, the latter visualizes the detected negation cues in red when opened in a web browser. 

### 2) Predicting negation scopes given marked negation cues
Download the pre-trained negation scope resolution model from [here](https://drive.google.com/file/d/13m-e1n4fovRpNtp6EHws9T6JNS0ee8QP/view?usp=sharing) and unzip it. 

```
python predict_negscopes.py \
--model_checkpoint ./pretrained_model_v1/model.pt  \
--datapath ./examples \
--test_datasets example#cues.jsonl \
--outdir ./examples
```

The script creates two files ```./examples/example#cues#scopes.jsonl``` and ```./examples/example#cues#scopes.html```.


## Training models
Coming soon

Information about how to pre-process the data will be added here in the future. In the meantime, please contact the authors is you have any questions about dataset preprocessing.

## Evaluating models
Coming soon 
### Contact the Authors
If you have questions, please contact the corresponding author at  ```mrkhartmann4@gmail.com``` .

Our work can be cited using \
@inproceedings{hartmann-sogaard-2021-multilingual, \
    title = "Multilingual Negation Scope Resolution for Clinical Text", \
    author = "Hartmann, Mareike  and
      S{\o}gaard, Anders",
    booktitle = "Proceedings of the 12th International Workshop on Health Text Mining and Information Analysis", \
    month = apr, \
    year = "2021",\
    address = "online", \
    publisher = "Association for Computational Linguistics", \
    url = "https://www.aclweb.org/anthology/2021.louhi-1.2", \
    pages = "7--18"}
