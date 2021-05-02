# Multilingual Negation Scope Resolution for Clinical Text
This repository contains code for identifying negation scopes in multilingual clinical data using a transformer-based multilingual language model, as described in the paper [Multilingual Negation Scope Resolution for Clinical Text](https://www.aclweb.org/anthology/2021.louhi-1.2.pdf) (LOUHI 2021).

The code in this repository is a re-implementation of the original code that was used for the experiments described in the paper. The original code is based on the [mt-dnn](https://github.com/namisan/mt-dnn) framework, and it was re-implemented in order to make it less complex. If you want to obtain the code that was used for the experiments in the paper, please contact the authors at ```mrkhartmann4@gmail.com``` 
## Using trained models to predict negation scopes

The results that these models achieve are comparabel to the results reported in the paper, and can be found [here](docs/results.md)
### Identifying negation cues using a pre-defined cue list
### Predicting negation scopes given marked negation cues
## Training models

Information about how to pre-process the data will be added here in the future. In the meantime, please contact the authors is you have any questions about dataset preprocessing.

## Evaluating models
### Contact the Authors
If you have questions, please contact the corresponding author at  ```mrkhartmann4@gmail.com``` . \

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
