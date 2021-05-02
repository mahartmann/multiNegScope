# Multilingual Negation Scope Resolution for Clinical Text
This repository contains code for identifying negation scopes in multilingual clinical data using a transformer-based multilingual language model (mbert), as described in the paper [Multilingual Negation Scope Resolution for Clinical Text](https://www.aclweb.org/anthology/2021.louhi-1.2.pdf) (LOUHI 2021).

The code in this repository is a re-implementation of the original code that was used for the negation scope resolution experiments described in Paper. The original code is based on the [mt-dnn](https://github.com/namisan/mt-dnn) framework, and it was re-implemented in order to make it less complex. If you want to obtain the code that was used for the experiments in the paper, please contact the authors at mrkhartmann4@gmail.com .
## Using trained models to predict negation scopes
### Identifying negation cues using a pre-defined cue list
### Predicting negation scopes given marked negation cues
## Training models
## Evaluating models
### Contact the Authors
If you have questions, please contact the corresponding author at  mrkhartmann4@gmail.com .
Our work can be cited using
@inproceedings{hartmann-sogaard-2021-multilingual, \
    title = "Multilingual Negation Scope Resolution for Clinical Text",
    author = "Hartmann, Mareike  and
      S{\o}gaard, Anders",
    booktitle = "Proceedings of the 12th International Workshop on Health Text Mining and Information Analysis",
    month = apr,
    year = "2021",
    address = "online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.louhi-1.2",
    pages = "7--18",
    abstract = "Negation scope resolution is key to high-quality information extraction from clinical texts, but so far, efforts to make encoders used for information extraction negation-aware have been limited to English. We present a universal approach to multilingual negation scope resolution, that overcomes the lack of training data by relying on disparate resources in different languages and domains. We evaluate two approaches to learn from these resources, training on combined data and training in a multi-task learning setup. Our experiments show that zero-shot scope resolution in clinical text is possible, and that combining available resources improves performance in most cases.",
}
