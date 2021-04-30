from transformers import BertTokenizer

def setup_customized_tokenizer(model, tokenizer_class, do_lower_case, config):
    additional_tokens = []
    for i in range(10):
        additional_tokens.append('[START{}]'.format(i))
        additional_tokens.append('[END{}]'.format(i))
    additional_tokens.append('@CHEMICAL$')
    additional_tokens.append('@GENE$')
    additional_tokens.append('@DRUG$')
    additional_tokens.append('@DISEASE$')
    additional_tokens.append('@CONCEPT$')
    additional_tokens.append('[CUE]')
    if model == 'bert-base-multilingual-cased':
        tokenizer = tokenizer_class.from_pretrained(model, vocab_file=config.get('Files', 'mbertvocab'),
                                    do_lower_case=do_lower_case, additional_special_tokens=additional_tokens)
    if model == 'bert-base-cased':
        tokenizer = tokenizer_class.from_pretrained(model, vocab_file=config.get('Files', 'bertvocab'),
                                                    do_lower_case=do_lower_case,
                                                    additional_special_tokens=additional_tokens)
    elif model == 'spanish-bert-cased':
        tokenizer = BertTokenizer(vocab_file=config.get('Files', 'spanishbertvocab'),
                                                   do_lower_case=do_lower_case,
                                                    additional_special_tokens=additional_tokens)
    elif model == 'test_config':
        tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path='bert-base-multilingual-cased', vocab_file=config.get('Files', 'mbertvocab'),
                                                    do_lower_case=do_lower_case,
                                                    additional_special_tokens=additional_tokens)

    return tokenizer