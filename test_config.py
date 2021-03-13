test_config = {
'output_attentions' : False  ,
'output_hidden_states' : False  ,
'output_past' : True  ,
'torchscript' : False  ,
'use_bfloat16' : False  ,
'pruned_heads' : {}  ,
'is_decoder' : False  ,
'max_length' : 20  ,
'do_sample' : False  ,
'num_beams' : 1  ,
'temperature' : 1.0  ,
'top_k' : 50  ,
'top_p' : 1.0  ,
'repetition_penalty' : 1.0  ,
'bos_token_id' : 0  ,
'pad_token_id' : 0  ,
'eos_token_ids' : 0  ,
'length_penalty' : 1.0  ,
'num_return_sequences' : 1  ,
'architectures' : ['BertForTokenClassification']  ,
'finetuning_task' : None  ,
'num_labels' : 9  ,
'id2label' : {0: 'LABEL_0', 1: 'LABEL_1'}  ,
'label2id' : {'LABEL_0': 0, 'LABEL_1': 1}  ,
'vocab_size' : 119547  ,
'hidden_size' : 8  ,
'num_hidden_layers' : 1  ,
'num_attention_heads' : 2  ,
'hidden_act' : 'gelu'  ,
'intermediate_size' : 8  ,
'hidden_dropout_prob' : 0.1  ,
'attention_probs_dropout_prob' : 0.1  ,
'max_position_embeddings' : 512  ,
'type_vocab_size' : 2  ,
'initializer_range' : 0.02  ,
'layer_norm_eps' : 1e-12  ,
'model_type' : 'bert'  , }