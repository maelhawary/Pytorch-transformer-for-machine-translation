def get_config():
    return {
        "dropout":0.2,
        "batch_size": 8,
        "number_of_layers":6,
        "number_of_heads":8,
        "eval_iter" : 50,
        "d_att_weigh":512,
        "d_FFN":2048,
        "num_epochs": 20,
        "lr": 10**-4,
        "split":0.9,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "tokenizer_file": "tokenizer_{0}.json"
    
    }
