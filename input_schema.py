INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["hello i'm not sad with your service"]
    },
    "system_prompt": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["you are sentiment analyzer, you should say yes or no, and response in json"]
    },
    "temperature": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.1]
    },
    "top_p": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.1]
    },
    "max_tokens": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [128]
    },
    "top_k":{
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [40]
    }
}
