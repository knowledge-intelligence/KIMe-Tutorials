from pathlib import Path

def get_config():
    return {
        "batch_size": 12,   # For A6000(48G)
        "num_epochs": 30,
        "lr": 10**-4,
        "seq": 1300,
        "d_model": 512,
        "datasource": 'lemon-mint/korean_english_parallel_wiki_augmented_v1',
        "lang_src_fieldname": "english",
        "lang_tgt_fieldname": "korean",
        "model_folder": "KIMe_BasicModel/Transformer/lemon-mint/weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "KIMe_BasicModel/Transformer/tokenizer_{0}.json",
        "experiment_name": "KIMe_BasicModel/Transformer/runs/tmodel"
    }

# def get_config():
#     return {
#         "batch_size": 64,   # For A6000(48G)
#         "num_epochs": 21,
#         "lr": 10**-4,
#         "seq": 350,
#         "d_model": 512,
#         "datasource": 'opus_books',
#         "lang_src": "en",
#         "lang_tgt": "it",
#         "model_folder": "weights",
#         "model_basename": "tmodel_",
#         "preload": "latest",
#         "tokenizer_file": "tokenizer_{0}.json",
#         "experiment_name": "runs/tmodel"
#     }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])