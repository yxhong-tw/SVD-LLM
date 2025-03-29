from transformers import AutoTokenizer

from utils.data_utils import get_calib_train_data


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='/workspace/models/Llama-3.2-3B',
)
train_dataset = get_calib_train_data(
    name='wikitext2',
    tokenizer=tokenizer,
    nsamples=10,
)
