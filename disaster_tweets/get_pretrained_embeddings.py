import numpy as np
import os
import pandas as pd
import re
import string
#pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001F923"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', text)


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\d', ' ', text)
    text = re.sub(' {2,}', ' ', text)
    return text


def prepare_data(data, text_col_name):
    
    df = data.copy()
    df[text_col_name] = df[text_col_name].apply(lambda x: remove_emoji(x))
    df[text_col_name] = df[text_col_name].apply(lambda x: clean_text(x))

    # drop if the string has become empty
    df = df.drop(np.where([x == ' ' for x in df[text_col_name].values])[0], axis=0)
    text = df.reset_index(drop=True)[text_col_name]  # indexing is important dataloader

    return text


def get_sentence_embedding(sentence, tokenizer, model, device):
    '''
    :param sentence:
    :param tokenizer:
    :param model:
    :param device:
    :return:
    '''
    encoding = tokenizer(
        sentence,
        add_special_tokens=True,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)  # `'tf'`: Return TensorFlow :obj:`tf.constant``'pt'`: Return PyTorch :obj:`torch.Tensor`

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask'].type(torch.cuda.FloatTensor)

    model.eval()
    with torch.no_grad():
        out = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
    return out.mean(1).detach().cpu().numpy()


def get_batched_embeddings(text, tokenizer, model, device, batch_size=64):
    '''

    :param text:
    :param tokenizer:
    :param model:
    :param embedding_size:
    :param device:
    :param batch_size:
    :return:

    '''
    from torch.utils.data import DataLoader

    text_embeddings = []

    dataloader = DataLoader(
        text,
        batch_size=batch_size,
        num_workers=8)

    model.eval()
    for d in tqdm(dataloader):

        encoding = tokenizer(
            d,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt').to(device)

        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            out = model.forward(**encoding)[0]

        sent_len = (attention_mask).sum(1, keepdim=True)
        token_embeddings = out * attention_mask[:, :, None]
        batch_embeddings = (token_embeddings.sum(dim=1) / sent_len).detach().cpu().numpy()
        text_embeddings.append(batch_embeddings)

    return np.vstack(text_embeddings)

def main():
    # Data loading and cleaning
    dir_data = "./datasets"
    file_name = "train.csv"
    df = pd.read_csv(os.path.join(dir_data, file_name))
    df = df.sample(100, random_state=34)  # for testing

    text = prepare_data(df, 'text')  # comment

#    print(torch.__version__)
#    print(torch.cuda.is_available())

    model_name = "gpt2-medium"
    device_name = "cuda:0"
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = '<pad>'
    model = AutoModel.from_pretrained(model_name).to(device)

    text_embeddings = []

    for sentence in tqdm(text):
        text_embeddings.append(get_sentence_embedding(sentence, tokenizer, model, device))

    text_embeddings1 = np.vstack((text_embeddings))
    print(text_embeddings1.sum())

    text_embeddings2 = get_batched_embeddings(text, tokenizer, model, device, batch_size=10)
    print(text_embeddings2.sum())

if __name__ == "__main__":
    main()