import numpy as np
import torch
import os
import json
import pickle
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from gensim.summarization.summarizer import summarize
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
def process_race(root, path):
    path = root + path
    dataset = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(root,filename)
            with open(file, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                data_raw['filename'] = file
                dataset.append(data_raw)
    processed_data = []
    for data in dataset:
        article = data['article']
        #if len(article.split(' ')) > 270:
        #    article = summarize(article, ratio = 0.4)
        for i in range(len(data['answers'])):
            entity = {}
            true_label = ord(data['answers'][i]) - ord('A')
            question = data['questions'][i]
            options = data['options'][i]
            # for j in options:
            #     if len(j.split(' ')) > 7:
            #         article = summarize(article, ratio = 0.4)
            #         break
            entity['id'] = data['filename']+'-'+str(i)
            entity['article'] = article
            entity['question'] = question
            entity['option'] = options
            entity['label'] = true_label
            processed_data.append(entity)
    return processed_data

def truncate_seq(token_a, token_b, max_seq_length):
    while True:
        total_length = len(token_a) + len(token_b)
        if total_length <= max_seq_length:
            break
        if len(token_a) > len(token_b):
            token_a.pop()
        else:
            token_b.pop()

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
def convert_to_bert_style(processed_data, tokenizer, max_seq_length):
    '''
    We will use the style suggested in this issue:
    https://github.com/google-research/bert/issues/38

    The input will be of type [CLS] + Article + [SEP] + Question + [SEP] + Option
    for each option. The model will output a single value for this input, we will
    then do a softmax over these outputs to get the final answer
    '''

    features = []
    for data in processed_data:
        # Get the tokens for the article
        article_tokens = tokenizer.tokenize(data['article'])
        # Get the token for the question
        question_tokens = tokenizer.tokenize(data['question'])
        options_features = []
        for i in data['option']:
            # Create a copy for option 0 because we are going to amend it
            article_tokens_option_i = article_tokens[:]
            # Combine the question tokens with the option_0 token
            options_i_tokens = question_tokens + tokenizer.tokenize(i)
            # Truncate the seq to max_seq_length-3 because we are going to add 3 tokens, [CLS] and 2 [SEP]
            truncate_seq(article_tokens_option_i, options_i_tokens, max_seq_length-3)
            # Generate the required token
            tokens_option_i = ["[CLS]"] + article_tokens_option_i + ["[SEP]"] + options_i_tokens + ["[SEP]"]
            # Generate the segment indices's for separating out article and options
            segment_indices_option_i = [0]*(len(article_tokens_option_i)+2) +[1]*(len(options_i_tokens)+1)
            # Generate id's related to tokens
            input_ids = tokenizer.convert_tokens_to_ids(tokens_option_i)
            # Generate the input mask
            input_mask = [1]*len(input_ids)
            # Zero pad to make everything equal to max_seq_length
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_indices_option_i+=padding
            options_features.append((tokens_option_i, input_ids, input_mask, segment_indices_option_i))
        feature = {}
        feature['id'] = data['id']
        feature['options_features'] =  options_features
        feature['label'] = data['label']
        features.append(feature)

    return features

def prepare_data(data_features):
    input_ids = [[ options[1] for options in features['options_features'] ]  for features in data_features]
    input_masks = [[ options[2] for options in features['options_features']] for features in data_features]
    segment_ids = [[ options[3] for options in features['options_features']] for features in data_features]
    labels = [features['label'] for features in data_features]
    input_ids = torch.tensor(input_ids, dtype = torch.long)
    input_masks = torch.tensor(input_masks, dtype = torch.long)
    segment_ids = torch.tensor(segment_ids, dtype = torch.long)
    labels = torch.tensor(labels, dtype = torch.long)
    return TensorDataset(input_ids, input_masks, segment_ids, labels)

def get_params(model):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    return optimizer_grouped_parameters

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    data_dir = 'RACE'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    bert_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    model = BertForMultipleChoice.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /'model', num_choices = 4)
    model.to(device)
    PATH = os.path.join('large_models','pytorch_model_2epochs.bin')
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    max_seq_length = 200
    output_dir = 'large_models'
    dev_data = process_race(data_dir+'/','test')
    dev_features = convert_to_bert_style(dev_data, tokenizer, max_seq_length)
    dev_data = prepare_data(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler = dev_sampler, batch_size = 1)
    model.eval()
    dev_loss, dev_accuracy = 0,0
    num_eval_examples, num_eval_steps = 0,0
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
            print(100*step/len(dev_dataloader))
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
        dev_loss += tmp_eval_loss.mean().item()
        dev_accuracy += tmp_eval_accuracy
        num_eval_examples += input_ids.size(0)
        num_eval_steps += 1
        del batch

    dev_loss = dev_loss/num_eval_steps
    dev_accuracy = dev_accuracy/num_eval_examples
    output_eval_file = os.path.join(output_dir, "dev_results.txt")
    result = {'dev_loss': dev_loss, 'dev_accuracy': dev_accuracy}

    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    # print(dev_accuracy)

if __name__ == '__main__':
    main()
