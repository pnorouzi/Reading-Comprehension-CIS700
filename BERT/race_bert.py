import numpy as np
import torch
import os
import json
import pickle
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

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
        for i in range(len(data['answers'])):
            entity = {}
            true_label = ord(data['answers'][i]) - ord('A')
            question = data['questions'][i]
            options = data['options'][i]
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
    # Parameters
    #The path to data
    data_dir = 'RACE'
    #The BERT model to train, can be amongst bert-base-uncased, bert-large-uncased,  bert-base-cased
    bert_model = 'bert-large-uncased'
    # The output directory to store model checkpoints
    output_dir = 'large_models'
    # The max total input length after tokenization
    # Longer Sequences are truncated
    # Shorter ones are padded
    max_seq_length = 320
    # The batch size (read https://github.com/google-research/bert/issues/38 to realize why 8 actually means 32 to the model)
    # On high level 4 options are taken as different inputs to the model, so 8*4 = 32
    train_batch_size = 8
    # The learning rate
    learning_rate = 1e-5
    # Number of epochs        print(data['id'] + " processed")

    num_epochs = 2
    # Gradient accumulation steps, loss scale and device setup
    gradient_accumulation_steps = 8
    loss_scale = 128
    warmup_proportion = 0.1
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    os.makedirs(output_dir, exist_ok=True)
    # Build the tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    # Process the dataset and store it once computed
    if os.path.isfile('train_data_processed.txt'):
        print('Loading train data')
        with open('train_data_processed.txt',"rb") as fp:
            train_data = pickle.load(fp)
        print('Loaded train data')
    else:
        print('Processing train data')
        train_data = process_race(data_dir+'/','train/')
        with open('train_data_processed.txt',"wb") as fp:
            pickle.dump(train_data, fp)
        print('Processed train data')

    # Find features and store it once computed
    if os.path.isfile('train_features.txt'):
        print('Loading train features')
        with open('train_features.txt',"rb") as fp:
            train_features = pickle.load(fp)
        print('Loaded train features')
    else:
        print('Converting to features')
        train_features = convert_to_bert_style(train_data, tokenizer, max_seq_length)
        with open('train_features.txt','wb') as fp:
            pickle.dump(train_features, fp)
        print('Converted to features')

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)
    # Find the total number of steps for training
    num_train_steps = int(len(train_data) / train_batch_size / gradient_accumulation_steps * num_epochs)
    # Initialize the model
    model = BertForMultipleChoice.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /'model', num_choices = 4)
    model.to(device)
    global_step = 0
    parameters = get_params(model)
    optimizer = BertAdam(parameters, lr = learning_rate, warmup = warmup_proportion, t_total = num_train_steps)
    train_data = prepare_data(train_features)
    sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = train_batch_size)
    model.train()
    for epochs in range(num_epochs):
        training_loss = 0
        num_training_examples, num_training_steps = 0,0
        logger.info("Training Epoch: {}/{}".format(epochs+1, int(num_epochs)))
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            loss = loss/gradient_accumulation_steps
            training_loss += loss.item()
            num_training_examples += input_ids.size(0)
            num_training_steps += 1
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            del batch
            if global_step%100==0:
                logger.info("Training loss: {}, global step: {}".format(training_loss/num_training_steps, global_step))
            break
        ## evaluate on dev set
        if global_step%1000 == 0:
                dev_data = process_race(data_dir+'/','dev/')
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

                result = {'dev_loss': dev_loss, 'dev_accuracy': dev_accuracy, 'global_step': global_step, 'loss': training_loss/num_training_steps}


                output_eval_file = os.path.join(output_dir, "dev_results.txt")

                with open(output_eval_file, "a+") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))


    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

if __name__=='__main__':
    main()
