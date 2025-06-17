import numpy as np
import torch
import re

import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy

import random

from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel
import os
from fnmatch import fnmatch
from torch.cuda.amp import autocast
import json
from pathlib import Path
HOME_PATH = str(Path.home())

PATH = os.path.join(HOME_PATH, 'edos_task10')

FOLDER_NAME = 'exp_with_definition_task_b'
m_name = 'pretrained3_ro_de_task_b_original_with_definitions_variations_five_classes_eval_ro_de'

NUM_CLASSES = 4
TEST = False

seeds = [1000, 2000]
seed_idx = 1
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

DEBERTA_MODEL_NAME = 'microsoft/deberta-v3-large'
DEBERTA_PRETRAINED_MODEL_NAME = \
    os.path.join(PATH, 'deberta_pretraining_2m','deberta-v3-pretrained', 'deberta-checkpoint-3')

ROBERTA_MODEL_NAME = 'roberta-large'
ROBERTA_PRETRAINED_MODEL_NAME = \
    os.path.join(PATH, 'deberta_pretraining_2m','roberta-pretrained', 'roberta_checkpoint-epoch_3')


d_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
print(f"load model from : {DEBERTA_PRETRAINED_MODEL_NAME}")
d_model = AutoModel.from_pretrained(DEBERTA_PRETRAINED_MODEL_NAME, ignore_mismatched_sizes=True, output_hidden_states=True)


r_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
r_model = AutoModel.from_pretrained(ROBERTA_PRETRAINED_MODEL_NAME, output_hidden_states=True)

print(f"Roberta Model Hidden Size: {r_model.config.hidden_size}")
print(f"Deberta Model Hidden Size: {d_model.config.hidden_size}")


task_c_labels = ['2.2 aggressive and emotive attacks',
                 '2.1 descriptive attacks',
                 '2.3 dehumanising attacks & overt sexual objectification',
                 '3.1 casual use of gendered slurs, profanities, and insults',
                 '3.2 immutable gender differences and gender stereotypes',
                 '4.1 supporting mistreatment of individual women',
                 '4.2 supporting systemic discrimination against women as a group',
                 '1.1 threats of harm',
                 '1.2 incitement and encouragement of harm',
                 '3.3 backhanded gendered compliments',
                 '3.4 condescending explanations or unwelcome advice']

task_b_labels = ['2. derogation',
                 '3. animosity',
                 '4. prejudiced discussions',
                 '1. threats, plans to harm and incitement']


mapping = {
    "1.1 threats of harm": task_b_labels[3],
    "1.2 incitement and encouragement of harm": task_b_labels[3],

    "2.1 descriptive attacks": task_b_labels[0],
    "2.2 aggressive and emotive attacks": task_b_labels[0],
    "2.3 dehumanising attacks & overt sexual objectification": task_b_labels[0],

    "3.1 casual use of gendered slurs, profanities, and insults": task_b_labels[1],
    "3.2 immutable gender differences and gender stereotypes": task_b_labels[1],
    "3.3 backhanded gendered compliments": task_b_labels[1],
    "3.4 condescending explanations or unwelcome advice": task_b_labels[1],

    "4.1 supporting mistreatment of individual women": task_b_labels[2],
    "4.2 supporting systemic discrimination against women as a group": task_b_labels[2]


}


def mapping_str_to_str(x):
    return mapping[x]


def load_data_and_split(file_path):
    df = pd.read_csv(file_path)
    df = df[df.label_sexist =='sexist']

    t_data = df[df['split'] == "train"]
    v_data = df[df['split'] == "dev"]
    te_data = df[df['split'] == "test"]

    t_data.drop(['split', 'rewire_id', 'label_sexist', 'label_vector'], axis=1, inplace=True)
    v_data.drop(['split', 'rewire_id', 'label_sexist', 'label_vector'], axis=1, inplace=True)
    te_data.drop(['split', 'rewire_id', 'label_sexist', 'label_vector'], axis=1, inplace=True)

    return df, t_data, v_data, te_data

tr_data = pd.read_csv(
    os.path.join(PATH,FOLDER_NAME, 'data','variations_augmentation_gpt4o_five_classes.csv'))

tr_data["label_category"] = tr_data["label_vector"].map(lambda x: mapping_str_to_str(x))
tr_data = tr_data[['generated_text', 'label_category']]
tr_data.rename(columns={'generated_text': 'text'}, inplace=True)

print(tr_data.shape)
print(tr_data.label_category.value_counts())

full_data, train_data, eval_data, test_data = load_data_and_split(
    os.path.join(PATH, FOLDER_NAME, 'data', 'edos_labelled_aggregated.csv'))

print(train_data.shape, eval_data.shape)
train_data = pd.concat([train_data, tr_data, eval_data], ignore_index=True)


def tolist(tensor):
    return tensor.detach().cpu().tolist()


def map_names_2_ids(names):
    A = dict()
    B = dict()
    for id, name in enumerate(names):
        A[name] = id
        B[id] = name
    return A, B


train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

label_vector_raw = np.unique(train_data['label_category']).tolist()
label_vector_map, vector_label_map = map_names_2_ids(label_vector_raw)
train_data['label_ids'] = [label_vector_map[i[1]] for i in train_data['label_category'].items()]

test_data['label_ids'] = [label_vector_map[i[1]] for i in test_data['label_category'].items()]
num_labels_C = len(train_data['label_category'].unique())

train_dataframe = train_data
test_dataframe = test_data
print(f"Data after Preprocessing: train_dataframe: {train_dataframe.shape} "
      f" test_dataframe: {test_dataframe.shape}")

print(train_dataframe.label_ids.value_counts())
print(test_dataframe.label_ids.value_counts())

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.array(list(range(num_labels_C))),
                                     y=train_data['label_ids'].values.tolist()).tolist()

print(class_weights)
print(f"Data after Preprocessing: train_dataframe: {train_dataframe.shape} "
      f" test_dataframe: {test_dataframe.shape}")


class SexistDataset(Dataset):

  def __init__(self, dataframe, d_tokenizer, r_tokenizer, max_length=100, is_test=False):
    self.dataframe = dataframe
    self.d_tokenizer = d_tokenizer
    self.r_tokenizer = r_tokenizer
    self.max_length = max_length
    self.is_test = is_test

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    sample = self.dataframe.loc[idx]
    tokenized_text = self.d_tokenizer(
          sample['text'],
          max_length=self.max_length,
          padding='max_length',
          truncation='only_first',
          return_tensors='pt')

    tokenized_text_r = self.r_tokenizer(
          sample['text'],
          max_length=self.max_length,
          padding='max_length',
          truncation='only_first',
          return_tensors='pt')
    tokenized_text['r_input_ids'] = torch.tensor(tokenized_text_r['input_ids'])
    tokenized_text['r_attention_mask'] = torch.tensor(tokenized_text_r['attention_mask'])

    if not self.is_test:
      labels = torch.LongTensor([sample['label_ids']])
      tokenized_text['labels'] = labels
    return tokenized_text


class SexistModel(nn.Module):

    def __init__(self, d_model, r_model):
        super().__init__()
        self.d_model = d_model
        # self.dropout = nn.Dropout(p=.1)
        self.r_model = r_model

        d_hidden_size = self.d_model.config.hidden_size
        r_hidden_size = self.r_model.config.hidden_size

        self.combine_head = nn.Linear(d_hidden_size + r_hidden_size, num_labels_C)

    def forward(self, data, c_batch_size):
        ids = data.pop('r_input_ids').view(c_batch_size, -1)
        mask = data.pop('r_attention_mask').view(c_batch_size, -1)

        hidden_d = self.d_model(**data).last_hidden_state
        cls = hidden_d[:, 0, :]

        hidden_r = self.r_model(input_ids=ids, attention_mask=mask).last_hidden_state
        cls_r = hidden_r[:, 0, :]
        x = torch.cat((cls, cls_r), 1)
        x = self.combine_head(x)

        x = x.view(-1, num_labels_C)
        return x, (hidden_d, hidden_r)


best_f1 = 0.


def train(dataloader, model, device, loss_fn, optimizer, scheduler):

  model.train()
  loss_collection = [[], []]

  for step, data in enumerate(dataloader):

    c_batch_size = data['input_ids'].shape[0]
    labels = data.pop('labels').to(device).view(-1)

    for key in data:
      data[key] = data[key].to(device).view(c_batch_size, -1)

    optimizer.zero_grad()
    logits, _ = model(data, c_batch_size)
    ce_loss = loss_fn(logits, labels)
    ce_loss.backward()
    loss_collection[0].append(ce_loss.item())

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    if len(loss_collection[0]) % log_step == 0:
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] |'
            f' CE Loss {round(sum(loss_collection[0]) / (len(loss_collection[0]) + 1e-8), 4)}')

      print('------------------------------------------------')
      loss_collection = [[] for _ in range(2)]


def eval(dataloader, model, device):
    model.eval()

    all_preds = list()
    with torch.no_grad():
        for data in dataloader:
          c_batch_size = data['input_ids'].shape[0]
          for key in data:
            data[key] = data[key].to(device).view(c_batch_size, -1)
          labels = data.pop('labels').to(device).view(-1)
          logits, _ = model(data, c_batch_size)
          preds = tolist(logits.argmax(1).view(-1))
          all_preds.extend(preds)
    return all_preds


def test(dataloader, model, device):
    model.eval()
    all_preds = list()
    with torch.no_grad():
        for data in dataloader:
            c_batch_size = data['input_ids'].shape[0]
            for key in data:
                data[key] = data[key].to(device).view(c_batch_size, -1)
                # with autocast():
            logits, _ = model(data, c_batch_size)
            preds = tolist(logits.argmax(1).view(-1))
            all_preds.extend(preds)
    return all_preds


epochs = 30
lr = 6e-6
beta_1 = .9
beta_2 = .98
eps = 1e-6
log_step = 200
batch_size = 10
weight_decay = 5e-3
max_length = 150
MAX_LEN = max_length


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sexist_model = SexistModel(d_model, r_model).to(device)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device)).to(device)
loss_collection = []

train_dataset = SexistDataset(train_dataframe, d_tokenizer, r_tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SexistDataset(test_dataframe, d_tokenizer, r_tokenizer, max_length, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


opt_step = 0
optimization_steps = epochs * len(train_dataloader)
warmup_ratio = .0
warmup_steps = int(optimization_steps * warmup_ratio)


optimizer = AdamW(sexist_model.parameters(), lr=lr, betas=(beta_1,beta_2),
                  eps=eps, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=optimization_steps)

best_icm = 0.
best_model = None
transformers.logging.set_verbosity_error()


checkpoint_dir = os.path.join(PATH, FOLDER_NAME, 'models', m_name)
filename = os.path.join(checkpoint_dir, 'best_ch.pt')


if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


def save_model(epoch, model, optimizer, scheduler):
  filename = os.path.join(checkpoint_dir, 'best_ch.pt')
  torch.save(
      {'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict()},
        filename)


def load_model():
  if os.path.exists(filename):
    saved_dict = torch.load(filename)
    return True, saved_dict
  else:
    return False, None


def early_stop(scores, current_score, patience):
  if len(scores) < patience:
    return False
  else:
    for score in scores[-patience: ]:
      if current_score > score:
        return False
    return True


all_f1 = list()
patience = 4
torch.cuda.empty_cache()

if not TEST:
    print("Start Training...")
    for epoch in range(epochs):
        train(train_dataloader, sexist_model, device, loss_fn, optimizer, scheduler)

        preds_test = test(test_dataloader, sexist_model, device)
        f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_test, average='macro')

        if f1_macro_test > best_f1:
            best_f1 = f1_macro_test
            best_preds = preds_test
            save_model(epoch + 1, sexist_model, optimizer, scheduler)

        print(f'EPOCH [{epoch + 1}/{epochs}] | Test F1-Macro {round(f1_macro_test * 100, 2)}')
        print(f'EPOCH [{epoch + 1}/{epochs}] | Test Best F1-Macro {round(best_f1 * 100, 2)}')
        print(confusion_matrix(test_dataframe['label_ids'].values.tolist(), preds_test))



        # if early_stop(all_f1, best_icm, patience):
        #     break
        # else:
        #     print('not early stopping')

print(f"Loading check points from {filename}")
_, saved_dict = load_model()
sexist_model.load_state_dict(saved_dict['model_state_dict'])
preds_test = test(test_dataloader, sexist_model, device)
f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_test, average='macro')

print(f'Test F1-Macro {round(f1_macro_test * 100, 2)}')

test_preds_names = [vector_label_map[label_id] for label_id in preds_test]
test_dataframe['label_pred'] = test_preds_names
test_dataframe['pred_id'] = preds_test

csv_file_name = f"{m_name}_preds.csv"
test_dataframe.to_csv(os.path.join(PATH, FOLDER_NAME, 'csv', csv_file_name))


copy_and_paste = f"f1_macro_test: {f1_macro_test}, model_name: {m_name} max_len: {MAX_LEN} LR: {lr} " \
                 f"batch_size: {batch_size} epochs: {epochs}\n"
with open(os.path.join(PATH, FOLDER_NAME,'info.txt'), 'a') as f:
    f.write(copy_and_paste)
