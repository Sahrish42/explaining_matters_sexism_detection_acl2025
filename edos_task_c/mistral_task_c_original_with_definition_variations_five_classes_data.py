import numpy as np

import re

import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import random

from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer
import os
from fnmatch import fnmatch

from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from pathlib import Path
HOME_PATH = str(Path.home())

PATH = os.path.join(HOME_PATH, 'edos_task10')
m_name = 'mistral_task_c_original_with_definition_variations_five_classes_data_eval_mistral'
FOLDER_NAME = 'edos_task_c'

model_name = "mistralai/Mistral-7B-v0.1"
VARIATIONS_AND_ORIGINAL = True
TEST = True
CHECK_POINTS = False
NUM_CLASSES = 11
LOAD_CHECKPOINTS = False

seeds = [1000, 2000]
seed_idx = 1
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def load_data_and_split(file_path):
    df = pd.read_csv(file_path)
    df = df[df.label_sexist == 'sexist']

    t_data = df[df['split'] == "train"]
    v_data = df[df['split'] == "dev"]
    te_data = df[df['split'] == "test"]

    t_data.drop(['split', 'rewire_id', 'label_sexist', 'label_category'], axis=1, inplace=True)
    v_data.drop(['split', 'rewire_id', 'label_sexist', 'label_category'], axis=1, inplace=True)
    te_data.drop(['split', 'rewire_id', 'label_sexist', 'label_category'], axis=1, inplace=True)

    return df, t_data, v_data, te_data


if VARIATIONS_AND_ORIGINAL:
    print('----------------------- Original, Variations and Eval Data------------------------')
    full_data, train_data, eval_data, test_data = load_data_and_split(
        os.path.join(PATH, FOLDER_NAME, 'data', 'edos_labelled_aggregated.csv'))
    print(f"Original data: {train_data.shape}")

    df = pd.read_csv(
        os.path.join(PATH, FOLDER_NAME, 'data', 'variations_augmentation_gpt4o_five_classes.csv'))

    tr_data = df[['generated_text', 'label_vector']]
    tr_data.rename(columns={'generated_text': 'text'}, inplace=True)
    print(f"causal analysis data: {tr_data.shape}")

    train_data = pd.concat([train_data, tr_data, eval_data], ignore_index=True)

else:
    full_data, train_data, eval_data, test_data = load_data_and_split(
        os.path.join(PATH, FOLDER_NAME, 'data', 'edos_labelled_aggregated.csv'))


print(f"Data Loaded: Train: {train_data.shape} Eval: {eval_data.shape} Test: {test_data.shape}")

print(train_data.label_vector.value_counts())


def tolist(tensor):
    return tensor.detach().cpu().tolist()


def map_names_2_ids(names):
  a = dict()
  b = dict()
  for id, name in enumerate(names):
    a[name] = id
    b[id] = name
  return a, b


train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

label_vector_raw = np.unique(train_data['label_vector']).tolist()
label_vector_map, vector_label_map = map_names_2_ids(label_vector_raw)
train_data['label_ids'] = [label_vector_map[i[1]] for i in train_data['label_vector'].items()]

test_data['label_ids'] = [label_vector_map[i[1]] for i in test_data['label_vector'].items()]
num_labels = len(train_data['label_vector'].unique())

train_dataframe = train_data
test_dataframe = test_data
print(f"Data after Preprocessing: train_dataframe: {train_dataframe.shape} "
      f" test_dataframe: {test_dataframe.shape}")

print(train_dataframe.columns)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

mistral_model = AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=model_name,
  num_labels=NUM_CLASSES,
  quantization_config=bnb_config,
  device_map={"": 0}
)

# lora config
mistral_model.config.pad_token_id = mistral_model.config.eos_token_id
mistral_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
)
mistral_model = prepare_model_for_kbit_training(mistral_model)

mistral_model = get_peft_model(mistral_model, mistral_peft_config)
mistral_model.print_trainable_parameters()


class ProcessDataset(Dataset):

  def __init__(self, dataframe, d_tokenizer, max_length=100, is_test=False):
    self.dataframe = dataframe
    self.d_tokenizer = d_tokenizer
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

    if not self.is_test:
      labels_C = torch.LongTensor([sample['label_ids']])
      tokenized_text['labels'] = labels_C
    return tokenized_text


def train(dataloader, model, device,
          loss_fn, optimizer, scheduler,
          adv_use_every_layer=False, use_adv=False):
    model.train()
    loss_collection = [[], [], [], [], []]

    for step, data in enumerate(dataloader):
        c_batch_size = data['input_ids'].shape[0]

        for key in data:
            data[key] = data[key].to(device).view(c_batch_size, -1)

        output = model(**data)

        ce_loss = loss_fn(output.logits, data['labels'].view(-1))
        ce_loss.backward()
        loss_collection[0].append(ce_loss.item())

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if len(loss_collection[0]) % log_step == 0:
            print(
                f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | CE Loss {round(sum(loss_collection[0]) / (len(loss_collection[0]) + 1e-8), 4)}')

            print('------------------------------------------------')
            loss_collection = [[] for _ in range(5)]


def eval(dataloader, model, device):
    model.eval()

    all_preds = list()

    for data in dataloader:
      c_batch_size = data['input_ids'].shape[0]
      for key in data:
        data[key] = data[key].to(device).view(c_batch_size, -1)
      labels = data.pop('labels').to(device).view(-1)
      with torch.no_grad():
          output = model(**data)

      preds = tolist(output.logits.argmax(1).view(-1))
      all_preds.extend(preds)
    return all_preds


def test(dataloader, model, device):
    model.eval()
    all_preds = list()

    for data in dataloader:
      c_batch_size = data['input_ids'].shape[0]
      for key in data:
        data[key] = data[key].to(device).view(c_batch_size, -1)
      with torch.no_grad():
        output = model(**data)
      preds = tolist(output.logits.argmax(1).view(-1))
      all_preds.extend(preds)
    return all_preds


MAX_LEN = 150
epochs = 10
lr = 1e-4
beta_1 = .9
beta_2 = .98
eps = 1e-6
log_step = 200
batch_size = 4
weight_decay = 5e-3


mistral_tokenizer = AutoTokenizer.from_pretrained(model_name)
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_model.config.pad_token_id = mistral_tokenizer.pad_token_id
mistral_model.config.pad_token_id = mistral_model.config.eos_token_id

train_dataset = ProcessDataset(train_dataframe, mistral_tokenizer, MAX_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ProcessDataset(test_dataframe, mistral_tokenizer, MAX_LEN, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sexist_model = mistral_model.to(device)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.array(list(range(num_labels))),
                                     y=train_data['label_ids'].values.tolist()).tolist()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device)).to(device)
loss_collection = []


opt_step = 0
optimization_steps = epochs * len(train_dataloader)
warmup_ratio = .0
warmup_steps = int(optimization_steps * warmup_ratio)


optimizer = AdamW(sexist_model.parameters(), lr=lr,
                  betas=(beta_1,beta_2), eps=eps, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=optimization_steps)

best_f1 = 0.
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
    if LOAD_CHECKPOINTS:
        print(f"Loading check points from {filename}")
        _, saved_dict = load_model()
        sexist_model.load_state_dict(saved_dict['model_state_dict'])
        preds_test = test(test_dataloader, sexist_model, device)
        f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_test, average='macro')
        print(f1_macro_test)
        if f1_macro_test > best_f1:
            best_f1 = f1_macro_test
            best_preds = preds_test
        print("ReStart Training...")
    else:
        print("Start Training...")

    for epoch in range(epochs):
        train(train_dataloader, sexist_model, device, loss_fn, optimizer, scheduler)

        # preds_eval = eval(eval_dataloader, sexist_model, device)
        # f1_macro_eval = f1_score(eval_dataframe['label_ids'].values.tolist(), preds_eval, average='macro')
        # all_f1.append(f1_macro_eval)
        #
        # print(f'EPOCH [{epoch + 1}/{epochs}] | Eval F1-Macro {round(f1_macro_eval * 100, 2)}')
        #
        # print(confusion_matrix(eval_dataframe['label_ids'].values.tolist(), preds_eval))

        preds_test = test(test_dataloader, sexist_model, device)
        f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_test, average='macro')

        if f1_macro_test > best_f1:
            best_f1 = f1_macro_test
            best_preds = preds_test
            save_model(epoch + 1, sexist_model, optimizer, scheduler)

        print(f'Current Test F1-Macro {round(f1_macro_test * 100, 2)}')
        print(f'EPOCH [{epoch + 1}/{epochs}] | Tesst Best F1-Macro {round(best_f1 * 100, 2)}')
        print(confusion_matrix(test_dataframe['label_ids'].values.tolist(), preds_test))

        # if early_stop(all_f1, best_f1, patience):
        #     break
        # else:
        #     print('not early stopping')

print(f"Loading check points from {filename}")
_, saved_dict = load_model()
sexist_model.load_state_dict(saved_dict['model_state_dict'])
preds_test = test(test_dataloader, sexist_model, device)
f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_test, average='macro')

print(f'Test F1-Macro {round(f1_macro_test * 100, 2)}')
print(confusion_matrix(test_dataframe['label_ids'].values.tolist(), preds_test))

test_preds_names = [vector_label_map[label_id] for label_id in preds_test]
test_dataframe['label_pred'] = test_preds_names
test_dataframe['pred_id'] = preds_test

csv_file_name = f"{m_name}_preds.csv"
test_dataframe.to_csv(os.path.join(PATH, FOLDER_NAME, 'csv', csv_file_name))

copy_and_paste = f"f1_macro_test: {f1_macro_test}, {m_name}\n"
with open(os.path.join(PATH, FOLDER_NAME,'info.txt'), 'a') as f:
    f.write(copy_and_paste)