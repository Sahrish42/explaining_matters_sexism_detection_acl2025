import time

import numpy as np
import torch

import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DebertaTokenizer,AutoModelForSequenceClassification,DebertaForSequenceClassification, DebertaModel, RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
# from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

from torch.optim import AdamW, Adam, RMSprop

import random
from torch import Tensor
from typing import Optional, Tuple
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel
import os
from fnmatch import fnmatch
from pathlib import Path
HOME_PATH = str(Path.home())

PATH = os.path.join(HOME_PATH, 'edos_task10')

FOLDER_NAME = 'exp_with_definition_task_b'
m_name = 'pretrained7_deberta_task_b_original_with_definitions_variations_five_classes_eval_deberta'

TEST = False
LOAD_CHECKPOINTS = False
NUM_CLASSES = 4

seeds = [1000, 2000]
seed_idx = 1
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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


model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"load model from : {os.path.join(PATH, 'deberta_pretraining_2m','deberta-v3-pretrained', 'deberta_checkpoint-epoch-52')}")
model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(PATH, 'deberta_pretraining_2m','deberta-v3-pretrained', 'deberta_checkpoint-epoch-52'), num_labels=NUM_CLASSES)

# model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True, output_hidden_states=True)
print(f"Deberta Model Hidden Size: {model.config.hidden_size}")


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


MAX_LEN = 150
epochs = 30
lr = 6e-6
beta_1 = .9
beta_2 = .98
eps = 1e-6
log_step = 400
batch_size = 10
weight_decay = 5e-3


# Tokenize and pad sequences
def tokenize_and_pad(texts, labels):
    encodings = tokenizer.batch_encode_plus(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return encodings["input_ids"], encodings["attention_mask"], torch.LongTensor(labels)


train_encodings, train_masks, train_labels = tokenize_and_pad(train_dataframe.text.tolist(), train_dataframe.label_ids.tolist())
test_encodings, test_masks, test_labels = tokenize_and_pad(test_dataframe.text.tolist(), test_dataframe.label_ids.tolist())

# Create datasets and dataloaders
train_dataset = TensorDataset(train_encodings, train_masks, train_labels)
test_dataset = TensorDataset(test_encodings, test_masks, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sexist_model = model.to(device)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.array(list(range(num_labels_C))),
                                     y=train_data['label_ids'].values.tolist()).tolist()
print(class_weights)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device)).to(device)
loss_collection = []

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

best_f1 = 0.
best_model = None
transformers.logging.set_verbosity_error()


def train(dataloader, model, device,
          loss_fn, optimizer, scheduler):
    model.train()
    loss_collection = [[], [], [], [], []]

    for step, (batch_encodings, batch_masks, batch_labels) in enumerate(dataloader):
        batch_encodings = batch_encodings.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_encodings, attention_mask=batch_masks, labels=batch_labels)
        ce_loss = loss_fn(outputs.logits, batch_labels)

        ce_loss.backward()
        loss_collection[0].append(ce_loss.item())

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if len(loss_collection[0]) % log_step == 0:
            print(
                f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | '
                f'CE Loss {round(sum(loss_collection[0]) / (len(loss_collection[0]) + 1e-8), 4)}')
            print(f"Epoch {epoch + 1}/{epochs}, Current learning rate: {scheduler.get_last_lr()[0]}")
            print('------------------------------------------------')
            loss_collection = [[] for _ in range(5)]


def eval(dataloader, model, device):
    model.eval()

    all_preds = list()

    for batch_encodings, batch_masks, batch_labels in dataloader:
        batch_encodings = batch_encodings.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            outputs = model(batch_encodings, attention_mask=batch_masks)

        preds = tolist(outputs.logits.argmax(1).view(-1))
        all_preds.extend(preds)
    return all_preds


def test(dataloader, model, device):
    model.eval()
    all_preds = list()
    all_preds_val = list()
    all_preds_val_sm = list()

    for batch_encodings, batch_masks, batch_labels in dataloader:
        batch_encodings = batch_encodings.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            outputs = model(batch_encodings, attention_mask=batch_masks)

        logits = outputs.logits
        sm_logits = torch.nn.Softmax(dim=1)(logits)
        preds = logits.argmax(1).view(-1)
        all_preds.extend(tolist(preds))

        sm_list = []
        logits_list = []
        for i, row in enumerate(sm_logits):

            sm_val = row.gather(dim=0, index=preds[i])
            sm_list.append(sm_val.item())

            val = logits[i].gather(dim=0, index=preds[i])
            logits_list.append(val.item())

        all_preds_val_sm.extend(sm_list)
        all_preds_val.extend(logits_list)

    return all_preds, all_preds_val_sm, all_preds_val


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
        print("ReStart Training...")
    else:
        print("Start Training...")

    for epoch in range(epochs):
        train(train_dataloader, sexist_model, device, loss_fn, optimizer, scheduler)

        preds_test, all_preds_val_sm, all_preds_val = test(test_dataloader, sexist_model, device)
        f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(),
                                   preds_test, average='macro')
        if f1_macro_test > best_f1:
            best_f1 = f1_macro_test
            best_preds = preds_test
            save_model(epoch + 1, sexist_model, optimizer, scheduler)

        print(f'EPOCH [{epoch + 1}/{epochs}] | Current F1-Macro {round(f1_macro_test * 100, 2)}')
        print(f'EPOCH [{epoch + 1}/{epochs}] | Best F1-Macro {round(best_f1 * 100, 2)}')
        print(f'Test F1-Macro {round(f1_macro_test * 100, 2)}')
        print(confusion_matrix(test_dataframe['label_ids'].values.tolist(), preds_test))

        # if early_stop(all_f1, best_f1, patience):
        #     break
        # else:
        #     print('not early stopping')


print(f"Loading check points from {filename}")
_, saved_dict = load_model()
sexist_model.load_state_dict(saved_dict['model_state_dict'])
preds_test, preds_test_sm, preds_test_val = test(test_dataloader, sexist_model, device)
f1_macro_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_test, average='macro')

print(f'Test F1-Macro {round(f1_macro_test * 100, 2)}')

cm = confusion_matrix(test_dataframe['label_ids'].values.tolist(), preds_test,
                      labels=np.array(test_dataframe['label_ids'].unique().tolist()))
print(cm)


test_preds_names = [vector_label_map[label_id] for label_id in preds_test]
test_dataframe['label_pred'] = test_preds_names
test_dataframe['pred_id'] = preds_test
test_dataframe['softmax_max_value'] = preds_test_sm
test_dataframe['logits_max_value'] = preds_test_val
csv_file_name = f"{m_name}_preds.csv"
test_dataframe.to_csv(os.path.join(PATH, FOLDER_NAME, 'csv', csv_file_name))


copy_and_paste = f"f1_macro_test: {f1_macro_test}, model_name: {m_name} max_len: {MAX_LEN} LR: {lr} " \
                 f"batch_size: {batch_size} epochs: {epochs}\n"
with open(os.path.join(PATH, FOLDER_NAME,'info.txt'), 'a') as f:
    f.write(copy_and_paste)
