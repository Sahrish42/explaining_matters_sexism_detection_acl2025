import numpy as np
import torch
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DebertaTokenizer,AutoModelForSequenceClassification,DebertaForSequenceClassification, DebertaModel, RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from torch.optim import AdamW

import random

from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel
import os
from fnmatch import fnmatch

from pathlib import Path
HOME_PATH = str(Path.home())

PATH = os.path.join(HOME_PATH, 'edos_task10')
VARIATIONS_AND_ORIGINAL = True
TEST = True
CHECK_POINTS = False
NUM_CLASSES = 11
m_name = 'pretrained9_task_c_original_five_classes_variations_eval_deberta'
FOLDER_NAME = 'edos_task_c'

seeds = [1000, 2000]
seed_idx = 1
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"load model from : {os.path.join(PATH, 'deberta_pretraining_2m','deberta-v3-pretrained', 'deberta_checkpoint-epoch-54')}")
model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(PATH, 'deberta_pretraining_2m','deberta-v3-pretrained', 'deberta_checkpoint-epoch-54'), num_labels=NUM_CLASSES)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)

# model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True, output_hidden_states=True)
print(f"Deberta Model Hidden Size: {model.config.hidden_size}")


# removing the new line characters
def load_old_aug_data(root):
    aug_dataset = pd.read_csv(root + "augmented_train.csv")
    # aug_dataset = aug_dataset[aug_dataset.label_sexist == 'sexist']
    a = aug_dataset[['text', 'label_sexist']]
    b = aug_dataset[['text_eda', 'label_sexist']]
    c = aug_dataset[['text_nlpaug', 'label_sexist']]

    b = b.rename(columns={"text_eda": "text"})
    c = c.rename(columns={"text_nlpaug": "text"})
    new_data = pd.concat([a, b, c], ignore_index=True)
    return new_data


def read_casual_data(root, col=''):
    not_sex_data = pd.read_csv(os.path.join(root, 'casual_analysis_not_sexist.csv'))
    sex_data = pd.read_csv(os.path.join(root, 'casual_analysis_sexist.csv'))
    new_data = pd.concat([not_sex_data, sex_data], ignore_index=True)
    if col != '':
        new_data = new_data[[col, 'label_sexist']]
        new_data = new_data.rename(columns={col: 'text'})

    return new_data


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
  A = dict()
  B = dict()
  for id, name in enumerate(names):
    A[name] = id
    B[id] = name
  return A, B


train_data = train_data.reset_index(drop=True)
eval_data = eval_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

label_vector_raw = np.unique(train_data['label_vector']).tolist()
label_vector_map, vector_label_map = map_names_2_ids(label_vector_raw)
train_data['label_ids'] = [label_vector_map[i[1]] for i in train_data['label_vector'].items()]

eval_data['label_ids'] = [label_vector_map[i[1]] for i in eval_data['label_vector'].items()]
test_data['label_ids'] = [label_vector_map[i[1]] for i in test_data['label_vector'].items()]
num_labels = len(train_data['label_vector'].unique())

train_dataframe = train_data
eval_dataframe = eval_data
test_dataframe = test_data
print(f"Data after Preprocessing: train_dataframe: {train_dataframe.shape} "
      f"eval_dataframe: {eval_dataframe.shape}"
      f" test_dataframe: {test_dataframe.shape}")

print(train_dataframe.columns)


MAX_LEN = 150
epochs = 30
lr = 6e-6
beta_1 = .9
beta_2 = .98
eps = 1e-6
log_step = 200
batch_size = 10
weight_decay = 5e-3


# Tokenize and pad sequences
def tokenize_and_pad(texts, labels):
    encodings = tokenizer.batch_encode_plus(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return encodings["input_ids"], encodings["attention_mask"], torch.LongTensor(labels)


train_encodings, train_masks, train_labels = tokenize_and_pad(train_dataframe.text.tolist(), train_dataframe.label_ids.tolist())
val_encodings, val_masks, val_labels = tokenize_and_pad(eval_dataframe.text.tolist(), eval_dataframe.label_ids.tolist())
test_encodings, test_masks, test_labels = tokenize_and_pad(test_dataframe.text.tolist(), test_dataframe.label_ids.tolist())

# print(train_labels.)
# Create datasets and dataloaders
train_dataset = TensorDataset(train_encodings, train_masks, train_labels)
val_dataset = TensorDataset(val_encodings, val_masks, val_labels)
test_dataset = TensorDataset(test_encodings, test_masks, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sexist_model = model.to(device)
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.array(list(range(num_labels))), y=train_data['label_ids'].values.tolist()).tolist()
print(class_weights)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device)).to(device)
loss_collection = []

opt_step = 0
optimization_steps = epochs * len(train_dataloader)
warmup_ratio = .0
warmup_steps = int(optimization_steps * warmup_ratio)


optimizer = AdamW(sexist_model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps, weight_decay=weight_decay)
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
                f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] |'
                f' CE Loss {round(sum(loss_collection[0]) / (len(loss_collection[0]) + 1e-8), 4)}')
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

    for batch_encodings, batch_masks, batch_labels in dataloader:
        batch_encodings = batch_encodings.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            outputs = model(batch_encodings, attention_mask=batch_masks)

        preds = tolist(outputs.logits.argmax(1).view(-1))
        all_preds.extend(preds)
    return all_preds


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

        # preds_C_eval = eval(val_dataloader, sexist_model, device)
        # # print(len(preds_C_eval))
        # f1_macro_C_eval = f1_score(eval_dataframe['label_ids'].values.tolist(), preds_C_eval, average='macro')
        # all_f1.append(f1_macro_C_eval)
        #
        # print(f'EPOCH [{epoch + 1}/{epochs}] | Eval F1-Macro {round(f1_macro_C_eval * 100, 2)}')
        #
        # print(confusion_matrix(eval_dataframe['label_ids'].values.tolist(), preds_C_eval))

        preds_C_test = test(test_dataloader, sexist_model, device)
        f1_macro_C_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_C_test, average='macro')

        if f1_macro_C_test > best_f1:
            best_f1 = f1_macro_C_test
            best_preds = preds_C_test
            save_model(epoch + 1, sexist_model, optimizer, scheduler)

        print(f'EPOCH [{epoch + 1}/{epochs}] | Test F1-Macro {round(f1_macro_C_test * 100, 2)}')
        print(f'EPOCH [{epoch + 1}/{epochs}] | Test Best F1-Macro {round(best_f1 * 100, 2)}')
        print(confusion_matrix(test_dataframe['label_ids'].values.tolist(), preds_C_test))

        if early_stop(all_f1, best_f1, patience):
            break
        else:
            print('not early stopping')


print(f"Loading check points from {filename}")
_, saved_dict = load_model()
sexist_model.load_state_dict(saved_dict['model_state_dict'])
preds_C_test = test(test_dataloader, sexist_model, device)
f1_macro_C_test = f1_score(test_dataframe['label_ids'].values.tolist(), preds_C_test, average='macro')

print(f'Test F1-Macro {round(f1_macro_C_test * 100, 2)}')

test_preds_names = [vector_label_map[label_id] for label_id in preds_C_test]
test_dataframe['label_pred'] = test_preds_names
test_dataframe['pred_id'] = preds_C_test

csv_file_name = f"{m_name}_preds.csv"
test_dataframe.to_csv(os.path.join(PATH, FOLDER_NAME, 'csv', csv_file_name))

copy_and_paste = f"f1_macro_test: {f1_macro_C_test}, {m_name}\n"
with open(os.path.join(PATH, FOLDER_NAME,'info.txt'), 'a') as f:
    f.write(copy_and_paste)
