# %% [markdown]
# ### Student Information
# Name: 黃俊瑋 
# 
# Student ID: 108062308
# 
# GitHub ID: [tangerine1202](https://github.com/tangerine1202/DM2022-Lab2-Homework)
# 
# Kaggle name: [Chun Wei Huang](https://www.kaggle.com/tangerine1202)
# 
# Kaggle private scoreboard snapshot:
# 
# [Snapshot](img/pic0.png)

# %% [markdown]
# ---

# %% [markdown]
# ### Instructions

# %% [markdown]
# 1. First: __This part is worth 30% of your grade.__ Do the **take home** exercises in the [DM2022-Lab2-master Repo](https://github.com/keziatamus/DM2022-Lab2-Master). You may need to copy some cells from the Lab notebook to this notebook. 
# 
# 
# 2. Second: __This part is worth 30% of your grade.__ Participate in the in-class [Kaggle Competition](https://www.kaggle.com/competitions/dm2022-isa5810-lab2-homework) regarding Emotion Recognition on Twitter by [this link](https://www.kaggle.com/t/2b0d14a829f340bc88d2660dc602d4bd). The scoring will be given according to your place in the Private Leaderboard ranking: 
#     - **Bottom 40%**: Get 20% of the 30% available for this section.
# 
#     - **Top 41% - 100%**: Get (60-x)/6 + 20 points, where x is your ranking in the leaderboard (ie. If you rank 3rd your score will be (60-3)/6 + 20 = 29.5% out of 30%)   
# 
#     Submit your last submission __BEFORE the deadline (Nov. 22th 11:59 pm, Tuesday)_. Make sure to take a screenshot of your position at the end of the competition and store it as '''pic0.png''' under the img folder of this repository and rerun the cell Student Information.
# 
# 3. Third: __This part is worth 30% of your grade.__ A report of your work developping the model for the competition (You can use code and comment it). This report should include what your preprocessing steps, the feature engineering steps and an explanation of your model. You can also mention different things you tried and insights you gained. 
# 
# 
# 4. Fourth: __This part is worth 10% of your grade.__ It's hard for us to follow if your code is messy :'(, so please **tidy up your notebook** and **add minimal comments where needed**.
# 
# 
# Upload your files to your repository then submit the link to it on the corresponding e-learn assignment.
# 
# Make sure to commit and save your changes to your repository __BEFORE the deadline (Nov. 25th 11:59 pm, Friday)__.

# %% [markdown]
# ## First Part: Take Home Exercises
# 
# - repo link: https://github.com/tangerine1202/DM2022-Lab2-Master

# %% [markdown]
# # Second Part: Kaggle Competition

# %%
from datetime import datetime 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from tqdm.auto import tqdm

# %% [markdown]
# # Custom Transferred model

# %%
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import Dataset, load_dataset
import evaluate

from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from transformers import get_scheduler
from transformers import DataCollatorWithPadding
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import XLNetTokenizer, XLNetForSequenceClassification

# %%
# data
TRAIN_DATA_PATH = 'data/train_data.pkl'
TRAIN_LABEL_PATH = 'data/train_labels.pkl'
TEST_DATA_PATH = 'data/test_data.pkl'
CHECKPOINTS_PATH = 'checkpoints/'
TEXT_COL_NAME = 'replace_user_text'
EMOTION_NAMES = ['joy', 'anticipation', 'trust' , 'sadness' , 'disgust' , 'fear' , 'surprise', 'anger']
INPUT_COLUMNS = [TEXT_COL_NAME, 'label', *[f'{emo}_ratio' for emo in EMOTION_NAMES]]

# constants
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
NUM_LABELS = 8
CHECKPOINTS_SIZE = 500

# hyper-parameters
EPOCHS = 1
BATCH_SIZE = 4
LR = 1e-5
LR_WARMUP_RATIO = 0.005
TRAIN_SIZE = 145_0000 # 145_5563
EVAL_SIZE = 2225
TEST_SIZE = 3338

# %%
emotion2id = {emotion: i for i, emotion in enumerate(EMOTION_NAMES)}
id2emotion = {i: emotion for i, emotion in enumerate(EMOTION_NAMES)}

# %%
df_train_X = pd.read_pickle(TRAIN_DATA_PATH)
df_train_y = pd.read_pickle(TRAIN_LABEL_PATH)
assert len(df_train_X) == len(df_train_y)
df_train = pd.concat([df_train_X, df_train_y], axis=1)

# numerical labels
df_train['label'] = df_train['emotion'].map(lambda x: emotion2id[x])

df_train.sample(5)

# %%
def tokenize_function(examples):
    return tokenizer(examples[TEXT_COL_NAME], padding="max_length", truncation=True, max_length=512)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# dataset = Dataset.from_pandas(df_train[INPUT_COLUMNS])
# dataset

# %%
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# tokenized_datasets.save_to_disk(f'data/{MODEL_NAME}_tokenized_datasets')

# %%
tokenized_datasets = Dataset.load_from_disk(f'data/{MODEL_NAME}_tokenized_datasets')

# %%
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.remove_columns([TEXT_COL_NAME, 'tweet_id'])
tokenized_datasets.set_format("torch")
tokenized_datasets

# %%
small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(TRAIN_SIZE))
small_eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))
test_dataset = tokenized_datasets.shuffle(seed=42).select(range(TRAIN_SIZE + EVAL_SIZE, TRAIN_SIZE + EVAL_SIZE + TEST_SIZE))
# do not shuffle in dataloader, shuffle in dataset to keep the same seed
train_dataloader = DataLoader(small_train_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

print(f'training size:\t{len(small_train_dataset)}\t{len(small_train_dataset) / 1455563 * 100 :.3f}%')
print(f'eval     size:\t{len(small_eval_dataset)}\t{len(small_eval_dataset) / 1455563 * 100 :.3f}%')
print(f'test     size:\t{len(test_dataset)}\t{len(test_dataset) / 1455563 * 100 :.3f}%')

# %% [markdown]
# ## Prepare model

# %%
# ref: https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd
class CustomModel(torch.nn.Module):
  def __init__(self, checkpoint, num_labels): 
    super(CustomModel,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    self.seq_dropout = torch.nn.Dropout(0.3) 
    self.features_dropout = torch.nn.Dropout(0.0)
    self.classifier = torch.nn.Linear(768 + NUM_LABELS, num_labels) # load and initialize weights
    # self.classifier = torch.nn.Linear(768, num_labels) # load and initialize weights

  def forward(self, 
    input_ids=None, attention_mask=None, labels=None,
    joy_ratio=None, anticipation_ratio=None, trust_ratio=None, sadness_ratio=None, disgust_ratio=None, fear_ratio=None, surprise_ratio=None, anger_ratio=None
  ):
    # Extract outputs from the body
    # pretrained_outputs[0]=last hidden state
    pretrained_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    seq_outputs = self.seq_dropout(pretrained_outputs[0])[:, 0, :].view(-1, 768) # seq shape torch.Size([16, 768])

    # Add emotion ratios
    emo_ratio_outputs = torch.stack([joy_ratio, anticipation_ratio, trust_ratio, sadness_ratio, disgust_ratio, fear_ratio, surprise_ratio, anger_ratio], dim=1)
    features_outputs = self.features_dropout(emo_ratio_outputs) # emo_ratio shape torch.Size([16, 8])

    # Concatenate
    outputs = torch.cat((seq_outputs, features_outputs), dim=1)
    # outputs = seq_outputs
    logits = self.classifier(outputs) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = torch.nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=pretrained_outputs.hidden_states ,attentions=pretrained_outputs.attentions)
  
  def save_checkpoint(self, steps):
    torch.save(self.state_dict(), f'{CHECKPOINTS_PATH}{steps}.pt')
  

# %%
if os.path.isfile(f'model/{MODEL_NAME}-custom-{TRAIN_SIZE}') and input('Use pretrain model? (y/n)') == 'y':
  model = torch.load(f'model/{MODEL_NAME}-custom-{TRAIN_SIZE}')
  print('Using pretrained model')
else:
  model = CustomModel(MODEL_NAME, NUM_LABELS)
  print('Instantiate new model')
model.to(device)

# %%
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer, num_warmup_steps=int(num_training_steps*LR_WARMUP_RATIO), num_training_steps=num_training_steps
)

# %%
f1_metric = evaluate.load("f1")
acu_metric = evaluate.load("accuracy")

def compute_metrics():
    f1_macro = f1_metric.compute(average='macro')['f1']
    acu = acu_metric.compute()['accuracy']
    return {'f1_macro': f1_macro, 'acu': acu}


# %%
accelerator = Accelerator()

device = accelerator.device
model.to(device)

model, optimizer, train_dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler, eval_dataloader
)

# %% [markdown]
# ## Training

# %%
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(int(num_training_steps / CHECKPOINTS_SIZE) * len(eval_dataloader)))

best_f1 = 0
early_stop_cnt = 0
steps = 0
cumulative_loss = 0
for epoch in range(EPOCHS):
  model.train()
  for batch in train_dataloader:
      optimizer.zero_grad()
      # batch = {k: v.to(device) for k, v in batch.items()}
      batch = {k: v for k, v in batch.items()}  # accelerator
      outputs = model(**batch)
      loss = outputs.loss
      # loss.backward()
      accelerator.backward(loss)  # accelerator
      
      optimizer.step()
      lr_scheduler.step()
      progress_bar_train.update(1)

      cumulative_loss += loss.item()
      steps += 1

      if steps % CHECKPOINTS_SIZE == 0:
        print(f'step {steps} training loss: {cumulative_loss / CHECKPOINTS_SIZE}')
        cumulative_loss = 0
        model.save_checkpoint(steps)

        model.eval()
        for batch in eval_dataloader:
          # batch = {k: v.to(device) for k, v in batch.items()}
          batch = {k: v for k, v in batch.items()}  # accelerator
          with torch.no_grad():
              outputs = model(**batch)

          logits = outputs.logits
          predictions = torch.argmax(logits, dim=-1)
          f1_metric.add_batch(predictions=predictions, references=batch["labels"])
          acu_metric.add_batch(predictions=predictions, references=batch["labels"])
          progress_bar_eval.update(1)
          
        metrics = compute_metrics()
        print(metrics)

        if metrics['f1_macro'] < (best_f1 - 0.1):
          early_stop_cnt += 1
          print(f'early stop cnt = {early_stop_cnt}')
          if early_stop_cnt == 3:
            print('Early stop')
            break

        best_f1 = max(best_f1, metrics['f1_macro'])
        model.train()

# %%
# load checkpoints
# model.load_state_dict(torch.load(f'{CHECKPOINTS_PATH}{12500}.pt'))
# save
saved_path = f'model/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
torch.save(model, saved_path)
print(f'Saved model to {saved_path}')

# %% [markdown]
# ## Evaluation

# %%
f1_metric = evaluate.load("f1")
acu_metric = evaluate.load("accuracy")

progress_bar_test = tqdm(range(len(test_dataloader)))

model.eval()
i = 0
for batch in test_dataloader:
    i += 1
    if i > 3000:
        break
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    f1_metric.add_batch(predictions=predictions, references=batch["labels"])
    acu_metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar_test.update(1)

print(compute_metrics())

# %% [markdown]
# ## Inference

# %%
model = CustomModel(MODEL_NAME, NUM_LABELS).to(device)
# model.load_state_dict(torch.load(f'{CHECKPOINTS_PATH}{195000}.pt'))
model = torch.load(f'model/20221124-235147')

# %%
TEST_BATCH_SIZE = 40

TEST_INPUT_COLUMNS = INPUT_COLUMNS.copy()
if 'label' in TEST_INPUT_COLUMNS:
  TEST_INPUT_COLUMNS.remove('label')

# %%
df_test = pd.read_pickle(TEST_DATA_PATH)
df_test = df_test[TEST_INPUT_COLUMNS]
df_test.sample(5)

# %%
test_dataset = Dataset.from_pandas(df_test[TEST_INPUT_COLUMNS])
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.remove_columns([TEXT_COL_NAME, 'tweet_id'])
test_dataset.set_format(type='torch')
test_dataset

# %%
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=data_collator)

# %%
df_result = pd.DataFrame({'id': df_test.index})
df_result['label'] = np.zeros(len(df_result)) - 1
gpu_tensor = torch.tensor(df_result['label'].values, dtype=torch.long, device=device)

progress_bar_test = tqdm(range(len(test_dataloader)))

idx = 0
model.eval()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    gpu_tensor[idx:idx+len(predictions)] = predictions
    idx += len(predictions)
    progress_bar_test.update(1)

df_result['label'] = gpu_tensor.cpu().numpy()

# convert label to emotion
df_result['emotion'] = df_result['label'].map(lambda x: id2emotion[x])

df_result.head(3)

# %%
saved_path = f'submission/{MODEL_NAME}-custom-{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
df_result[['id', 'emotion']].to_csv(saved_path, index=False)
print(saved_path)

# %%



