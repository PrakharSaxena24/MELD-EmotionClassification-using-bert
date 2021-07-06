import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from evaluate import f1_score_func,accuracy_per_class

seed_val = 994
device=torch.device("cuda")
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
batch_size=16
epochs=5

train_data=pd.read_csv("./MELD/data/MELD/train_sent_emo.csv")
dev_data=pd.read_csv("./MELD/data/MELD/dev_sent_emo.csv")
test_data=pd.read_csv("./MELD/data/MELD/test_sent_emo.csv")

utterances_train=train_data["Utterance"]
utterances_dev=dev_data["Utterance"]
utterances_test=test_data["Utterance"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

processed_data_train=[]
processed_data_dev=[]
processed_data_test=[]
# labelsT=train_data["Emotion"]
# labelsD=dev_data["Emotion"]
# labels_train=[]
# labels_dev=[]
possible_label=train_data.Emotion.unique()
label_dict={}
for index,possible_label in enumerate(possible_label):
    label_dict[possible_label]=index

train_data['Emotion'] = train_data.Emotion.replace(label_dict)
dev_data["Emotion"]= dev_data.Emotion.replace(label_dict)
test_data["Emotion"]=test_data.Emotion.replace(label_dict)
# print(train_data.Emotion.values)
# processed_data_train1=[]
# processing data
# for i in utterances_train:
#     # print(i)
#     processed_data_train.append(i)
processed_data_train=utterances_train.values
processed_data_dev=utterances_dev.values
processed_data_test=utterances_test.values

# print(type(processed_data_train))
# print(type(processed_data_train1))
# for i in utterances_dev:
#     processed_data_dev.append(i)
# print(processed_data[1])



# # tokenizing data
encoded_inputs_train=tokenizer.batch_encode_plus(processed_data_train,add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=50,
    truncation=True,
    return_tensors='pt')

encoded_inputs_dev=tokenizer.batch_encode_plus(processed_data_dev,add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=50,
    truncation=True,
    return_tensors='pt')

encoded_inputs_test=tokenizer.batch_encode_plus(processed_data_test,add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=50,
    truncation=True,
    return_tensors='pt')


input_id_train=encoded_inputs_train["input_ids"]
attention_mask_train=encoded_inputs_train["attention_mask"]
labels_train=torch.tensor(train_data.Emotion.values)
# print(input_id_train)
#
input_id_dev=encoded_inputs_dev["input_ids"]
attention_mask_dev=encoded_inputs_dev["attention_mask"]
labels_dev=torch.tensor(dev_data.Emotion.values)


input_id_test=encoded_inputs_test["input_ids"]
attention_mask_test=encoded_inputs_test["attention_mask"]
labels_test=torch.tensor(test_data.Emotion.values)


#
dataset_train=TensorDataset(input_id_train,attention_mask_train,labels_train)
dataset_dev=TensorDataset(input_id_dev,attention_mask_dev,labels_dev)
dataset_test=TensorDataset(input_id_test,attention_mask_test,labels_test)

model=BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=7,
                                                    output_attentions=False,
                                                    output_hidden_states=False)

# dataset_train=dataset_train.to("cuda")
# dataset_dev=dataset_dev.to("cuda")
model=model.to("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}
dataloader_train=DataLoader(dataset_train,
                            sampler=RandomSampler(dataset_train),
                            batch_size=batch_size,
                            **kwargs)
dataloader_dev=DataLoader(dataset_dev,
                          sampler=SequentialSampler(dataset_dev),
                          batch_size=batch_size,
                          **kwargs)
dataloader_test=DataLoader(dataset_test,
                          sampler=SequentialSampler(dataset_test),
                          batch_size=len(test_data),
                          **kwargs)


optimizer=AdamW(model.parameters(),
                lr=5e-05,
                eps=1e-08)

scheduler=get_linear_schedule_with_warmup(optimizer,
                                          num_warmup_steps=(0.1*(len(dataloader_train)*epochs)),
                                          num_training_steps=len(dataloader_train)*epochs)


def evaluate(dataloader_val):
    model.eval()
    loss_val_total=0
    predictions,true_vals=[],[]

    for batch in dataloader_val:
        batch=tuple(b.to(device) for b in batch)

        inputs={"input_ids":batch[0],
                "attention_mask": batch[1],
                "labels":batch[2]}


        with torch.no_grad():
            outputs=model(**inputs)

        loss=outputs[0]
        logits=outputs[1]
        loss_val_total+=loss.item()

        logits=logits.detach().cpu().numpy()
        label_ids=inputs["labels"].cpu().numpy()

        predictions.append(logits)
        true_vals.append(label_ids)
        loss_val_avg=loss_val_total/len(dataloader_val)
        predictions=np.concatenate(predictions,axis=0)
        true_vals=np.concatenate(true_vals,axis=0)

        return loss_val_avg,predictions,true_vals


# Train
# for epoch in tqdm(range(1,epochs+1)):
#     model.train()
#     loss_train_total=0
#
#     progress_bar=tqdm(dataloader_train,desc="Epoch{:1d}".format(epoch),leave=False,disable=False)
#     for batch in progress_bar:
#         model.zero_grad()
#
#         batch=tuple(b.to(device) for b in batch)
#
#         inputs={"input_ids":batch[0],
#                 "attention_mask":batch[1],
#                 "labels":batch[2]}
#         outputs=model(**inputs)
#         loss=outputs[0]
#
#         loss_train_total+=loss.item()
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
#         optimizer.step()
#         scheduler.step()
#         progress_bar.set_postfix({"training loss":"{:0.2f}".format(loss.item()/len(batch))})
#
#     torch.save(model.state_dict(),f"./finetuned_epoch{epoch}.model")
#     tqdm.write(f"Epoch {epoch}")
#     loss_train_avg=loss_train_total/len(dataloader_train)
#     tqdm.write(f"Training loss: {loss_train_avg}")
#
#     val_loss,predictions,true_vals=evaluate(dataloader_dev)
#     val_f1 = f1_score_func(predictions, true_vals)
#     tqdm.write(f'Validation loss: {val_loss}')
#     tqdm.write(f'F1 Score (Weighted): {val_f1}')
#

# test
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('./finetuned_epoch5.model', map_location=torch.device('cuda')))

_, predictions, true_vals = evaluate(dataloader_test)
accuracy_per_class(predictions, true_vals)
# print(len(encoded_inputs_train["attention_mask"]))


