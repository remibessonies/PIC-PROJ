from transformers import AdamW
from models.LayoutLM import model
import torch
import os
import pandas as pd
from transformers import LayoutLMTokenizer
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

dataset_path ="SROIE/data_loading/SROIE2019/train"

images = []
labels = []

for label_folder, _, file_names in os.walk(dataset_path):
  if label_folder != dataset_path:
    label = label_folder[40:]
    for _, _, image_names in os.walk(label_folder):
      relative_image_names = []
      for image in image_names:
        relative_image_names.append(dataset_path + "/" + label + "/" + image)
      images.extend(relative_image_names)
      labels.extend([label] * len (relative_image_names)) 

data = pd.DataFrame.from_dict({'image_path': images, 'label': labels})




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 30
t_total = len(data) * num_train_epochs # total number of training steps 

#put the model in training mode
model.train()
for epoch in range(num_train_epochs):
  print("Epoch:", epoch)
  running_loss = 0.0
  correct = 0
  for batch in data:

      #https://github.com/microsoft/unilm/blob/master/layoutlm/deprecated/examples/classification/run_classification.py
      #https://huggingface.co/transformers/v3.5.1/glossary.html#input-ids
      encoding = tokenizer(" ".join(batch), return_tensors="tf")
      # convert the sequence to a list of tokens then associate keys to the tokens
      input_ids = encoding["input_ids"]

      bbox = encoding["bbox"]

      # indicates to the model which tokens should be attended to and which should not
      attention_mask = encoding["attention_mask"]
      token_type_ids = encoding["token_type_ids"]
      labels = encoding["label"]
        
      # forward pass
      outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      labels=labels)
      loss = outputs.loss

      running_loss += loss.item()
      predictions = outputs.logits.argmax(-1)
      correct += (predictions == labels).float().sum()

      # backward pass to get the gradients 
      loss.backward()

      # update
      optimizer.step()
      optimizer.zero_grad()
      global_step += 1
  
  print("Loss:", running_loss / batch["input_ids"].shape[0])
  accuracy = 100 * correct / len(data)
  print("Training accuracy:", accuracy.item())