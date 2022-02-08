from sroie.py import dataloader_train_sroie, dataloader_test_sroie
from funsd.py import train_dataloader, eval_dataloader

def dataloader_train(Dataset):
  if Dataset=='SROIE':
    return(dataloader_train_sroie)
  elif Dataset=='FUNSD':
    return(train_dataloader)
  
def dataloader_test(Dataset):
  if Dataset=='SROIE':
    return(dataloader_test_sroie)
  elif Dataset=='FUNSD':
    return(eval_dataloader)
  

