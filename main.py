
from transformers import LayoutLMForTokenClassification
import torch
from transformers import AdamW
from tqdm import tqdm
<<<<<<< HEAD
# from data_loading.utils import dataloader_train, dataloader_test
from data_loading.funsd import train_dataloader,eval_dataloader
=======
from data_loading.utils import dataloader_train, dataloader_test
#from data_loading.funsd import train_dataloader,eval_dataloader
>>>>>>> 13d3e2d670999d9a25169db2bbc883a356b9515f
#from data_loading.sroie import dataloader_test_sroie, dataloader_train_sroie
import os
import numpy as np
import random
from training.train import train
from training.evaluate import evaluate

import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./data_loading/FUNSD',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='funsd',
                        choices=['funsd', 'sroie'], help='Name of dataset')

    # Models Options
    parser.add_argument("--model", type=str, default='LayoutLM',
                        choices=['LayoutLM', 'LayoutLM'], help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save results to \"./results\"")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="epoch number (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate (default: 0.00005)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    return parser

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # define number of labels refer to different dataset
    if opts.dataset.lower() == 'funsd':
        def get_labels(path):
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        labels = get_labels(os.path.join(opts.data_root,"labels.txt"))
        num_labels = len(labels)
    else:
        pass

    # define model
    if opts.model == 'LayoutLM':
        model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
    else:
        pass
    model.to(device)

    # define optimization
    optimizer = AdamW(model.parameters(), lr=opts.lr)

    if opts.test_only:
        model.load_state_dict(torch.load('./checkpoint_LayoutLMF_funsd.pth'))
        print('load checkpoint')
        # print(len(dataloader_test))
        # exit()
        evaluate(model=model, device=device, eval_dataloader=dataloader_test(opts.dataset),labels=labels)
        return
    else:
        globel_step, loss = train(model=model, device=device, train_dataloader=dataloader_train(opts.dataset), eval_dataloader=dataloader_test(opts.dataset),optimizer=optimizer, labels=labels, num_train_epochs=opts.num_train_epochs)
        print('the globel step is {} and the loss is {}'.format(globel_step,loss))
        return

if __name__ == '__main__':
    main()








