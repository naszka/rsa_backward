import numpy as np
from torch.optim import AdamW
import torch.nn as nn
import torch
from vision import Conv4
from models import Listener
from rsa_data import VOCAB, init_vocab, RSAShapeWorld, create_all_messages, generate_distributions, RSAShapeWorldGenerator
from data import ShapeWorld
from torch.utils.data import DataLoader
from rsa_backward.rsa_models import RSAListener
import wandb
import random
import pdb
import json
from itertools import product
from datetime import datetime


max_alphas = (5,1)
num_distractors = (2,3,4)
speaker_levels = (1,2)
listener_levels = (0,1,2)
epochs = (10,)
scost = (0.6, 0.0)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    ce_loss = nn.CrossEntropyLoss()




    #create rsa_listner that we will upgrade
    vocabulary = init_vocab(VOCAB)
    messages = create_all_messages()
    P_S, P_C, P_S_S, P_C_C = generate_distributions(scale_factor=1000, max_alpha=1)
    shape_world = RSAShapeWorldGenerator(2, P_S, P_C, P_S_S, P_C_C, vocabulary, level=1,
                                         cost=0.6)
    all_messages_ids, lengths = shape_world.to_idx(messages)

    listener_embs = nn.Embedding(len(VOCAB) + 4, 50)
    listener_vision = Conv4()

    rsa_listener = RSAListener((torch.tensor(all_messages_ids), torch.tensor(lengths)),
                               level=0, cost=0.6)

    pdb.set_trace()
    rsa_listener.cuda()

    random.seed(6)
    np.random.seed(6)

    results = []
    with open(f'/ivi/ilps/personal/knaszad/rsa_backward/results/{datetime.now().date()}_0_upgraded_rsa_results_epoch_10.ljson', 'w') as outfile:
        for listener_level, max_alpha_l, num_distractors_l, speaker_level_l, epoch, cost_l in product(listener_levels, max_alphas, num_distractors, speaker_levels, epochs, scost):
            name_str = f'RSA_listener_{0}_speaker_{speaker_level_l}_distractors_{num_distractors_l}_maxalpha_{max_alpha_l}_sf_1k_scost_{cost_l}_triangle'
            rsa_listener_trained = torch.load(f'/ivi/ilps/personal/knaszad/rsa_backward/models/epoch_{epoch}/' + name_str +'.pt')
            rsa_listener_trained.cuda()

            rsa_listener.embedding = rsa_listener_trained.embedding
            rsa_listener.lang_model = rsa_listener_trained.lang_model
            rsa_listener.feat_model = rsa_listener_trained.feat_model
            rsa_listener.bilinear = rsa_listener_trained.bilinear
            rsa_listener.level = listener_level
            rsa_listener.cuda()





            for max_alpha_s, num_distractors_s, speaker_level_s, cost_s in product(max_alphas, num_distractors, speaker_levels, scost):

                rsa_val_accuracy = []
                rsa_val_losses = []
                for dist_i in range(0, 10):
                    P_S, P_C, P_S_S, P_C_C = generate_distributions(scale_factor=1000, max_alpha=max_alpha_s)

                    val_sw = RSAShapeWorldGenerator(num_distractors_s, P_S, P_C, P_S_S, P_C_C, vocabulary, level=speaker_level_s, cost=cost_s)
                    all_messages_ids, lengths = val_sw.to_idx(messages)
                    dataloader = DataLoader(val_sw, batch_size=32, shuffle=False)
                    for batch_i, (img, y, lang, length) in enumerate(dataloader):
                        with torch.no_grad():
                            y = y.argmax(dim=1).cuda().long()
                            img = torch.transpose(img, 2, 4).double()
                            val_rsa_lis_scores = rsa_listener(img.cuda(), lang.cuda(), length.cuda())
                            val_rsa_loss = ce_loss(val_rsa_lis_scores, y)
                            rsa_val_accuracy += ((y == val_rsa_lis_scores.argmax(dim=1)) * 1).tolist()
                            rsa_val_losses.append(val_rsa_loss.item())
                        if batch_i > 100:
                            break
                avg_val_loss = sum(rsa_val_losses) / len(rsa_val_losses)
                rsa_acc = sum(rsa_val_accuracy) / len(rsa_val_accuracy)

                results = {
                    'listener_level': listener_level,
                    'max_alpha_training': max_alpha_l,
                    'num_distractors_training': num_distractors_l,
                    'speaker_level_training': speaker_level_l,
                    'max_alpha_eval': max_alpha_s,
                    'num_distractors_eval': num_distractors_s,
                    'speaker_level_eval': speaker_level_s,
                    'eval_loss': avg_val_loss,
                    'eval_acc': rsa_acc,
                    'listener_level_training': 0,
                    'all_accuracies': rsa_val_accuracy,
                    'all_costs': rsa_val_losses,
                    'epoch': epoch,
                    'l_cost': cost_l,
                    's_cost': cost_s

                }
                json.dump(results, outfile)
                outfile.write('\n')
                random.seed(6)
                np.random.seed(6)




