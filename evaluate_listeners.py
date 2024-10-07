import numpy as np
import torch.nn as nn
import torch
from rsa_data import VOCAB, init_vocab, create_all_messages, generate_distributions, RSAShapeWorldGenerator
from torch.utils.data import DataLoader
import random
import json
from itertools import product
from datetime import datetime

max_alphas = (5,1)
num_distractors_params = (4,3,2)
speaker_levels = (1,2)
listener_levels = (1,2,4)
epochs = (10,)
costs = (0.0, 0.6)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    ce_loss = nn.CrossEntropyLoss()



    random.seed(6)
    np.random.seed(6)

    results = []
    with open(f'rsa_backward/results/{datetime.now().date()}rsa_results.ljson', 'w') as outfile:
        for listener_level, max_alpha_l, num_distractors_l, speaker_level_l, epoch, cost_l in product(listener_levels, max_alphas, num_distractors_params, speaker_levels, epochs, costs):
            name_str = f'RSA_listener_{listener_level}_speaker_{speaker_level_l}_distractors_{num_distractors_l}_maxalpha_{max_alpha_l}_sf_1k_scost_{cost_l}_triangle'
            rsa_listener = torch.load(f'rsa_backward/models/epoch_{epoch}/' + name_str + '.pt')
            rsa_listener.cost = cost_l
            rsa_listener.cuda()

            vocabulary = init_vocab(VOCAB)
            messages = create_all_messages()

            for max_alpha_s, num_distractors_s, speaker_level_s, cost_s in product(max_alphas,
                                                                           num_distractors_params,
                                                                           speaker_levels, costs):

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
                    'all_accuracies': rsa_val_accuracy,
                    'all_costs': rsa_val_losses,
                    'epoch': epoch,
                    'l_cost': cost_l ,
                    "s_cost": cost_s ,
                }
                json.dump(results, outfile)
                outfile.write('\n')
                random.seed(6)
                np.random.seed(6)




