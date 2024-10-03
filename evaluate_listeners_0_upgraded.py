import numpy as np
from torch.optim import AdamW
import torch.nn as nn
import torch
from vision import Conv4
from rsa_data import VOCAB, init_vocab, RSAShapeWorld, create_all_messages, generate_distributions, RSAShapeWorldGenerator
from torch.utils.data import DataLoader
from rsa_models import RSAListener
import random
import json
from itertools import product


#correlation parameter for P(S|S)
max_alphas = (1,10)
num_distractors_params = (2,3,4)
speaker_levels = (1,2)
listener_levels = (1,2,3)



if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    ce_loss = nn.CrossEntropyLoss()

    #create rsa_listener that we will upgrade
    vocabulary = init_vocab(VOCAB)
    messages = create_all_messages()
    P_S, P_S_S, P_C_S = generate_distributions(scale_factor=1000, max_alpha=1)
    shape_world = RSAShapeWorldGenerator(2, P_S, P_S_S, P_C_S, vocabulary, level=1,
                                         cost=0.6)
    all_messages_ids, lengths = shape_world.to_idx(messages)

    listener_embs = nn.Embedding(len(VOCAB) + 4, 50)
    listener_vision = Conv4()

    rsa_listener = RSAListener((torch.tensor(all_messages_ids), torch.tensor(lengths)), level=0)
    if torch.cuda.is_available():
         rsa_listener.cuda()

    random.seed(6)
    np.random.seed(6)

    results = []
    with open('results/rsa_results_0_upgraded.ljson', 'w') as outfile:
        for listener_level, max_alpha_l, num_distractors_l, speaker_level_l in product(listener_levels, max_alphas, num_distractors_params, speaker_levels):


            if torch.cuda.is_available():
                device = 'cuda:1'
            else:
                device =  'cpu'
            rsa_listener_trained = torch.load(f'models/epoch_10/RSA_listener_{0}_speaker_{speaker_level_l}_distractors_{num_distractors_l}_maxalpha_{max_alpha_l}_scost_06.pt',
                                              map_location=torch.device(device))
            if torch.cuda.is_available():
                rsa_listener_trained.cuda()

            rsa_listener.embedding = rsa_listener_trained.embedding
            rsa_listener.lang_model = rsa_listener_trained.lang_model
            rsa_listener.feat_model = rsa_listener_trained.feat_model
            rsa_listener.bilinear = rsa_listener_trained.bilinear
            rsa_listener.level = listener_level
            if torch.cuda.is_available():
                 rsa_listener.cuda()
            for max_alpha_s, num_distractors_s, speaker_level_s in product(max_alphas,
                                                                           num_distractors_params,
                                                                           speaker_levels):

                rsa_val_accuracy = []
                rsa_val_losses = []
                for dist_i in range(0, 10):
                    P_S, P_S_S, P_C_S = generate_distributions(scale_factor=1000, max_alpha=max_alpha_s)

                    val_sw = RSAShapeWorldGenerator(num_distractors_s, P_S, P_S_S, P_C_S, vocabulary, level=speaker_level_s, cost=0.6)
                    all_messages_ids, lengths = val_sw.to_idx(messages)
                    dataloader = DataLoader(val_sw, batch_size=32, shuffle=False)
                    for batch_i, (img, y, lang, length) in enumerate(dataloader):
                        if torch.cuda.is_available():
                            img, y, lang, length = img.cuda(), y.cuda(), lang.cuda(), length.cuda()
                        with torch.no_grad():
                            y = y.argmax(dim=1).long()
                            img = torch.transpose(img, 2, 4).double()
                            val_rsa_lis_scores = rsa_listener(img, lang, length)
                            val_rsa_loss = ce_loss(val_rsa_lis_scores, y)
                            rsa_val_accuracy += ((y == val_rsa_lis_scores.argmax(dim=1)) * 1).tolist()
                            rsa_val_losses.append(val_rsa_loss.item())
                        if batch_i > 100:
                            break
                avg_val_loss = sum(rsa_val_losses) / len(rsa_val_losses)
                rsa_acc = sum(rsa_val_accuracy) / len(rsa_val_accuracy)
                print(avg_val_loss)
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
                    'all_costs': rsa_val_losses
                }
                json.dump(results, outfile)
                outfile.write('\n')
                random.seed(6)
                np.random.seed(6)




