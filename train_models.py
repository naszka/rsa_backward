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
from itertools import product
import wandb
import random
import pdb
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


batch_size = 32
epochs = 12
max_alphas = (5,1)
num_distractors_params = (2,3,4,)
speaker_levels = (1,2)

if __name__ == "__main__":
    parser = ArgumentParser(description='Train', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--listener_level', default=1, type=int, help='Level of the trained listener')
    parser.add_argument('--word_cost', default=0, type=float, help='Cost of each word')
    args = parser.parse_args()

    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(0)

    torch.set_default_dtype(torch.float64)
    ce_loss = nn.CrossEntropyLoss()



    vocabulary = init_vocab(VOCAB)
    messages = create_all_messages()


    for max_alpha, num_distractors, speaker_level in product(max_alphas, num_distractors_params, speaker_levels):
        name_str = f'RSA_listener_{args.listener_level}_speaker_{speaker_level}_distractors_{num_distractors}_maxalpha_{max_alpha}_sf_1k_scost_{args.word_cost}_triangle'
        wandb.init(project='RSA_listener',
                   name=name_str)
        torch.manual_seed(3)
        random.seed(3)
        np.random.seed(3)

        P_S, P_C, P_S_S, P_C_C = generate_distributions(scale_factor=1000, max_alpha=max_alpha)
        shape_world = RSAShapeWorldGenerator(num_distractors, P_S, P_C, P_S_S, P_C_C, vocabulary, level=speaker_level, cost=args.word_cost)
        all_messages_ids, lengths = shape_world.to_idx(messages)
        dataloader = DataLoader(shape_world, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(shape_world, batch_size=batch_size, shuffle=True)

        listener_embs = nn.Embedding(len(VOCAB) + 4, 50)
        listener_vision = Conv4()

        if args.listener_level == 0:
            listener = Listener(listener_vision, listener_embs)
        else:
            listener = RSAListener((torch.tensor(all_messages_ids), torch.tensor(lengths)), level=args.listener_level, cost = args.word_cost)



        if torch.cuda.is_available():
            listener.cuda()
        rsa_optimizer = AdamW(listener.parameters(), lr=1e-5)
        min_avg_loss = 1e9

        #eval untrained model
        rsa_val_accuracy = []
        rsa_val_losses = []
        for val_i, (val_img, val_y, val_lang, val_length) in enumerate(val_dataloader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    val_y = val_y.argmax(dim=1).cuda().long()
                else:
                    val_y = val_y.argmax(dim=1).long()

                val_img = torch.transpose(val_img, 2, 4).double()
                val_rsa_lis_scores = listener(val_img.cuda(), val_lang.cuda(), val_length.cuda())
                val_rsa_loss = ce_loss(val_rsa_lis_scores, val_y)
                rsa_val_accuracy += ((val_y == val_rsa_lis_scores.argmax(dim=1)) * 1).tolist()
                rsa_val_losses.append(val_rsa_loss.item())
            if val_i > 100:
                break
        avg_val_loss = sum(rsa_val_losses) / len(rsa_val_losses)
        wandb.log({"val_loss": avg_val_loss})

        rsa_acc = sum(rsa_val_accuracy) / len(rsa_val_accuracy)
        wandb.log({"listener_accuracy": rsa_acc})
        for epoch_i in range(0,epochs):
            for batch_i, (img, y, lang, length) in enumerate(dataloader):
                if torch.cuda.is_available():
                    y = y.argmax(dim=1).cuda().long()
                else:
                    y = y.argmax(dim=1).long()
                img = torch.transpose(img, 2, 4).double()
                if torch.cuda.is_available():
                    img, lang, length = img.cuda(), lang.cuda(), length.cuda()
                rsa_lis_scores = listener(img, lang, length)
                rsa_loss = ce_loss(rsa_lis_scores, y)
                if batch_i % 50 == 49:
                    rsa_val_accuracy = []
                    rsa_val_losses = []
                    for val_i, (val_img, val_y, val_lang, val_length) in enumerate(val_dataloader):
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                val_y = val_y.argmax(dim=1).cuda().long()
                            else:
                                val_y = val_y.argmax(dim=1).long()

                            val_img = torch.transpose(val_img, 2, 4).double()
                            val_rsa_lis_scores = listener(val_img.cuda(), val_lang.cuda(), val_length.cuda())
                            val_rsa_loss = ce_loss(val_rsa_lis_scores, val_y)
                            rsa_val_accuracy += ((val_y == val_rsa_lis_scores.argmax(dim=1)) * 1).tolist()
                            rsa_val_losses.append(val_rsa_loss.item())
                        if val_i > 100:
                            break
                    avg_val_loss = sum(rsa_val_losses) / len(rsa_val_losses)
                    if avg_val_loss < min_avg_loss:
                        min_avg_loss = avg_val_loss
                        torch.save(listener, f'rsa_backward/models/'+ name_str + '.pt')
                    wandb.log({"val_loss": avg_val_loss})

                    rsa_acc = sum(rsa_val_accuracy) / len(rsa_val_accuracy)
                    wandb.log({"listener_accuracy": rsa_acc})

                rsa_loss.backward()
                rsa_optimizer.step()
                rsa_optimizer.zero_grad()
            if epoch_i == 1:
                torch.save(listener,
                           f'rsa_backward/models/epoch_1/' + name_str + '.pt')
            if epoch_i == 5:
                torch.save(listener,
                           f'rsa_backward/models/epoch_5/' + name_str + '.pt')
            if epoch_i == 10:
                torch.save(listener,
                           f'rsa_backward/models/epoch_10/' + name_str + '.pt')

        wandb.finish()



