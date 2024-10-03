from models import Listener
from vision import Conv4
from torch import nn
import torch
import numpy as np
from rsa_data import VOCAB, init_vocab, RSAShapeWorld, create_all_messages
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pdb

class RSAListener(Listener):
    def __init__(self, messages, cost=0.6, level = 1):
        listener_embs = nn.Embedding(len(VOCAB) + 4, 50)
        listener_vision = Conv4()
        super(RSAListener, self).__init__(listener_vision, listener_embs)
        messages, message_lengths = messages
        if torch.cuda.is_available():
            self.message_lengths = message_lengths.cuda()
            self.messages = messages.cuda()
        else:
            self.message_lengths = message_lengths
            self.messages = messages

        self.cost = cost
        self.level = level
        self.speaker_norm = nn.LogSoftmax(dim=2)
        self.listener_norm = nn.LogSoftmax(dim=1)

    def forward_multiple_messages(self, feats, lang, lang_length):
        # Embed features
        feats_emb = self.embed_features(feats)

        # Embed language
        if torch.cuda.is_available():
            lang = lang.cuda()
            lang_length = lang_length.cuda()
        lang_emb = self.lang_model(lang, lang_length)

        # Bilinear term: lang embedding space -> feature embedding space
        lang_bilinear = self.bilinear(lang_emb)

        # Compute dot products
        outs = torch.matmul(feats_emb, torch.transpose(lang_bilinear, 1, 0))
        scores = self.listener_norm(outs)
        return scores

    
    def forward(self, img, lang, length):
        #(batch_size, num_obj, num_messages)
        if torch.cuda.is_available():
            self.messages.cuda()
            self.message_lengths.cuda()

        lis_log_scores = self.forward_multiple_messages(img, self.messages, self.message_lengths)
        

        for i in range(0, self.level):

            speaker_dist = self.speaker_norm((lis_log_scores - (self.message_lengths * self.cost)))
            lis_log_scores = self.listener_norm(speaker_dist)


        bools = ((self.messages - lang.unsqueeze(1)) == 0).sum(dim=2) == 4
        obj_given_lang = lis_log_scores.transpose(1,0)[:,bools].transpose(1,0)
        

        return torch.exp(obj_given_lang)







if __name__ == "__main__":
    listener_embs = nn.Embedding(len(VOCAB) + 4, 50)
    listener_vision = Conv4()
    listener = Listener(listener_vision, listener_embs)
    ce_loss = nn.CrossEntropyLoss()
    
    messages = create_all_messages()
    vocabulary = init_vocab(VOCAB)
    data_sw = np.load('/Users/knaszad/shapeworld/rsa_data/reference-1000-_54.npz')
    sw = RSAShapeWorld(data_sw, vocabulary)
    all_messages_ids, lengths = sw.to_idx(messages)
    rsa_speaker = RSAListener( (torch.tensor(all_messages_ids), torch.tensor(lengths)))
    dataloader = DataLoader(sw, batch_size=5, shuffle=True)
    for batch_i, (img, y, lang, length) in enumerate(dataloader):
        img = torch.transpose(img, 2, 4).double()
        lis_scores = rsa_speaker(img, lang)
