from shapely.geometry import Point, box
from shapely import affinity
import numpy as np
import random
import math
from PIL import Image
import aggdraw
from enum import Enum
from tqdm import tqdm
import os
import multiprocessing as mp
from collections import namedtuple
from scipy.stats import dirichlet
from itertools import product
from data import ShapeWorld


from shapeworld import ShapeSpec, ConfigProps, SingleConfig, I, SHAPE_IMPLS


DIM = 64
X_MIN, X_MAX = (8, 48)
ONE_QUARTER = (X_MAX - X_MIN) // 3
X_MIN_34, X_MAX_34 = (X_MIN + ONE_QUARTER, X_MAX - ONE_QUARTER)
BUFFER = 10
SIZE_MIN, SIZE_MAX = (3, 8)

TWOFIVEFIVE = np.float32(255)

SHAPES = ['circle', 'square', 'rectangle', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
VOCAB = ['gray', 'shape', 'blue', 'square', 'circle', 'green', 'red', 'rectangle', 'yellow', 'ellipse', 'white']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}


MAX_PLACEMENT_ATTEMPTS = 5

PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def init_vocab(langs):
    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for tok in langs:
        if tok not in w2i:
            i = len(w2i)
            w2i[tok] = i
            i2w[i] = tok
    return {'w2i': w2i, 'i2w': i2w}

class RSAShapeWorld(ShapeWorld):
    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t.split()) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(langs), max(lang_len)), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks.split(), start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len

    def __getitem__(self, i):
        # Reference game format.
        img = self.imgs[i]
        label = self.labels[i]
        lang = self.lang_idx[i]
        length = self.lang_len[i]
        return (img, label, lang, length)

class RSAShapeWorldGenerator:
    def __init__(self, num_distractors, P_S, P_S_S, P_C_S, vocab, level, max_len = 4, cost=0):
        self.num_distractors = num_distractors
        self.P_S = P_S
        self.P_S_S = P_S_S
        self.P_C_S = P_C_S
        # Get vocab
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']
        self.max_len = max_len
        self.level = level
        self.cost = cost

    def __len__(self):
        return (( len(SHAPES) * len(COLORS)) ** 3) * 3


    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t.split()) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(langs), self.max_len), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks.split(), start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len


    def __getitem__(self,ind):
        imgs = np.zeros((self.num_distractors + 1, 64, 64, 3), dtype=np.uint8)

        # there is no uniquely identifiable target in these configs
        configs = sample_scenario(self.num_distractors, self.P_S, self.P_S_S, self.P_C_S)
        target_label = generate_target(configs)
        while target_label is None:
            configs = sample_scenario(self.num_distractors, self.P_S, self.P_S_S, self.P_C_S)
            target_label = generate_target(configs)
        message = generate_messages(configs, target_label, speaker_level=self.level, cost = self.cost)
        for i in range(0, len(configs)):
            img = I()
            obj = SHAPE_IMPLS[configs[i].shape](color=configs[i].color)
            img.draw_shapes([obj])
            imgs[i] = img.array()
        labels = np.zeros(self.num_distractors + 1)
        labels[target_label] = 1

        full_message =" ".join(message)
        return (imgs, labels,self.to_idx([full_message])[0].squeeze(), len(message) + 2)



def generate_distributions(scale_factor = 1, max_alpha = 1):
    P_S = dirichlet(alpha=scale_factor * np.ones(len(SHAPES))).rvs(1)[0]
    P_S_S = {}
    P_C_S = {}
    for i, shape in enumerate(SHAPES):
        d_o = np.ones(len(SHAPES))
        d_o[i] = max_alpha
        p_s_s = dirichlet(alpha=scale_factor * d_o).rvs(1)[0]
        P_S_S[shape] = p_s_s
    for i, shape in enumerate(SHAPES):
        d_o = np.ones(len(COLORS))
        p_c_s = dirichlet(alpha=scale_factor * d_o).rvs(1)[0]
        P_C_S[shape] = p_c_s
    return P_S, P_S_S, P_C_S

def sample_scenario(
    num_distractors,
    P_S,
    P_S_S,
    P_C_S):
    configs = []
    S_0 = random.choices(SHAPES, weights=P_S, k=1)[0]
    C_0 = random.choices(COLORS, weights=P_C_S[S_0], k=1)[0]
    configs.append(SingleConfig(S_0, C_0))

    for n in range(0, num_distractors):
        S_N = random.choices(SHAPES, weights=P_S_S[S_0], k=1)[0]
        C_N = random.choices(COLORS, weights=P_C_S[S_N], k=1)[0]
        configs.append(SingleConfig(S_N, C_N))
    return configs

def check_target_unique(
        config,
        configs
):
    found = 0
    for o_config in configs:
        if o_config == config:
            found +=1
    if found != 1:
        return False
    return True



def generate_target(
        configs):

    target_label = None
    for label in np.random.permutation(len(configs)):
        target_config = configs[label]
        unique = check_target_unique(target_config, configs)
        if unique:
            target_label = label
            break
    return target_label


def create_association_table(configs):
    rows = []
    for config in configs:
        row = []
        for word in SHAPES:
            if config.shape == word:
                row.append(1)
            else:
                row.append(0)
        for word in COLORS:
            if config.color == word:
                row.append(1)
            else:
                row.append(0)
        for shape, color in product(SHAPES, COLORS):
            if shape == config.shape and color == config.color:
                row.append(1)
            else:
                row.append(0)
        rows.append(row)
    return np.array(rows)

def compute_listener_dist(speaker_dist):
    denom = speaker_dist.sum(axis=0, keepdims=True)
    return np.nan_to_num(speaker_dist/ denom)

def compute_speaker_dist(listener_dist, cost=0):
    message_lengths = np.concatenate([
        np.ones(len(SHAPES) + len(COLORS)),
        np.ones(len(SHAPES) * len(COLORS)) * 2
        ]
    )
    costs = cost * message_lengths
    with_cost_dist = np.log(listener_dist) - costs[np.newaxis, :]
    denom = np.sum(np.exp(with_cost_dist), axis=1, keepdims=True)
    out_dist = np.exp(with_cost_dist) / denom
    return out_dist

def create_all_messages():
    all_messages = []
    all_messages += SHAPES
    all_messages += COLORS
    all_messages += [" ".join([shape, color]) for shape, color in product(SHAPES, COLORS)]
    return all_messages


def generate_messages(configs, target_label, cost = 0, speaker_level=1):
    all_messages = [(shape,) for shape in SHAPES]
    all_messages += [(color,) for color in COLORS]
    all_messages += [(shape, color) for shape, color in product(SHAPES, COLORS)]
    base_array = create_association_table(configs)
    listener_dist = compute_listener_dist(base_array)
    speaker_dist = compute_speaker_dist(listener_dist, cost=cost)
    if speaker_level > 1 :
        for i in range(1,speaker_level):
            listener_dist = compute_listener_dist(speaker_dist)
            speaker_dist = compute_speaker_dist(listener_dist, cost=cost)


    words_given_target = speaker_dist[target_label, :]
    indx_of_messages = np.argwhere(words_given_target == np.amax(words_given_target))
    ind_of_message = random.choice(indx_of_messages)
    message_to_send = all_messages[ind_of_message.item()]

    return message_to_send




def generate_rsa(num_samples, num_distractors, P_S, P_S_S, P_C_S):
    all_imgs = []
    all_labels = []
    all_messages = []
    all_configs = []
    while len(all_imgs) < num_samples:
        imgs = np.zeros((num_distractors + 1, 64, 64, 3), dtype=np.uint8)
        configs = sample_scenario(num_distractors, P_S, P_S_S, P_C_S)
        target_label = generate_target(configs)
        #there is no uniquely identifiable target in these configs
        if target_label is None:
            continue
        message = generate_messages(configs, target_label, speaker_level=1)
        for i in range(0,len(configs)):
            img = I()
            obj = SHAPE_IMPLS[configs[i].shape](color=configs[i].color)
            img.draw_shapes([obj])
            imgs[i] = img.array()
        all_imgs.append(imgs)
        labels = np.zeros(num_distractors+1)
        labels[target_label] = 1
        all_labels.append(labels)
        all_messages.append(" ".join(message))
        all_configs.append(configs)
    all_messages_np = np.array(all_messages)
    all_imgs_np = np.stack(all_imgs)
    all_labels_np = np.stack(all_labels)
    return {'imgs': all_imgs_np, 'labels': all_labels_np, 'langs': all_messages_np, 'configs': configs}


if __name__ == '__main__':
    P_S, P_S_S, P_C_S = generate_distributions(scale_factor=1, )
    data = generate_rsa(1000, 2, P_S, P_S_S, P_C_S)

    data_dir = '/Users/knaszad/shapeworld/rsa_data/reference-S_S-100-2D'
    for i in  range(0,100):
        data = generate_rsa(1000, 2, P_S, P_S_S, P_C_S)
        np.savez_compressed(f'{data_dir}_{i}.npz', **data)

