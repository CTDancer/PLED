import numpy as np
import os
import argparse
from consts import label_list, id2label, label2id, list_length, pascal_labels, coco_labels, nuswide_labels

pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'])
args = pp.parse_args()

base_path = os.path.join('./data/{}'.format(args.dataset))

if args.dataset == 'pascal':
    labels = pascal_labels
elif args.dataset == 'coco':
    labels = coco_labels
elif args.dataset == 'nuswide':
    labels = nuswide_labels
else:
    raise NotImplementedError

if args.dataset == 'nuswide':
    for phase in ['train', 'val']:
        label_matrix = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))    # formatted_{}_labels_obs.npy
        N, L = label_matrix.shape
        reshaped = np.zeros((N, list_length))
        reshaped[:, :L] = label_matrix

        np.save(os.path.join(base_path, 'reshaped_{}_labels.npy'.format(phase)), reshaped)  # reshaped_{}_labels.npy
else:
    for phase in ['train', 'val']:
        # load ground truth binary label matrix:
        label_matrix = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))    # formatted_{}_labels_obs.npy
        assert np.max(label_matrix) == 1
        assert np.min(label_matrix) == 0
        
        N, _ = label_matrix.shape
        reshaped = np.zeros((N, list_length))
        
        for i in range(N):
            index = np.where(label_matrix[i])[0]
            for j in index:
                reshaped[i][label2id[labels[j]]] = 1
        
        np.save(os.path.join(base_path, 'reshaped_{}_labels.npy'.format(phase)), reshaped)  # reshaped_{}_labels_obs.npy
    