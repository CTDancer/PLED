import numpy as np
import os
import argparse
from consts import label_list, id2label, label2id, list_length, pascal_labels, coco_labels, nuswide_labels
import pdb

pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'])
pp.add_argument('--seed', type=int, default=42, required=False, help='random seed')
pp.add_argument('--total', type=int, default=500, required=False, help='maximum total number of images for each label')
args = pp.parse_args()

np.random.seed(args.seed)

base_path = os.path.join('./data/{}'.format(args.dataset))

label_matrix = np.load(os.path.join(base_path, 'reshaped_train_labels_obs.npy'))    # (N,L)
image_matrix = np.load(os.path.join(base_path, 'formatted_train_images.npy'))   # (N,)
clean_label_matrix = np.load(os.path.join(base_path, 'reshaped_train_labels.npy'))  #(N,L)
# pdb.set_trace()
assert(label_matrix.shape[0] == label_matrix.sum())
assert(label_matrix.shape[0] == image_matrix.shape[0] and label_matrix.shape[0] == clean_label_matrix.shape[0])

N, L = label_matrix.shape

if args.dataset == 'pascal':
    for i in range(L):
        mask = label_matrix[:, i] == 1
        indices = np.where(mask)[0]
        count = len(indices)
        choices = int(count - args.total)
        if choices > 0:
            selected_indices = np.random.choice(indices, choices, replace=False)
            keep_mask = ~np.isin(np.arange(label_matrix.shape[0]), selected_indices)
            label_matrix = label_matrix[keep_mask]
            image_matrix = image_matrix[keep_mask]
            clean_label_matrix = clean_label_matrix[keep_mask]
    np.save(os.path.join(base_path, 'sampled_train_labels_obs.npy'), label_matrix)
    np.save(os.path.join(base_path, 'sampled_train_images.npy'), image_matrix)
    np.save(os.path.join(base_path, 'sampled_train_labels.npy'), clean_label_matrix)

elif args.dataset == 'coco':
    pascal_labels = np.load('./data/pascal/sampled_train_labels_obs.npy')
    pascal_labels = pascal_labels.sum(axis=0)
    for i in range(L):
        mask = label_matrix[:, i] == 1
        indices = np.where(mask)[0]
        count = len(indices)
        choices = int(count + pascal_labels[i] - args.total)
        if choices > 0 and choices < count:
            selected_indices = np.random.choice(indices, choices, replace=False)
            keep_mask = ~np.isin(np.arange(label_matrix.shape[0]), selected_indices)
            label_matrix = label_matrix[keep_mask]
            image_matrix = image_matrix[keep_mask]
            clean_label_matrix = clean_label_matrix[keep_mask]
        elif choices >= count: # pascal_labels[i] >= args.total
            keep_mask = ~np.isin(np.arange(label_matrix.shape[0]), indices)
            label_matrix = label_matrix[keep_mask]
            image_matrix = image_matrix[keep_mask]
            clean_label_matrix = clean_label_matrix[keep_mask]
    np.save(os.path.join(base_path, 'sampled_train_labels_obs.npy'), label_matrix)
    np.save(os.path.join(base_path, 'sampled_train_images.npy'), image_matrix)
    np.save(os.path.join(base_path, 'sampled_train_labels.npy'), clean_label_matrix)
    
elif args.dataset == 'nuswide':
    pascal_labels = np.load('./data/pascal/sampled_train_labels_obs.npy')
    pascal_labels = pascal_labels.sum(axis=0)
    coco_labels = np.load('./data/coco/sampled_train_labels_obs.npy')
    coco_labels = coco_labels.sum(axis=0)
    for i in range(L):
        mask = label_matrix[:, i] == 1
        indices = np.where(mask)[0]
        count = len(indices)
        choices = int(count + pascal_labels[i] + coco_labels[i] - args.total)
        if choices > 0 and choices < count:
            # import pdb 
            # pdb.set_trace()
            selected_indices = np.random.choice(indices, choices, replace=False)
            keep_mask = ~np.isin(np.arange(label_matrix.shape[0]), selected_indices)
            label_matrix = label_matrix[keep_mask]
            image_matrix = image_matrix[keep_mask]
            clean_label_matrix = clean_label_matrix[keep_mask]
        elif choices >= count: # pascal_labels[i] + coco_labels[i] >= args.total
            keep_mask = ~np.isin(np.arange(label_matrix.shape[0]), indices)
            label_matrix = label_matrix[keep_mask]
            image_matrix = image_matrix[keep_mask]
            clean_label_matrix = clean_label_matrix[keep_mask]
            
    # # delete all the rows that are all zeros
    # mask = np.any(label_matrix != 0, axis = 1)
    # label_matrix = label_matrix[mask]
    # image_matrix = image_matrix[mask]
    np.save(os.path.join(base_path, 'sampled_train_labels_obs.npy'), label_matrix)
    np.save(os.path.join(base_path, 'sampled_train_images.npy'), image_matrix)
    np.save(os.path.join(base_path, 'sampled_train_labels.npy'), clean_label_matrix)
    
else:
    raise notImplementedError