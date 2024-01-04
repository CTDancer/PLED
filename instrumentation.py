import numpy as np
import copy
import metrics
import time
import torch
import pdb
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

class train_logger:

    '''
    An instance of this class keeps track of various metrics throughout
    the training process.
    '''

    def __init__(self, params):

        self.params = params

        # epoch-level objects:
        self.best_stop_metric = -np.Inf
        self.best_epoch = -1
        self.running_loss = 0.0
        self.num_examples = 0

        # batch-level objects:
        self.temp_preds = []
        self.temp_true = [] # true labels
        self.temp_obs = [] # observed labels
        self.temp_indices = [] # indices for each example
        self.temp_batch_loss = []
        self.temp_batch_reg = []

        # output objects:
        self.logs = {}
        self.logs['metrics'] = {}
        self.logs['best_preds'] = {}
        self.logs['gt'] ={}
        self.logs['obs'] = {}
        self.logs['targ'] = {}
        self.logs['idx'] = {}
        for field in self.logs:
            for phase in ['train', 'val', 'test']:
                self.logs[field][phase] = {}

    def compute_phase_metrics(self, phase, epoch):

        '''
        Compute and store end-of-phase metrics.
        '''

        self.logs['metrics'][phase][epoch] = {}

        # compute metrics w.r.t. clean ground truth labels:
        metrics_clean = compute_metrics(self.temp_preds, self.temp_true)
        for k in metrics_clean:
            self.logs['metrics'][phase][epoch][k + '_clean'] = metrics_clean[k]

        # compute metrics w.r.t. observed labels:
        metrics_observed = compute_metrics(self.temp_preds, self.temp_obs)
        for k in metrics_observed:
            self.logs['metrics'][phase][epoch][k + '_observed'] = metrics_observed[k]

        if phase == 'train':
            self.logs['metrics'][phase][epoch]['loss'] = self.running_loss / self.num_examples
            self.logs['metrics'][phase][epoch]['avg_batch_reg'] = np.mean(self.temp_batch_reg)
        else:
            self.logs['metrics'][phase][epoch]['loss'] = -999
            self.logs['metrics'][phase][epoch]['avg_batch_reg'] = -999
        self.logs['metrics'][phase][epoch]['preds_k_hat'] = np.mean(np.sum(self.temp_preds, axis=1))

    def get_stop_metric(self, phase, epoch, variant):
        '''
        Query the stop metric.
        '''
        assert variant in ['clean', 'observed']
        return self.logs['metrics'][phase][epoch][self.params['stop_metric'] + '_' + variant]

    def update_phase_data(self, batch):

        '''
        Store data from a batch for later use in computing metrics.
        '''

        for i in range(len(batch['idx'])):
            self.temp_preds.append(batch['preds_np'][i, :].tolist())
            self.temp_true.append(batch['label_vec_true'][i, :].tolist())
            self.temp_obs.append(batch['label_vec_obs'][i, :].tolist())
            self.temp_indices.append(int(batch['idx'][i]))
            self.num_examples += 1
        self.temp_batch_loss.append(float(batch['loss_np']))
        self.temp_batch_reg.append(float(batch['reg_loss_np']))
        self.running_loss += float(batch['loss_np'] * batch['image'].size(0))

    def reset_phase_data(self):

        '''
        Reset for a new phase.
        '''

        self.temp_preds = []
        self.temp_true = []
        self.temp_obs = []
        self.temp_indices = []
        self.temp_batch_reg = []
        self.running_loss = 0.0
        self.num_examples = 0.0

    def update_best_results(self, phase, epoch, variant):

        '''
        Update the current best epoch info if applicable.
        '''

        if phase == 'train':
            return False
        elif phase == 'val':
            assert variant in ['clean', 'observed']
            cur_stop_metric = self.get_stop_metric(phase, epoch, variant)
            if cur_stop_metric > self.best_stop_metric:
                self.best_stop_metric = cur_stop_metric
                self.best_epoch = epoch
                self.logs['best_preds'][phase] = self.temp_preds
                self.logs['gt'][phase] = self.temp_true
                self.logs['obs'][phase] = self.temp_obs
                self.logs['idx'][phase] = self.temp_indices
                return True # new best found
            else:
                return False # new best not found
        elif phase == 'test':
            if epoch == self.best_epoch:
                self.logs['best_preds'][phase] = self.temp_preds
                self.logs['gt'][phase] = self.temp_true
                self.logs['obs'][phase] = self.temp_obs
                self.logs['idx'][phase] = self.temp_indices
            return False

    def get_logs(self):

        '''
        Return a copy of all log data.
        '''

        return copy.deepcopy(self.logs)

    def report(self, t_i, t_f, phase, epoch):
        report = '[{}] time: {:.2f} min, loss: {:.3f}, {}: {:.2f}, {}: {:.2f}'.format(
            phase,
            (t_f - t_i) / 60.0,
            self.logs['metrics'][phase][epoch]['loss'],
            self.params['stop_metric'] + '_clean',
            self.get_stop_metric(phase, epoch, 'clean'),
            self.params['stop_metric'] + '_observed',
            self.get_stop_metric(phase, epoch, 'observed'),
            )
        print(report)


def compute_metrics(y_pred, y_true):

    '''
    Given predictions and labels, compute a few metrics.
    '''

    num_examples, num_classes = np.shape(y_true)

    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))

    results['map'] = 100.0 * float(np.mean(average_precision_list))

    for k in [1, 3, 5]:
        rec_at_k = np.array([metrics.compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        prec_at_k = np.array([metrics.compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
        results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
        results['top_{}'.format(k)] = np.mean(prec_at_k > 0)

    return results


def embedding_search(image_embedding, threshold, k, w, label_list=None):
    '''
    image_embedding: the embedding vector of the query image
    
    threshold: the cosine similarity between the image embedding and the only given positive label embedding
    
    k, w: hyperparameteres for determining positive, unknown, and negative labels
    
    label_list: The given list of labels of the query image in the dataset. It should not be None during training.
    '''
    fmt = "\n=== {:30} ===\n"
    search_latency_fmt = "search latency = {:.4f}s"

    # print(fmt.format("start connecting to Milvus"))
    connections.connect(alias="default", host="localhost", port="19530")

    has = utility.has_collection("label_embedding")
    # print(f"Does collection label_embedding exist in Milvus: {has}")
    
    # print(fmt.format("Start loading"))
    collection_name = 'label_embedding'
    collection = Collection(name=collection_name)
    collection.load()

    # print(fmt.format("Start searching based on vector similarity"))
    
    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False, 
        "params": {"nprobe": 150},
    }
    
    if type(image_embedding) == torch.Tensor:
        vector_to_search = image_embedding.detach().cpu().numpy()
    else:
        assert(type(image_embedding) == np.ndarray), "Image embedding with unexpected type"
        vector_to_search = image_embedding
    
    start_time = time.time()
    result = collection.search(
        data=[vector_to_search], 
        anns_field="embeddings", 
        param=search_params, 
        limit=collection.num_entities, 
        expr=None,
        output_fields=["label"],
        consistency_level="Strong")
    end_time = time.time()
    
    positive_labels = []
    unknown_labels = []
    negative_labels = []
    # pdb.set_trace()
    hit_distances = []
    for hits in result:
        for hit in hits:
            # if int(hit.distance*10000) == 1701:
            # if hit.entity.get('label') == 'remote':
            #     pdb.set_trace()
            hit_distances.append(hit.distance)
            if hit.distance >= k*threshold:
                positive_labels.append(hit.entity.get('label'))
            elif hit.distance < k*threshold and hit.distance >= w*threshold:
                unknown_labels.append(hit.entity.get('label'))
            else:
                negative_labels = []
    
    if label_list is not None:
        pseudo_positive_labels = [x for x in positive_labels if x in label_list]
        extra_positive_labels = [x for x in positive_labels if x not in pseudo_positive_labels]
        pseudo_unknown_labels = [x for x in unknown_labels if x in label_list]
        pseudo_negative_labels = [x for x in negative_labels if x in label_list]
        if len(pseudo_positive_labels) > 5:
            pseudo_positive_labels = pseudo_positive_labels[:5]
        if len(extra_positive_labels) > 10:
            extra_positive_labels = extra_positive_labels[:10]
        return hit_distances, pseudo_positive_labels, extra_positive_labels, pseudo_unknown_labels, pseudo_negative_labels
    else:
        return hit_distances, positive_labels, [], [], []

