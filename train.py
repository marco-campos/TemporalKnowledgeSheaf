import torch
import math
import time
import copy
import torch.autograd

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
from tgb.linkproppred.dataset import LinkPropPredDataset
import pandas as pd
import numpy as np

from preprocess.preprocess import preprocess_thgl, reindex
from preprocess.data import get_data, compute_time_statistics
from eval.sampler import RandEdgeSampler
from eval.eval import eval_edge_prediction
from model.tks import TGN
from model.neighbor import get_neighbor_finder
from utils.utils import EarlyStopMonitor

# # data loading
dataset = LinkPropPredDataset(name= 'thgl-software', root="datasets", preprocess=True)
data = dataset.full_data
metric = dataset.eval_metric
sources = dataset.full_data['sources']
print ("finished loading numpy arrays")

thgl_df, thgl_feat = preprocess_thgl(data)

procesed_df, thgl_node_feat = reindex(thgl_df, bipartite=False)

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(thgl_feat,thgl_node_feat, dataset_name= '/data/ml_thgl_software.csv',
                              different_new_nodes_between_val_and_test=True,
                              randomize_features=True)
# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, True)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, True)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

############################################################################################################

tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=1,
            n_heads=2, dropout=0.1, use_memory=True,
            message_dimension=100, memory_dimension=172,
            memory_update_at_start=False,
            embedding_module_type='graph_sum',
            message_function='identity',
            aggregator_type='last',
            memory_updater_type='gru',
            n_neighbors=10,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=True,
            use_source_embedding_in_message=True,
            dyrep=True)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(tgn.parameters(), lr=0.0001)
tgn = tgn.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / 200)

print('num of training instances: {}'.format(num_instance))
print('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)

new_nodes_val_aps = []
val_aps = []
epoch_times = []
total_epoch_times = []
train_losses = []

USE_MEMORY = True
NUM_EPOCH = 50
NUM_NEIGHBORS = 10
BATCH_SIZE = 200

torch.autograd.set_detect_anomaly(True)

early_stopper = EarlyStopMonitor(max_round= 5)

############################################################################################################

for epoch in range(NUM_EPOCH):
  start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
  if USE_MEMORY:
    tgn.memory.__init_memory__()

    # Train using only training graph
  tgn.set_neighbor_finder(train_ngh_finder)
  m_loss = []

  print('start {} epoch'.format(epoch))
  for k in range(0, num_batch, 1):
    loss = 0
    optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
    for j in range(1):
      batch_idx = k + j

      if batch_idx >= num_batch:
        continue

      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
      edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
      timestamps_batch = train_data.timestamps[start_idx:end_idx]

      size = len(sources_batch)
      _, negatives_batch = train_rand_sampler.sample(size)

      with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)

      tgn = tgn.train()
      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

      loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

    loss /= 1

    loss.backward()
    optimizer.step()
    m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
    if USE_MEMORY:
      tgn.memory.detach_memory()

  epoch_time = time.time() - start_epoch
  epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
  tgn.set_neighbor_finder(full_ngh_finder)

  if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
    train_memory_backup = tgn.memory.backup_memory()

  val_ap, val_auc = eval_edge_prediction(model=tgn,
                                        negative_edge_sampler=val_rand_sampler,
                                        data=val_data,
                                        n_neighbors=NUM_NEIGHBORS)
  if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
  nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
      # Restore memory we had at the end of validation
    tgn.memory.restore_memory(val_memory_backup)

  new_nodes_val_aps.append(nn_val_ap)
  val_aps.append(val_ap)
  train_losses.append(np.mean(m_loss))


  total_epoch_time = time.time() - start_epoch
  total_epoch_times.append(total_epoch_time)

  print('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
  print('Epoch mean loss: {}'.format(np.mean(m_loss)))
  print(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
  print(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
  if early_stopper.early_stop_check(val_ap, tgn):
    print(f'No improvement over {early_stopper.max_round} epochs, stop training')
    print(f'Loading the best model at epoch {early_stopper.best_epoch}')
    tgn.load_state_dict(early_stopper.best_state)
    print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    tgn.eval()
    break

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
if USE_MEMORY:
  val_memory_backup = tgn.memory.backup_memory()

### Test
tgn.embedding_module.neighbor_finder = full_ngh_finder
test_ap, test_auc = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)