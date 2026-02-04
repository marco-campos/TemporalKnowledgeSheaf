import torch
from torch import nn
import numpy as np
from collections import defaultdict
from copy import deepcopy

class Memory(nn.Module):
  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device= 'cpu', combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device
    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):

    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                                requires_grad = False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    # Use .data to perform in-place modification without affecting autograd graph
    self.memory.data[node_idxs, :] = values.data

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def set_last_update(self, node_idxs, values):
    # Use .data to perform in-place modification without affecting autograd graph
    self.last_update.data[node_idxs] = values.data if isinstance(values, torch.Tensor) else values

  def backup_memory(self):
    messages_clone = {}

    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    self.memory.detach()

    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []

class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"
    # Get the memory slice BEFORE updating last_update, as get_memory uses current state
    memory_slice = self.memory.get_memory(unique_node_ids)

    # Use the new setter which operates on .data
    self.memory.set_last_update(unique_node_ids, timestamps)

    # Perform the GRU update. `unique_messages` typically comes from `MLPMessageFunction` output, which will have requires_grad=True.
    # `memory_slice` comes from `self.memory.memory[unique_node_ids, :]`, which is a view of requires_grad=False Parameter.
    # The output `updated_memory_tensor` will have requires_grad=True because of unique_messages.
    updated_memory_tensor = self.memory_updater(unique_messages, memory_slice)

    # Use the new setter which operates on .data. This takes `updated_memory_tensor.data` for in-place modification.
    self.memory.set_memory(unique_node_ids, updated_memory_tensor)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_last_update = self.memory.last_update.data.clone()

    # Get the relevant slice from the *detached* current memory state
    current_memory_slice = updated_memory[unique_node_ids]

    # Apply the GRU update logic. The memory_updater output will have its own grad history.
    # It should be used to update the `updated_memory` (the clone of .data) at specified indices.
    updated_slice = self.memory_updater(unique_messages, current_memory_slice)

    updated_memory[unique_node_ids] = updated_slice
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)