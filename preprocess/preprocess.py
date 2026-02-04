import pandas as pd
import numpy as np 

def preprocess(data):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def preprocess_thgl(tgb_data):
    """
    Converts TGB dictionary data directly to the DataFrame and feature array
    expected by the TGN implementation.
    """

    # Extract arrays from TGB dictionary
    sources = tgb_data['sources']
    destinations = tgb_data['destinations']
    timestamps = tgb_data['timestamps']

    # Handle Labels: TGB uses 'edge_label' or 'w'.
    # If missing, we assume 1 (existence of link) or 0.
    if 'edge_label' in tgb_data:
        labels = tgb_data['edge_label']
    elif 'w' in tgb_data:
        labels = tgb_data['w']
    else:
        # Default to all zeros if no specific edge labels exist
        labels = np.zeros(len(sources))

    # Handle Edge Features
    if 'edge_feat' in tgb_data and tgb_data['edge_feat'] is not None:
        edge_features = tgb_data['edge_feat']
    else:
        # Create dummy edge features if none exist (Shape: [Num_Edges, 1])
        # TGN usually requires at least 1 dimension for edge features
        print("No edge features found. Creating dummy zero features.")
        edge_features = np.zeros((len(sources), 1))

    # Create the DataFrame expected by your pipeline
    df = pd.DataFrame({
        'u': sources,
        'i': destinations,
        'ts': timestamps,
        'label': labels,
        'idx': np.arange(len(sources))  # Simple 0 to N index
    })

    return df, edge_features

def reindex(df, bipartite=True):
    new_df = df.copy()

    # Note: thgl-software is likely NOT bipartite (package A -> package B).
    # If you treat it as bipartite, node IDs will be shifted to be disjoint.
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        # Standard 1-based indexing for homogeneous graphs
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    max_idx = max(new_df.u.max(), new_df.i.max())

    rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv('ml_thgl_software.csv', index=False)

    print(f"Processed {len(new_df)} edges.")
    print(f"Max Node ID: {max_idx}")

    return new_df, rand_feat