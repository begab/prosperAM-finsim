import os
import sys
import glob
import gzip
import json
import argparse

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

def load_sparse_embeddings(path, words_to_keep=None, max_words=-1):
    """
    Reads in the sparse embedding file.
    Parameters
    ----------
    path : str
        Location of the gzipped sparse embedding file
        If None, no filtering takes place.
    words_to_keep : list, optional
        list of words to keep
    max_words : int, optional
        Indicates the number of lines to read in.
        If negative, the entire file gets processed.
    Returns
    -------
    tuple:
        w2i:
            Wordform to identifier dictionary,
        i2w:
            Identifier to wordform dictionary,
        W:
            The sparse embedding matrix
    """

    i2w = {}
    data, indices, indptr = [], [], [0]
    with gzip.open(path, 'rt') as f:
        for line_number, line in enumerate(f):

            if line_number == max_words:
                break
            parts = line.rstrip().split(' ')

            if words_to_keep is not None and parts[0] not in words_to_keep:
                continue

            i2w[len(i2w)] = parts[0]
            for i, value in enumerate(parts[1:]):
                value = float(value)
                if value != 0:
                    data.append(float(value))
                    indices.append(i)
            indptr.append(len(indices))
    return {w: i for i, w in i2w.items()}, i2w, sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, i + 1))

def get_word_index(w2i, token):
    if token in w2i:
        return w2i[token]
    elif token.lower() in w2i:
        return w2i[token.lower()]
    else:
        return -1

def sparse_pmi(indices, vals, row_marginal, col_marginal, total, nonneg_pmi=True):
    pmis = np.ma.log((total * vals) / (row_marginal * col_marginal)).filled(0)
    pmis /= -np.ma.log(vals/total).filled(1)
    indices_to_return, pmis_to_return = [], []
    for idx in range(len(indices)):
        if not nonneg_pmi or pmis[0,idx] > 0:
            indices_to_return.append(indices[idx])
            pmis_to_return.append(pmis[0,idx])
    return indices_to_return, pmis_to_return

def calc_pmi(M):
    total, row_sum, col_sum = M.sum(), M.sum(axis=1), M.sum(axis=0)+1e-11
    data, indices, ind_ptr = [], [], [0]
    for i, r in enumerate(M):
        if np.any(r.data==0):
            zero_idx = np.where(r.data==0)[0]
            #logging.warning(("contains 0: ",i,self.id_to_label[i], [r.indices[z] for z in zero_idx]))
        idxs, pmi_values = sparse_pmi(r.indices, r.data, row_sum[i,0], col_sum[0, r.indices], total)
        indices.extend(idxs)
        data.extend(pmi_values)
        ind_ptr.append(len(data))
    return sp.csr_matrix((data, indices, ind_ptr), shape=(M.shape[0], M.shape[1]))

def get_rank(gold, list_predictions, max_k=3):
    list_predictions = list_predictions[:max_k]
    try:
        rank = list_predictions.index(gold) + 1
    except ValueError:
        rank = max_k + 1
    return rank


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_files', nargs='+', default='models/sparse_DLSC_cbow_1_2000.gz models/sparse_DLSC_cbow_3_1500.gz models/sparse_DLSC_cbow_3_2000.gz models/sparse_DLSC_cbow_4_1000.gz models/sparse_DLSC_cbow_4_1500.gz models/sparse_DLSC_cbow_5_1000.gz models/sparse_DLSC_cbow_5_2000.gz'.split())
  parser.add_argument('--out_dir', type=str, default='final_submission')
  parser.add_argument('--verbose', dest='verbose', action='store_true')
  parser.add_argument('--no-verbose', dest='verbose', action='store_false')
  parser.set_defaults(verbosity=False)

  parser.add_argument('--normalize', dest='normalize', action='store_true')
  parser.add_argument('--no-normalize', dest='normalize', action='store_false')
  parser.set_defaults(normalize=False)
  args = parser.parse_args()

  training_data = json.load(open('data/terms/train.json'))
  test_data = json.load(open('data/terms/test.json'))
  tags = {i:t['label'] for i, t in enumerate(json.load(open('data/tagset/finsim.json')))}
  golds = {}
  if os.path.exists('data/terms/gold.json'):
      golds = {g['term']:g['label'] for g in json.load(open('data/terms/gold.json'))}

  labels_to_ids = {v:k for k,v in tags.items()}

  if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

  aggregated_ranks, aggregated_corrects = [], []
  aggregated_train_predictions = [[] for _ in range(len(training_data))]
  aggregated_test_predictions = [[] for _ in range(len(test_data))]

  if args.input_files:
      files_used = sorted(args.input_files)
  else:
      files_used = sorted(glob.glob('models/*.gz'))

  print(len(files_used))

  for fn in files_used:
    print(fn)

    w2i, i2w, S = load_sparse_embeddings(fn)
    labels_to_vecs = {}

    oovs = {}
    for t in training_data:
      label = t['label']
      label_id = labels_to_ids[label]
      term_tokens = t['term'].split()
      oovs[t['term']] = []

      for ti, tt in enumerate([ttt for T in term_tokens for ttt in T.split('-')]):
        ind = get_word_index(w2i, tt)
        if ind==-1:
          oovs[t['term']].append(tt)
          continue

        vec = S[ind,:]

        if 'vec' in t:
            t['vec'] += vec
        else:
            t['vec'] = vec

      if 'vec' in t and args.normalize:
        if 'sparse' in fn:
          t['vec'].data /= t['vec'].sum()
        else:
          t['vec'].data /= np.linalg.norm(t['vec'].data)
      elif not 'vec' in t:
        t['vec'] = sp.csr_matrix((1, S.shape[1]))

      if label_id in labels_to_vecs:
          labels_to_vecs[label_id] += t['vec']
      else:
          labels_to_vecs[label_id] = t['vec']

    mtx = sp.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    etalon, predictions = [],[]
    ranking_scores = {}
    for i,t in enumerate(training_data):
      gold_label = labels_to_ids[t['label']]
      etalon.append(gold_label)

      mtx[gold_label] -= t['vec']
      if 'sparse' in fn:
        product = (-t['vec'] @ calc_pmi(mtx).T).todense()
      else:
        row_norms = np.linalg.norm(mtx.todense(), axis=1)
        M = mtx / row_norms[:, np.newaxis]
        product = np.array(-t['vec'] @ M.T)
      ranking_scores[t['term']] = product

      aggregated_train_predictions[i].append(product)

      ranked_labels = np.argsort(product)
      ranked_labels = [ranked_labels[0,r] for r in range(len(tags))][0:5]
      mtx[gold_label] += t['vec']
      if args.verbose and ranked_labels[0]!=gold_label:
          term = t['term']
          print('{}\t{}\t{}\t{}\tOOVs: {}'.format(i, term, t['label'], ' '.join([tags[r] for r in ranked_labels]), ' '.join(oovs[term])))
      predictions.append(ranked_labels)
      del training_data[i]['vec']

    corrects = 100*sum([1 if p[0]==g else 0 for g,p in zip(etalon, predictions)]) / len(etalon)
    aggregated_corrects.append(corrects)
    avg_rank_metric = np.mean([get_rank(g, p) for g,p in zip(etalon, predictions)])
    aggregated_ranks.append(avg_rank_metric)
    print("Accuracy_loo, rank: ", corrects, avg_rank_metric)

    if 'sparse' in fn:
      M = calc_pmi(mtx).toarray().T
    else:
      row_norms = np.linalg.norm(mtx.todense(), axis=1)
      M = np.transpose(mtx / row_norms[:, np.newaxis])
    
    gold_etalons, gold_predictions = [], []
    for i,t in enumerate(test_data):
      t['label'] = None
      gold_etalons.append(golds[t['term']])
      term_tokens = t['term'].split()

      for ti, tt in enumerate([ttt for T in term_tokens for ttt in T.split('-')]):
        ind = get_word_index(w2i, tt)
        if ind==-1: continue

        vec = S[ind,:]

        if 'vec' in t:
            t['vec'] += vec
        else:
            t['vec'] = vec

      if not 'vec' in t:
        t['vec'] = sp.csr_matrix((1, S.shape[1]))

      product = (-t['vec'] @ M)
      aggregated_test_predictions[i].append(product)
      ranked_labels = np.argsort(product)
      ranked_labels = [ranked_labels[0,r] for r in range(len(tags))]
      t['predicted_labels'] = [tags[r] for r in ranked_labels][0:5]
      gold_predictions.append(t['predicted_labels'])
      del t['vec']
    #print(len(test_data), t)
    corrects = 100*sum([1 if p[0]==g else 0 for g,p in zip(gold_etalons, gold_predictions)]) / len(gold_etalons)
    avg_rank_metric = np.mean([get_rank(g, p) for g,p in zip(gold_etalons, gold_predictions)])
    print("Accuracy_test, rank: ", corrects, avg_rank_metric)

    bn = os.path.basename(fn)
    with open('{}/{}.json'.format(args.out_dir, bn), 'w') as outfile:
      json.dump(test_data, outfile)

  correct = 3*[0]
  ranks = [[] for _ in range(3)]

  for i,(p,c) in enumerate(zip(aggregated_train_predictions, etalon)):
      stacked_scores = np.vstack(p)
      rankings = np.argsort(stacked_scores, axis=1)
      scores1 = np.zeros(rankings.shape[1])
      for r in np.array(rankings):
          for j,v in enumerate(r):
              scores1[v] += j

      row_norms = np.linalg.norm(stacked_scores, axis=1)
      scores2 = np.array(np.sum(stacked_scores / (row_norms[:, np.newaxis]+1e-9), axis=0)).flatten()

      scores3 = np.array(np.sum(stacked_scores, axis=0)).flatten()

      for si, scores in enumerate([scores1, scores2, scores3]):
          ranked_labels = np.argsort(scores).tolist()
          ranks[si].append(get_rank(c, ranked_labels))
          training_data[i]['predicted_labels'.format(si)] = [tags[r] for r in ranked_labels][0:5]
          if ranked_labels[0]==c:
              correct[si] += 1
          elif args.verbose:
              print(i, si, training_data[i])
  print(stacked_scores.shape, [correct[j] for j in range(3)], [np.mean(ranks[j]) for j in range(3)])

  test_predictions = []
  for test_ind in range(len(test_data)):
      test_predictions.append([])
      stacked_scores = np.vstack(aggregated_test_predictions[test_ind])
      rankings = np.argsort(stacked_scores, axis=1)
      scores = np.zeros(rankings.shape[1])
      for r in np.array(rankings):
          for j,v in enumerate(r):
              scores[v] += j
      test_predictions[-1].append([tags[sorted_ind] for sorted_ind in np.argsort(scores)])

      row_norms = np.linalg.norm(stacked_scores, axis=1)
      scores = np.array(np.sum(stacked_scores / (row_norms[:, np.newaxis]+1e-9), axis=0)).flatten()
      test_predictions[-1].append([tags[sorted_ind] for sorted_ind in np.argsort(scores)])

      scores = np.array(np.sum(stacked_scores, axis=0)).flatten()
      test_predictions[-1].append([tags[sorted_ind] for sorted_ind in np.argsort(scores)])

  for si in range(3):
      for ti, aggregated_preds in zip(test_data, test_predictions):
          ti['predicted_labels'] = aggregated_preds[si]
      with open('{}/prosperamnet_{}_predictions.json'.format(args.out_dir, si+1), 'w') as outfile:
          json.dump(test_data, outfile)
