#probability function
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor

def cond_probs_product(full_table, evidence_value, target_column, target_value):
  table_columns = up_list_column_names(full_table)  #new puddles function
  evidence_columns = table_columns[:-1]
  complete_evidence = up_zip_lists(evidence_columns, evidence_value)
  con_prob_list = []
  for line, line2 in complete_evidence:
     con_prob_list += [cond_prob(full_table, line, line2, target_column, target_value)]
  partial_numerator = up_product(con_prob_list)  #new puddles function
  return partial_numerator

def prior_prob(full_table, column, value):
  t_list = up_get_column(full_table, column)
  prob = sum([1 if v==value else 0 for v in t_list])/len(t_list)
  return(prob)

def naive_bayes(table, evidence_row, target):
  prob_zero = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)
  prob_one = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)
  neg, pos = compute_probs(prob_zero, prob_one)
  return[neg, pos]

#new metrics function
def metrics(parameter):
  assert isinstance(parameter, list), f'Expected parameter to be of type list but is instead {type(parameter)}'
  assert all(isinstance(value, list) for value in parameter), f'Your parameter is supposed to be a list of lists, but is not!'
  assert all(len(value) == 2 for value in parameter), f'Your list is not zipped- please zip it.'
  # assert all(isinstance(value[0], int) and isinstance(value[1], int) for value in parameter), f'Each value in pair should be an integer'
  assert all(value[0] >= 0 and value[1] >= 0 for value in parameter), f'Your values cannot be negative numbers'

  Accuracy = sum(p==a for p, a in parameter)/len(parameter)
  tp = sum([1 if pair==[1,1] else 0 for pair in parameter])
  fp = sum([1 if pair==[1,0] else 0 for pair in parameter])
  fn = sum([1 if pair==[0,1] else 0 for pair in parameter])
  Recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
  Precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
  F1 = 0 if Precision + Recall == 0 else 2*(Precision * Recall) / (Precision + Recall)
  return {'Precision': Precision, 'Recall': Recall, 'F1': F1, 'Accuracy' : Accuracy}


def generate_random(n):
  random_weights = [round(uniform(-1, 1), 2) for i in range(n)]
  return random_weights


#I'll give you a start

def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)

  for arch in architectures:
    all_results = up_neural_net(train_table, test_table, arch, target)

    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))
  


def testing():
  return f'loaded!'

