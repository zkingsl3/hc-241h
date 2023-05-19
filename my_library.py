#probability function
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(full_table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(full_table, target, "equals", target_value)     #subset of table where target column has target value
  e_list = up_get_column(t_subset, evidence) 
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a

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

def testing():
  return 'loaded!'

