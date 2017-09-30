from __future__ import division
import numpy as np
from scipy import special as special
from itertools import chain,combinations
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_p_val_func_for(num_small ,num_total):
    """
    Returns function that evaluates p-value given sum_statisitc
    from num_small,num_total distribution
    """
    num_sums = int(round(special.binom(num_total,num_small)))
    vals = np.zeros(num_sums)
    i = 0
    # Note even though we are working in python, the rankings are starting with 1
    for combination in combinations(np.arange(num_total),num_small):
        vals[i] = sum(combination)
        i += 1
    vals = np.sort(vals)
    def get_p_val(obs_sum,vals = vals):
        num_bigger_or_equal_to = 0
        for val in vals:
            if val > obs_sum:
                num_bigger_or_equal_to += 1
        prop_bigger = float(num_bigger_or_equal_to) / float(len(vals))
        # Double Check Math but I think I got it
        p_value = 2*min(prop_bigger,1-prop_bigger)
        return p_value         
    return get_p_val

def compute_rank_sum_stat(depressed,happy):
    """
    Part 0 Compute CDF for given sample sizes (Assume already have)
    Part 1 get rank sum statistic for depressed
    Part 2 Use CDF to Tell if rank sum computed is "statistically significant"
    """
    concatenated = np.concatenate((depressed, happy), axis=0)
    sorted_lst = np.sort(concatenated)
    def compute_rank_of(x,sorted_lst):
        ix = np.isin(sorted_lst,x)
        loc = np.where(ix)*1
        tuple_to_num = lambda x : x[0][0]
        # Annoying: Have to add 1 due to index difference
        rank = tuple_to_num(loc) + 1
        return rank
    ranks = np.sum([compute_rank_of(depressed_brain,sorted_lst) for depressed_brain in depressed])
    return ranks


def compute_p_value(depressed,happy):
    """ 
    Compute p-value using rank Wilcoxon Rank Sum statistic to see it their is
    a difference between the connectivity levels for depressed and happy brains 
    """
    p_val = get_p_val_func_for(len(depressed),len(happy)+len(depressed))
    obs_sum = compute_rank_sum_stat(depressed,happy)
    return p_val(obs_sum)

if __name__ == "__main__":  
    depressed = np.random.uniform(2,10,1)
    happy = np.random.uniform(20,30,4)
    print(compute_p_value(depressed,happy))

    # test_statistic,p_value= stats.ranksums(depressed,happy)
    # print(test_statistic)
    # p_val_func = get_p_val_func_for(2,5) 
    # print(p_val_func(3))

