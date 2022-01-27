# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:18:11 2021

@author: Mark York
"""

'''
This script defines a set of borrowers and a set of agents who have beliefs about those
borrowers. The system then simulates the outcomes of different rating strategies
for both recommenders and the principal in terms of repayments and recommender compensation
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.stats import rankdata


def main(n_recommenders, n_borrowers, budget, c):
    #inputs = recommenders, borrowers, budget, lending threshold c
    #This function simulates the truncated-winkler or 0-1 VCG for n_rounds rounds

    #0 Parameters
    '''
    #2021 Feb recommender knowledge parameters
    alpha = .75 #This represents the quality of recommenders' knowledge of borrower repayment probability. Alpha is the weight of the true probabilities, and 1-alpha is the weight of random noise in recommender beliefs
    p = alpha*np.tile(repayment_probs, (n_recommenders, 1)) + \
            (1-alpha)*np.random.random((n_recommenders, n_borrowers)) #recommender beliefs - |N| x |M|
    '''
    mechanism = 'truncated_winkler' #'truncated_winkler', '0_1_vcg'
    shift = beta.rvs(.5, 5, size=n_recommenders) #The upward bias of each recommender
    rec_accuracy_a_b_vector = [10]*n_recommenders #value to use as a and b in the beta distribution for each recommender's accuracy. True belief will be trim(true_prob + skew + beta(a,b) - .5, 0, 1)
    rec_accuracy_a_b_matrix = np.tile(rec_accuracy_a_b_vector,(n_borrowers,1)).T
    n_rounds = 10
    use_weights = True

    #0.1 Honesty Parameters
    honesty_type = 'honest', #{'honest', 'random_misreports', 'misreports_select_recommenders', 'collusion'}
    rand_misreport_rate = .1 #fraction of reports which are misreports under 'random_misreports' honesty type
    frac_misreports_1 = .5 #fraction of misreports which are 1 in all honesty types. All misreports other than 1 are 0.
    frac_dishonest_recommenders = .2 #fraction of recommenders who are dishonest in 'misreports_select_recommenders' and 'collusion' honesty types
    frac_misreports_by_dishonest_recs = .5 #In honesty type 'misreports_select_recommenders', the fraction of reports by dishonest recommenders which are misreports. In honesty type 'collusion', this is the fraction of all potential borrowers on whom dishonest recommenders misreport

    #0.2 Setting Recommender Honesty
    recommender_honesty = np.zeros(n_recommenders)
    recommender_honesty = np.where(rankdata(np.random.random(n_recommenders)) \
                        > frac_dishonest_recommenders*n_recommenders,\
                        1, 0) #1 for honest, 0 for dishonest

    #0.3 Creating lists where round-by-round data will be stored
    n_loans_made = []
    n_repayments = []
    mean_comp = []
    std_comp = []
    rec_comp_negative_pct = []

    for i in range(n_rounds):
        if use_weights == True and (i == 0 or np.sum(multi_round_lending_decisions) \
                    == 0): weights = np.ones(n_recommenders)/n_recommenders #Use equal weights for all recommenders
        elif use_weights == True: weights = \
                        generate_weights(reports, multi_round_lending_decisions,\
                        multi_round_outcomes, n_recommenders, n_borrowers)
        print('Round: ', i, 'weights: ',weights)
        #1 Borrow Prob and Recommender Belief Creation
        repayment_probs = beta.rvs(4,1, size=n_borrowers) #True repayment probability for each borrower
        '''
        print('repayment_probs',repayment_probs)
        print('n_recommenders',n_recommenders)
        print('shift',shift)
        print('n_borrowers',n_borrowers)
        print('rec_accuracy_a_b_matrix',rec_accuracy_a_b_matrix)
        print('np.tile(repayment_probs, (n_recommenders,1))',np.tile(repayment_probs, (n_recommenders,1)))
        print('np.tile(shift, (n_borrowers,1)).T',np.tile(shift, (n_borrowers,1)).T)
        print('beta.rvs(rec_accuracy_a_b_matrix, rec_accuracy_a_b_matrix, size=(n_recommenders, n_borrowers))',\
                    beta.rvs(rec_accuracy_a_b_matrix, rec_accuracy_a_b_matrix, size=(n_recommenders, n_borrowers)))
        '''
        p = np.tile(repayment_probs, (n_recommenders,1)) + np.tile(shift, \
                            (n_borrowers,1)).T + beta.rvs(rec_accuracy_a_b_matrix, \
                            rec_accuracy_a_b_matrix, size=(n_recommenders, \
                            n_borrowers)) - .5
        p = np.clip(p,0,1) #ensures that no values lie outside of 0 and 1

        #2 Reporting / Misreporting Mechanism
        p_hat = p.copy() #truthful reporting
        if honesty_type == 'random_misreports':
            n_reports = n_recommenders * n_borrowers
            rand_ranks = rankdata(np.random.random(p.shape))
            np.where(rand_ranks <= n_reports * rand_misreport_rate * \
                     (1 - frac_misreports_1), p_hat = 0)
            np.where(rand_ranks > n_reports * rand_misreport_rate * \
                     (1 - frac_misreports_1) and rand_ranks <= \
                     n_reports * rand_misreport_rate, p_hat = 1)
        '''
        elif honesty_type == 'misreports_select_recommenders':
            for i in range(n_recommenders):
                if i == 1.: continue
                rand_ranks = rankdata(np.random.random(n_borrowers))
                np.where(rand_ranks <= n_borrowers*frac_misreports_by_dishonest_recs \
                     * (1 - frac_misreports_1), p_hat[i] = 0)
                np.where(rand_ranks > n_borrowers*frac_misreports_by_dishonest_recs \
                     * (1 - frac_misreports_1) and rand_ranks <= \
                     n_borrowers*frac_misreports_by_dishonest_recs, p_hat[i] = 1)
        elif honesty_type == 'collusion':
             for i in range(n_recommenders):
                rand_ranks = rankdata(np.random.random(n_borrowers))
                if i == 1.: continue
                np.where(rand_ranks <= n_borrowers*frac_misreports_by_dishonest_recs \
                     * (1 - frac_misreports_1), p_hat[i] = 0)
                np.where(rand_ranks > n_borrowers*frac_misreports_by_dishonest_recs \
                     * (1 - frac_misreports_1) and rand_ranks <= \
                     n_borrowers*frac_misreports_by_dishonest_recs, p_hat[i] = 1)
       '''

        #3 Repayment and Recommender Compensation Calculation
        if mechanism == 'truncated_winkler':
                    expected_payout, actual_payout, lending_decisions, \
                    repayment_outcomes = truncated_winkler(p, p_hat, \
                    repayment_probs, c, weights)
        elif mechanism == '0_1_vcg':
                    expected_payout, actual_payout, lending_decisions, \
                    repayment_outcomes = vcg(p, p_hat, \
                    repayment_probs, budget, c, weights)

        #4 Data Calculation & Storage
        if mechanism == 'truncated_winkler': comp_by_recommender = np.sum(actual_payout, axis=1)
        else: comp_by_recommender = actual_payout
        n_loans_made += [np.sum(lending_decisions)]
        n_repayments += [np.sum(repayment_outcomes)]
        mean_comp += [np.mean(comp_by_recommender)]
        std_comp += [np.std(comp_by_recommender)]
        rec_comp_negative_pct += [np.sum(np.where(comp_by_recommender < 0, 1,\
                            0)) / p.shape[0]]
        if i == 0 and use_weights == True:
            reports = p_hat
            multi_round_lending_decisions = lending_decisions
            multi_round_outcomes = repayment_outcomes
        elif i > 0 and use_weights == True:
            reports = np.hstack((reports, p_hat))
            multi_round_lending_decisions = np.hstack((multi_round_lending_decisions, \
                        lending_decisions))
            multi_round_outcomes = np.hstack((multi_round_outcomes,repayment_outcomes))

        #5 Printing
        '''
        print('lending_decisions',lending_decisions)
        #print('expected_payout',expected_payout)

        #Results
        print('Results: ')
        repayment_rate = np.sum(repayment_outcomes) / np.sum(lending_decisions)
        #print('repayment_outcomes',repayment_outcomes)
        #print('actual_payout',actual_payout)
        print('repayment_rate',repayment_rate)
        print('mean_rec_comp_std',np.mean(comp_by_recommender))
        print('rec_comp_negative_pct',rec_comp_negative_pct)
        '''
    return np.asarray(n_loans_made), np.asarray(n_repayments), \
                np.asarray(mean_comp), np.asarray(std_comp), \
                np.asarray(rec_comp_negative_pct)

def truncated_winkler(p, p_hat, repayment_probs, c, weights):
    #This function runs the truncated winkler scoring system with the given parameters
    base_rule = 'quadratic' #logarithmic option not yet programmed up.
    #0 Lending Decisions
    #print('repayment_probs',repayment_probs)

    #lending_decisions = np.where(p_hat.sum(axis=0) > c*p.shape[0], 1, 0) #w/o weights
    lending_decisions = np.where(np.matmul(weights.T, p_hat) > c, 1, 0)
    #print('lending_decisions',lending_decisions)

    #1.0 Calculate Thresholds ciq
    ciq = np.zeros(p.shape) #matrix of lending thresholds by recommender / borrower combo
    for i in range(p.shape[0]): #index over recommenders
        #ciq[i,:] = p.shape[0] * c - (p_hat.sum(axis=0) - p_hat[i]) #w/o weights
        if weights[i] < 10**-6: ciq[i,:] = np.ones(p.shape[1]) #this person has no weight and will not be able to decide who gets a loan.
        else:
            ciq[i,:] = (c - np.matmul(weights.T, p_hat) + weights[i]*p_hat[i,:])/weights[i]
    ciq = np.clip(ciq, 0.01, .99)
    #print('ciq',ciq)

    #1.5 Calculate expected scores for each recommender and borrower
    if base_rule == 'quadratic':
        scores_repay, scores_default = winkler_quadratic_scores(p_hat, ciq,\
                        lending_decisions)
    expected_payout = np.multiply(scores_repay, p) + np.multiply(\
                        scores_default, (1-p))

    #2 Repayment outcomes
    repayment_outcomes = np.multiply(np.where(np.random.random(\
                lending_decisions.shape) > (1-repayment_probs), 1, 0),\
                lending_decisions)
    actual_payout = np.multiply(scores_repay, repayment_outcomes) + \
                np.multiply(scores_default, (1-repayment_outcomes)) #payouts when all recommenders have weight 1
    actual_payout = np.multiply(actual_payout, \
                np.asarray(weights).reshape((len(weights),1))) #payouts scaled by recommender weight

    if np.abs(np.sum(actual_payout)) > 10**5: #Loop to check into extreme value problems
        print('weights',weights)
        print('p_hat',p_hat)
        print('lending_decisions',lending_decisions)
        print('repayment_outcomes',repayment_outcomes)
        print('expected_payout',expected_payout)
        print('actual_payout',actual_payout)

        print('p',p)
        print('scores_repay',scores_repay)
        print('scores_default',scores_default)
        exit()

    return expected_payout, actual_payout, lending_decisions, repayment_outcomes

def winkler_quadratic_scores(p_hat, ciq, lending_decisions):
    #Returns matrices of payouts in the case of repayment or default. 0 where
    #loans not made. Assumes all recommenders have weight 1 (we weight when calculating actual payout)
    denominator = 2 - 4*ciq + 2 * ciq*ciq
    scores_repay = (2*p_hat - p_hat*p_hat - (1-p_hat)*(1-p_hat) - 2*ciq + \
                    ciq*ciq + (1-ciq)*(1-ciq)) / denominator
    scores_default = (2*(1-p_hat) - p_hat*p_hat - (1-p_hat)*(1-p_hat) - \
                      2*(1-ciq) + ciq*ciq + (1-ciq)*(1-ciq)) / denominator

    if np.max(scores_repay) > 10**5 or np.min(scores_repay) < -10**4: #Loop to check into extreme value problems
        print('p_hat',p_hat)
        print('ciq',ciq)
        print('denominator',denominator)
        print('scores_repay',scores_repay)
        print('scores_default',scores_default)
        exit()

    return lending_decisions*scores_repay, lending_decisions*scores_default


def vcg(p, p_hat, repayment_probs, budget, c, weights):
    #This function runs the VCG scoring mechanism with a reserve
    #0 Setup
    n_recommenders = p.shape[0]
    n_borrowers = p.shape[1]
    #augment p_hat with reserve borrowers and reserve recommender
    p_hat = np.vstack((p_hat, np.zeros((1,n_borrowers))))
    tmp_array = np.vstack((np.zeros((n_recommenders,budget)), np.full((1,budget),\
                    c*n_recommenders)))
    p_hat = np.hstack((p_hat, tmp_array))

    #1 Lending decisions
    #sums = np.sum(p_hat, axis = 0) #w/o weights
    sums = np.matmul(p_hat, weights.T)
    sums.sort()
    threshold = sums[-budget]
    sums = np.matmul(p_hat, weights.T)
    lending_decisions = np.where(sums[:n_borrowers] >= threshold, 1, 0) #Will not allocate any reserve borrowers yet
    lending_decisions = np.hstack((lending_decisions, np.zeros((budget,))))

    if np.sum(lending_decisions) < budget: #allocates just enough reserve borrowers to fill the budget quota
        lending_decisions[n_borrowers: int(n_borrowers + budget - \
                    np.sum(lending_decisions))] = 1

    #2 Expected Score
    payments = np.zeros(n_recommenders)
    expected_payout = np.zeros(n_recommenders)
    ''' #without weights
    for i in range(n_recommenders): #loop over borrowers to calculate payment t and expected score
        p_hat_less_i = np.delete(p_hat, i, 0)
        sums_i = np.sum(p_hat_less_i, axis = 0)
        sums_i.sort()
        threshold = sums_i[-budget]
        sums_i = np.sum(p_hat_less_i, axis = 0)
        lending_decisions_i = np.where(sums_i[:n_borrowers] > threshold, 1, 0) #Will not allocate any reserve borrowers yet
        lending_decisions_i = np.hstack((lending_decisions_i, np.zeros((budget,))))
        if np.sum(lending_decisions_i) < budget: #allocates just enough reserve borrowers to fill the budget quota
            lending_decisions_i[n_borrowers: int(n_borrowers + budget - \
                             np.sum(lending_decisions_i))] = 1
        payments[i] = np.sum(np.multiply(sums_i, lending_decisions_i)) - \
                    np.sum(np.multiply(sums_i, lending_decisions)) #value to everyone else without i present - value to everyone else with i present
        expected_payout[i] = np.sum(np.multiply(p[i,:],\
                    lending_decisions[:n_borrowers])) - payments[i] #This is the expected payout from the recommender's perspective, assuming his/her beliefs are true
    '''
    for i in range(n_recommenders): #loop over borrowers to calculate payment t and expected score
        p_hat_less_i = np.delete(p_hat, i, 0)
        weights_less_i = np.delete(weight_less_i, i)
        sums_i = np.matmul(p_hat_less_i, weights_less_i.T)
        sums_i.sort()
        threshold = sums_i[-budget]
        sums_i = np.matmul(p_hat_less_i, weights_less_i.T)
        lending_decisions_i = np.where(sums_i[:n_borrowers] > threshold, 1, 0) #Will not allocate any reserve borrowers yet
        lending_decisions_i = np.hstack((lending_decisions_i, np.zeros((budget,))))
        if np.sum(lending_decisions_i) < budget: #allocates just enough reserve borrowers to fill the budget quota
            lending_decisions_i[n_borrowers: int(n_borrowers + budget - \
                             np.sum(lending_decisions_i))] = 1
        payments[i] = np.sum(np.multiply(sums_i, lending_decisions_i)) - \
                    np.sum(np.multiply(sums_i, lending_decisions)) #value to everyone else without i present - value to everyone else with i present
        expected_payout[i] = np.sum(np.multiply(p[i,:],\
                    lending_decisions[:n_borrowers])) - payments[i] #This is the expected payout from the recommender's perspective, assuming his/her beliefs are true

    #3 Repayment Outcomes
    repayment_outcomes = np.multiply(np.where(np.random.random(\
                n_borrowers) > (1-repayment_probs), 1, 0),\
                lending_decisions[:n_borrowers])
    actual_payout = np.sum(repayment_outcomes) - payments

    return expected_payout, actual_payout, lending_decisions[:n_borrowers],\
                repayment_outcomes

def generate_weights(reports, multi_round_lending_decisions,
                     multi_round_outcomes, n_recommenders, n_borrowers):
    '''This function generates linear weights using the method by Budescu & Chen.
    reports is an n_recommenders  by rounds_observed * n_borrowers matrix
    multi_round_outcomes is a 1 by n_borrowers * rounds_observed matrix
    multi_round_lending_decisions is a 1 by n_borrowers * rounds_observed matrix
    '''
    '''
    print('reports: ', reports)
    print('multi_round_lending_decisions',multi_round_lending_decisions)
    print('multi_round_outcomes',multi_round_outcomes)
    print('n_recommenders',n_recommenders)
    print('n_borrowers',n_borrowers)
    '''
    mean_recommender_contributions = np.zeros(n_recommenders)
    aggregated_reports = np.sum(reports, axis=0) / n_recommenders #simple average of reports; we could use weighted reports
    group_scores = brier(aggregated_reports, multi_round_outcomes) #includes an outcome of 0 for borrowers who did not receive a loan
    mean_group_score = np.matmul(group_scores, multi_round_lending_decisions.T) \
                / np.sum(multi_round_lending_decisions)
    for i in range(n_recommenders):
        agg_reports_without_recommender = (np.sum(reports, axis=0) - reports[i,:])\
                    / (n_recommenders - 1)
        scores_without_recommender = brier(agg_reports_without_recommender, \
                    multi_round_outcomes)
        mean_recommender_contributions[i] = mean_group_score - \
                    np.matmul(scores_without_recommender, multi_round_lending_decisions.T) \
                / np.sum(multi_round_lending_decisions)
    positive_contributions = np.clip(mean_recommender_contributions, 0, 10**6)
    if np.sum(positive_contributions) > 0: #case where at least one person has made a postiive contribution
        weights = positive_contributions / np.sum(positive_contributions)
    else: #case where no one has made a positive contribution
        weights = np.ones(n_recommenders)/n_recommenders #Use equal weights for all recommenders

    #print('weights',weights)
    return weights

def brier(preds, outcomes):
    #This function takes a set of preds \in [0,1] for binary events and calculates the brier score against actual outcomes \in {0,1}
    preds_of_outcomes = 1 - preds - outcomes + 2*np.multiply(preds, outcomes) #predictions made of the outcomes that actually occurred
    return 2*preds_of_outcomes - np.power(preds,2) - np.power(1-preds,2)



##### CODE TO RUN THE SIMULATION #####

n = 10 #Run the simulation n times

for k in range(n):
    print('Iteration: ',k)
    #1 Parameter Settings
    n_recommenders = 5
    n_borrowers = 20
    budget = 2 #Maximum number of borrowers who can receive a loan
    c = .7 #Lending threshold rating

    #2 Simulation Loops
    if k == 0: n_loans_array, n_repayments_array, mean_recommender_comp_array,\
                rec_comp_vol_array, rec_comp_negative_pct_array = \
                main(n_recommenders, n_borrowers, budget, c)
    else:
        n_loans_made, n_repayments, mean_comp, std_comp, rec_comp_negative_pct\
                    = main(n_recommenders, n_borrowers, budget, c)
        n_loans_array = np.vstack((n_loans_array,n_loans_made))
        n_repayments_array = np.vstack((n_repayments_array,n_repayments))
        mean_recommender_comp_array = np.vstack((mean_recommender_comp_array,mean_comp))
        rec_comp_vol_array = np.vstack((rec_comp_vol_array,std_comp))
        rec_comp_negative_pct_array = np.vstack((rec_comp_negative_pct_array,rec_comp_negative_pct))


print('n_loans',np.round(np.mean(n_loans_array),2))
print('repayment_rate',np.round(np.sum(n_repayments_array) / np.sum(n_loans_array),2))
print('mean_recommender_comp',np.round(np.mean(mean_recommender_comp_array),2))
print('rec_comp_vol',np.round(np.nanmean(rec_comp_vol_array),2))
print('rec_comp_negative_pct',np.round(np.mean(rec_comp_negative_pct_array),2))


'''Technical mean_recommender_contributions
1. shift seems to always give positive values
2. Weights don't seems to add up to 1; they sum to larger than 1
'''
