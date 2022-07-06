"""
Written by Jonghwan Choi at 06 July 2022
https://github.com/mathcom/MTMR
"""
import tqdm
import numpy as np
import pandas as pd
from MTMR.properties import FastTanimotoOneToBulk
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def evaluate_metric(df_generated, smiles_train_high, num_decode=20, threshold_sim=0.4, threshold_pro=0.0):
    metrics = {"VALID_RATIO":0.,
               "PROPERTY":0.,
               "IMPROVEMENT":0.,
               "SIMILARITY":0.,
               "NOVELTY":0.,
               "SUCCESS":0.,
               "SUCCESS_WO_NOVEL":0.,
               "DIVERSITY":0.}
    
    num_molecules = len(df_generated) // num_decode
    assert len(df_generated) % num_decode == 0
    
    for i in range(0, len(df_generated), num_decode):
        sources = set([x for x in df_generated.iloc[i:i+num_decode, 0]])
        assert len(sources) == 1
        
        ###################################
        ## Metric 1) Validity
        ###################################
        targets_valid = []
        for src,tar,sim,prop_tar,prop_src in df_generated.iloc[i:i+num_decode,:].values:
            if 1 > sim > 0 and prop_tar > 0:
                targets_valid.append((tar, sim, prop_tar, prop_src))
                
        if len(targets_valid) > 0:
            metrics["VALID_RATIO"] += 1
            
        ###################################
        ## Metric 2) Property
        ###################################
        targets_valid_prop = [prop for _, _, prop, _ in targets_valid]
        if len(targets_valid_prop) > 0:
            metrics["PROPERTY"] += np.mean(targets_valid_prop)
    
        ###################################
        ## Metric 2) Improvement
        ###################################
        targets_valid_impr = [prop_tar - prop_src for _, _, prop_tar, prop_src in targets_valid]
        if len(targets_valid_impr) > 0:
            metrics["IMPROVEMENT"] += np.mean(targets_valid_impr)
    
        ###################################
        ## Metric 3) Similarity
        ###################################
        targets_valid_sim = [sim for _, sim, _, _ in targets_valid]
        if len(targets_valid_sim) > 0:
            metrics["SIMILARITY"] += np.mean(targets_valid_sim)
    
        ###################################
        ## Metric 4) Novelty
        ###################################
        targets_novel = [(tar,sim,prop_tar,prop_src) for tar, sim, prop_tar, prop_src in targets_valid if tar not in smiles_train_high]
        if len(targets_novel) > 0:
            metrics["NOVELTY"] += 1
            
        ###################################
        ## Metric 5) Success
        ###################################
        targets_success = [(tar,sim,prop_tar,prop_src) for tar,sim,prop_tar,prop_src in targets_novel if sim >= threshold_sim and prop_tar - prop_src >= threshold_pro]
        if len(targets_success) > 0:
            metrics["SUCCESS"] += 1
            
        ###################################
        ## Metric 6) Success without novelty condition
        ###################################
        targets_success_wo_novelty = [(tar,sim,prop_tar,prop_src) for tar,sim,prop_tar,prop_src in targets_valid if sim >= threshold_sim and prop_tar - prop_src >= threshold_pro]
        if len(targets_success_wo_novelty) > 0:
            metrics["SUCCESS_WO_NOVEL"] += 1
            
        ###################################
        ## Metric 6) Diversity
        ###################################
        if len(targets_valid) > 1:
            calc_bulk_sim = FastTanimotoOneToBulk([x[0] for x in targets_valid])
            targets_diversity = []            
            for j in range(len(targets_valid)):
                div = calc_bulk_sim(targets_valid[j][0])
                targets_diversity += div[:j-1].tolist() + div[j+1:].tolist()
            metrics["DIVERSITY"] += np.mean(targets_diversity)
    
    ###################################
    ## Final average
    ###################################
    metrics["VALID_RATIO"]         /= num_molecules
    metrics["PROPERTY"]            /= num_molecules
    metrics["IMPROVEMENT"]         /= num_molecules
    metrics["SIMILARITY"]          /= num_molecules
    metrics["NOVELTY"]             /= num_molecules
    metrics["SUCCESS"]             /= num_molecules
    metrics["SUCCESS_WO_NOVEL"]    /= num_molecules
    metrics["DIVERSITY"]           /= num_molecules
  
    df_metrics = pd.Series(metrics).to_frame()
    return df_metrics


def evaluate_metric_validation(df_generated, num_decode=20):
    metrics = {"VALID_RATIO":0.,
               "AVERAGE_PROPERTY":0.,
               "AVERAGE_SIMILARITY":0.}               
    
    num_molecules = len(df_generated) // num_decode
    assert len(df_generated) % num_decode == 0
    
    for i in range(0, len(df_generated), num_decode):
        sources = set([x for x in df_generated.iloc[i:i+num_decode, 0]])
        assert len(sources) == 1
        
        ###################################
        ## Metric 1) Validity
        ###################################
        targets_valid = [(tar,sim,prop) for _,tar,sim,prop in df_generated.iloc[i:i+num_decode,:].values if 1 > sim > 0 and prop > 0]
        if len(targets_valid) > 0:
            metrics["VALID_RATIO"] += 1
            
        ###################################
        ## Metric 2) Property
        ###################################
        targets_valid_prop = [prop for _, _, prop in targets_valid]
        if len(targets_valid_prop) > 0:
            metrics["AVERAGE_PROPERTY"] += np.mean(targets_valid_prop)
    
        ###################################
        ## Metric 3) Similarity
        ###################################
        targets_valid_sim = [sim for _, sim, _ in targets_valid]
        if len(targets_valid_sim) > 0:
            metrics["AVERAGE_SIMILARITY"] += np.mean(targets_valid_sim)
            
    ###################################
    ## Final average
    ###################################
    metrics["VALID_RATIO"]        /= num_molecules
    metrics["AVERAGE_PROPERTY"]   /= num_molecules
    metrics["AVERAGE_SIMILARITY"] /= num_molecules
  
    df_metrics = pd.Series(metrics).to_frame()
    return df_metrics
