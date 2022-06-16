"""
Written by Jonghwan Choi at 5 Apr 2022
https://github.com/mathcom/MTMR
"""
import numpy as np
import pandas as pd
from MTMR.properties import similarity
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def evaluate_metric(df_generated, smiles_train_high, num_decode=20, threshold_sim=0.4, threshold_pro=0.5):
    metrics = {"VALID_RATIO":0.,
               "AVERAGE_PROPERTY":0.,
               "AVERAGE_SIMILARITY":0.,
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
        ## Metric 4) Novelty
        ###################################
        targets_novel = [(tar,sim,prop) for tar, sim, prop in targets_valid if tar not in smiles_train_high]
        if len(targets_novel) > 0:
            metrics["NOVELTY"] += 1
            
        ###################################
        ## Metric 5) Success
        ###################################
        targets_success = [(tar,sim,prop) for tar,sim,prop in targets_novel if sim > threshold_sim and prop > threshold_pro]
        if len(targets_success) > 0:
            metrics["SUCCESS"] += 1
            
        ###################################
        ## Metric 6) Success without novelty condition
        ###################################
        targets_success_wo_novelty = [(tar,sim,prop) for tar,sim,prop in targets_valid if sim > threshold_sim and prop > threshold_pro]
        if len(targets_success_wo_novelty) > 0:
            metrics["SUCCESS_WO_NOVEL"] += 1
        
    ###################################
    ## Metric 6) Diversity
    ###################################
    targets_unique = [tar for tar in df_generated.iloc[:,1].unique() if tar != 'None']
    metrics["DIVERSITY"] = len(targets_unique)
    
    ###################################
    ## Final average
    ###################################
    metrics["VALID_RATIO"]        /= num_molecules
    metrics["AVERAGE_PROPERTY"]   /= num_molecules
    metrics["AVERAGE_SIMILARITY"] /= num_molecules
    metrics["NOVELTY"]            /= num_molecules
    metrics["SUCCESS"]            /= num_molecules
    metrics["SUCCESS_WO_NOVEL"]   /= num_molecules
    metrics["DIVERSITY"]          /= num_molecules
  
    df_metrics = pd.Series(metrics).to_frame()
    return df_metrics


def evaluate_metric_validation(df_generated, num_decode=20, threshold_sim=0.4, threshold_pro=0.5):
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