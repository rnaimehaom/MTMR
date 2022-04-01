import sys
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.append("./bin")
from properties import similarity


def evaluate_metric(df_generated, smiles_train_high, score_fuction, threshold_sim=0.3, threshold_pro=0.8, lower_is_good=False, score_function_2=None):
    metrics = {"VALID_RATIO":0.,
               "AVERAGE_PROPERTY":0.,
               "AVERAGE_SIMILARITY":0.,
               "NOVELTY":0.,
               "SUCCESS":0.,
               "DIVERSITY":0.}               
    
    ###################################
    ## Metric 1) Validity
    ###################################
    valid_generated = []
    for src, tgt in zip(df_generated.iloc[:,0], df_generated.iloc[:,1]):
        if tgt != "":
            valid_generated.append((src, tgt))
    valid_ratio = len(valid_generated) / len(df_generated)
    metrics["VALID_RATIO"] = valid_ratio
    
    ###################################
    ## Metric 2) Property
    ###################################
    avg_prop = 0.
    for src, tgt in valid_generated:
        avg_prop += score_fuction(tgt)
    avg_prop /= len(valid_generated)
    metrics["AVERAGE_PROPERTY"] = avg_prop
    
    ###################################
    ## Metric 3) Similarity
    ###################################
    avg_sim = 0.
    for src, tgt in valid_generated:
        avg_sim += similarity(src, tgt)
    avg_sim /= len(valid_generated)
    metrics["AVERAGE_SIMILARITY"] = avg_sim
    
    ###################################
    ## Metric 4) Novelty
    ###################################
    novelty = 0
    for src, tgt in valid_generated:
        if (src != tgt) and (tgt not in smiles_train_high):
            novelty += 1
    novelty /= len(valid_generated)
    metrics["NOVELTY"] = novelty
    
    ###################################
    ## Metric 5) Success
    ###################################
    success = 0
    if lower_is_good:
        for src, tgt in valid_generated:
            if (similarity(src, tgt) > threshold_sim) and (score_fuction(tgt) < threshold_pro) and (tgt not in smiles_train_high):
                success += 1
    else:
        for src, tgt in valid_generated:
            if (similarity(src, tgt) > threshold_sim) and (score_fuction(tgt) > threshold_pro) and (tgt not in smiles_train_high):
                success += 1
    success /= len(valid_generated)
    metrics["SUCCESS"] = success
    
    ###################################
    ## Metric 6) Diversity
    ###################################
    unique_generated = set([tgt for src, tgt in valid_generated])
    diversity = len(unique_generated) / len(valid_generated)
    metrics["DIVERSITY"] = diversity
    
    ###################################
    ## Metric 2) Property
    ###################################
    if score_function_2 is not None:
        avg_prop_2 = 0.
        for src, tgt in valid_generated:
            avg_prop_2 += score_function_2(tgt)
        avg_prop_2 /= len(valid_generated)
        metrics["AVERAGE_PROPERTY_2"] = avg_prop_2
    
    df_metrics = pd.Series(metrics).to_frame()
    return df_metrics
    

def run_on_dataset(generator, dataset, use_cuda, use_tqdm=True):
    if use_tqdm: enum = tqdm.tqdm(DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=use_cuda), total=len(dataset))
    else: enum = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=use_cuda)
    
    generated = []
    for batch in enum:
        batch_smiles = dataset.encode(batch["smiles"], batch["length"].max())
        batch_length = batch["length"]
        ## predict
        is_generated = False
        for k in range(20):
            seq = generator.predict(batch_smiles, batch_length)
            smi = dataset.decode(seq)[0] # note: batch_size = 1
            if MolFromSmiles(smi):
                generated.append((batch["smiles"][0][1:-1], smi))
                is_generated = True
                break
        if not is_generated:
            generated.append((batch["smiles"][0][1:-1],""))
    print(f"len(generated): {len(generated)}")    
    df_generated = pd.DataFrame.from_records(generated)
    return df_generated
