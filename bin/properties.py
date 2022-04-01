"""
Written by Jonghwan Choi at 1 Apr 2022
https://github.com/mathcom/MTMR
"""
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import rdkit.Chem.QED as QED
import drd2_scorer


def rdkit_kekulize_handling(original_fn):
    def wrapper_fn(*args, **kwargs):
        try:
            score = original_fn(*args, **kwargs)
        except rdkit.Chem.rdchem.KekulizeException:
            score = 0.
        finally:
            return score
    return wrapper_fn


@rdkit_kekulize_handling
def qed(s):
    if s is None:
        return 0.
    else:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return 0.
        else:
            return QED.qed(mol)


def drd2(s):
    if s is None:
        return 0.0
    elif Chem.MolFromSmiles(s) is None:
        return 0.0
    else:
        return drd2_scorer.get_score(s)


def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    else:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
        return DataStructs.TanimotoSimilarity(fp1, fp2) 




if __name__ == "__main__":
    ex = 'ClC1=CC=C2C(C=C(C(C)=O)C(C(NC3=CC(NC(NC4=CC(C5=C(C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1'
    print(drd2(ex))
    print(qed(ex))
