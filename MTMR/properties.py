"""
Written by Jonghwan Choi at 22 June 2022
https://github.com/mathcom/MTMR
"""
import os
import sys
MTMR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path = sys.path if MTMR_PATH in sys.path else [MTMR_PATH] + sys.path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import rdkit.Chem.QED as QED
import MTMR.drd2_scorer as DRD2
import MTMR.sascorer as sascorer
import networkx as nx
from MTMR.gsk3_scorer import gsk3_model
from MTMR.jnk3_scorer import jnk3_model

from numpy.core.umath_tests import inner1d
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_kekuleSmiles(smi):
    mol = Chem.MolFromSmiles(smi)
    smi_rdkit = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=False,   # modified because this option allows special tokens (e.g. [125I])
                    kekuleSmiles=True,      # modified for downstream analysis with rdkit
                    rootedAtAtom=-1,        # default
                    canonical=True,         # default
                    allBondsExplicit=False, # default
                    allHsExplicit=False     # default
                )
    return smi_rdkit


def rdkit_kekulize_handling(original_fn):
    def wrapper_fn(*args, **kwargs):
        try:
            score = original_fn(*args, **kwargs)
        except Chem.rdchem.KekulizeException:
            score = 0.
        finally:
            return score
    return wrapper_fn


@rdkit_kekulize_handling
def qed(s):
    if s is None:
        return 0.
    else:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return 0.
            else:
                return QED.qed(mol)
        except:
            return 0.


def drd2(s):
    if s is None:
        return 0.0
    elif Chem.MolFromSmiles(s) is None:
        return 0.0
    else:
        return DRD2.get_score(s)


# Modified from https://github.com/bowenliu16/rl_graph_generation
def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    ## logp
    log_p = Descriptors.MolLogP(mol)
    ## synthetic accessiblity
    SA = -sascorer.calculateScore(mol)
    ## cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    cycle_length = max([len(j) for j in cycle_list]) if len(cycle_list) > 0 else 0
    cycle_score = min(0, 6 - cycle_length)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


class gsk3:
    def __init__(self):
        self.model = gsk3_model()

    def __call__(self, s):
        if s is None:
            return 0.0
        elif Chem.MolFromSmiles(s) is None:
            return 0.0
        else:
            return self.model(s)


class jnk3:
    def __init__(self):
        self.model = jnk3_model()

    def __call__(self, s):
        if s is None:
            return 0.0
        elif Chem.MolFromSmiles(s) is None:
            return 0.0
        else:
            return self.model(s)


def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    else:
        fp1 = GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
        return TanimotoSimilarity(fp1, fp2) 




if __name__ == "__main__":
    ex = 'ClC1=CC=C2C(C=C(C(C)=O)C(C(NC3=CC(NC(NC4=CC(C5=C(C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1'
    print(drd2(ex))
    print(qed(ex))
    print(penalized_logp(ex))
    print(gsk3()(ex))
    print(jnk3()(ex))
