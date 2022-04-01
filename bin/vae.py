"""
Written by Jonghwan Choi at 1 Apr 2022
https://github.com/mathcom/MTMR
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset


class ReplayBufferDataset(Dataset):
    def __init__(self):
        super(ReplayBufferDataset, self).__init__()
        self.smiles_src_list = []
        self.smiles_tar_list = []
        self.reward_list = []
        self.similarity_list = []
        self.property_list = []
        self.pop_list = []
        
    def push(self, smiles_src, smiles_tar, reward, similarity, property):
        self.smiles_src_list.append(smiles_src)
        self.smiles_tar_list.append(smiles_tar)
        self.reward_list.append(reward)
        self.similarity_list.append(similarity)
        self.property_list.append(property)
        
    def stats(self):
        return np.mean(self.reward_list), np.mean(self.similarity_list), np.mean(self.property_list)
        
    def pop(self):
        for idx in list(reversed(sorted(self.pop_list))):
            _ = self.smiles_src_list.pop(idx)
            _ = self.smiles_tar_list.pop(idx)
            _ = self.reward_list.pop(idx)
            _ = self.similarity_list.pop(idx)
            _ = self.property_list.pop(idx)
        self.pop_list = []
        
    def __len__(self):
        return len(self.smiles_src_list)
        
    def __getitem__(self, idx):
        self.pop_list.append(idx)
        smiles_src = self.smiles_src_list[idx]
        length_src = len(smiles_src)
        smiles_tar = self.smiles_tar_list[idx]
        length_tar = len(smiles_tar)
        reward = self.reward_list[idx]
        similarity = self.similarity_list[idx]
        property = self.property_list[idx]
        return {"smiles_src": smiles_src,
                "length_src": length_src,
                "smiles_tar": smiles_tar,
                "length_tar": length_tar,
                "reward": reward,
                "similarity": similarity,
                "property": property}


class CyclicalAnnealingScheduler(object):
    def __init__(self, T, M=4):
        super(CyclicalAnnealingScheduler, self).__init__()
        '''
        Params
        ------
        T : the total number of training iterations
        M : number of cycles
        R : proportin used to increase beta within a cycle
        R_inv : the inverse of R
        '''
        self.T = T
        self.M = M
        if M != 1:
            self.normalizer = T/M
            self.modulo = math.ceil(self.normalizer)
        else:
            self.normalizer = T
            self.modulo = T
        
    def __call__(self, step):
        if step < self.T:
            tau = (step % self.modulo) / self.normalizer # 0 <= tau < 1
            beta = max(min(self._monotonically_increasing_ft(tau), 1.), 0.)
        else:
            beta = 1.
        return beta
        
    def _monotonically_increasing_ft(self, x):
        return 2. * x


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, pad_idx, num_layers, dropout, device=None):
        super(SmilesEncoder, self).__init__()
        ## params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cpu') if device is None else device
        
        ## special tokens
        self.pad_idx = pad_idx
        
        ## Neural Network
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, padding_idx=self.pad_idx)
        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2mean = nn.Linear(self.num_layers * 2 * self.hidden_size, self.latent_size)
        self.hidden2logvar = nn.Linear(self.num_layers * 2 * self.hidden_size, self.latent_size)

    def forward(self, inps, lens):
        '''
        Params
        ------
        inps.shape = (batch, maxseqlen)
        lens.shape = (batch,)
        '''
        batch_size = inps.size(0)
        
        ## Sorting by seqlen
        sorted_seqlen, sorted_idx = torch.sort(lens, descending=True) # sorted_seqlen.shape = (batch,), sorted_idx.shape = (batch,)
        sorted_inps = inps[sorted_idx] # sorted_inps.shape = (batch, maxseqlen)
        
        ## Packing for encoder
        inps_emb = self.embedding(sorted_inps) # inps_emb.shape = (batch, maxseqlen, hidden)
        packed_inps = rnn_utils.pack_padded_sequence(inps_emb, sorted_seqlen.data.tolist(), batch_first=True)
        
        ## RNN
        _, sorted_hiddens = self.rnn(packed_inps) # sorted_hiddens.shape = (numlayer * 2, batch, hidden)
        sorted_hiddens = sorted_hiddens.transpose(0,1).contiguous() # sorted_hiddens.shape = (batch, numlayer * 2, hidden)
        sorted_hiddens = sorted_hiddens.view(batch_size, -1) # sorted_hiddens.shape = (batch, numlayer * 2 * hidden)
        
        ## Latent vector
        sorted_mean = self.hidden2mean(sorted_hiddens) # sorted_mean.shape = (batch, latent)
        sorted_logvar = self.hidden2logvar(sorted_hiddens) # sorted_logvar.shape = (batch, latent)
        
        ## Reordering
        mean, logvar = self.reordering(sorted_mean, sorted_logvar, sorted_idx)
        return mean, logvar
        
        
    def sampling(self, mean, logvar):
        '''
        Params
        ------
        mean.shape = (batch, latent)
        logvar.shape = (batch, latent)
        '''
        batch_size = mean.size(0)
        std = torch.exp(0.5 * logvar) # std.shape = (batch, latent)
        epsilon = torch.randn([batch_size, self.latent_size], device=self.device) # epsilon.shape = (batch, latent)
        z = epsilon * std + mean # z.shape = (batch, latent)
        return z
        
        
    def reordering(self, sorted_mean, sorted_logvar, sorted_idx):
        '''
        Params
        ------
        sorted_mean.shape = (batch, latent)
        sorted_logvar.shape = (batch, latent)
        sorted_idx = (batch, )
        '''
        _, original_idx = torch.sort(sorted_idx, descending=False) # original_idx.shape = (batch, )
        mean = sorted_mean[original_idx] # mean.shape = (batch, latent)
        logvar = sorted_logvar[original_idx] # logvar.shape = (batch, latent)
        return mean, logvar
        
        
class SmilesDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, pad_idx, num_layers, dropout, device=None):
        super(SmilesDecoder, self).__init__()
        ## params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cpu') if device is None else device
        
        ## special tokens
        self.pad_idx = pad_idx
        
        ## Neural Network
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, padding_idx=self.pad_idx)
        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.dense = nn.Sequential(nn.Linear(self.hidden_size + self.latent_size, self.hidden_size), nn.ReLU())
        self.output2vocab = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, inp, z, hidden=None):
        '''
        Params
        ------
        inp.shape = (batch, 1)
        z.shape = (batch, latent)
        hidden.shape = (numlayer, batch, hidden)
        '''
        if hidden is None:
            return self._forward(inp, z)
        else:
            return self._forward_single(inp, hidden, z)
        
        
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device) # hidden.shape = (numlayer, batch, hidden)
        return hidden
        
    
    def _forward(self, inp, z):
        '''
        Params
        ------
        inp.shape = (batch, seq)
        z.shape = (batch, latent)
        '''
        batch_size = inp.size(0)
        seqlen = inp.size(1)
        
        ## Embedding
        inp_emb = self.embedding(inp) # inp_emb.shape = (batch, seq, hidden)
        
        ## Condition
        cond = z.unsqueeze(1) # cond.shape = (batch, 1, latent)
        cond = cond.repeat(1,seqlen,1) # cond.shape = (batch, seq, latent)
        inp_cond = torch.cat((inp_emb, cond), 2) # inp_cond.shape = (batch, seq, hidden + latent)
        inp_cond = self.dense(inp_cond) # inp_cond.shape = (batch, seq, hidden)
        
        ## Decoder - Teacher forcing
        out, _ = self.rnn(inp_cond) # out.shape = (batch, seq, hidden)
        
        ## Prediction
        logits = self.output2vocab(out) # logits.shape = (batch, seq, vocab)
        return logits
        
    
    def _forward_single(self, inp, hidden, z):
        '''
        Params
        ------
        inp.shape = (batch, 1)
        hidden.shape = (numlayer, batch, hidden)
        z.shape = (batch, latent)
        '''
        batch_size = inp.size(0)
        
        ## Embedding
        inp_emb = self.embedding(inp) # inp_emb.shape = (batch, 1, hidden)
        
        ## Condition
        inp = torch.cat((inp_emb, z.unsqueeze(1)), 2) # inp.shape = (batch, 1, hidden + latent)
        inp = self.dense(inp) # inp.shape = (batch, 1, hidden)
        
        ## Decoder - Teacher forcing
        out, hidden = self.rnn(inp, hidden) # out.shape = (batch, 1, hidden), hidden.shape = (numlayer, batch, hidden)
        
        ## Prediction
        logits = self.output2vocab(out) # logits.shape = (batch, 1, vocab)
        return logits, hidden
        

class SmilesAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, sos_idx, eos_idx, pad_idx, num_layers=2, dropout=0., delta=4., device=None, filepath_config=None):
        super(SmilesAutoencoder, self).__init__()
        
        ## params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.delta = delta
        self.device = torch.device('cpu') if device is None else device
        self.filepath_config = filepath_config

        ## special tokens
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        ## is there predefined configurations?
        if self.filepath_config is not None:
            self.load_config(self.filepath_config)
            
        ## model build
        self.encoder = SmilesEncoder(self.vocab_size, self.hidden_size, self.latent_size, self.pad_idx, self.num_layers, self.dropout, self.device)
        self.decoder = SmilesDecoder(self.vocab_size, self.hidden_size, self.latent_size, self.pad_idx, self.num_layers, self.dropout, self.device)

        ## device
        self.to(self.device)

    
    def policy_gradient(self, smiles_A, length_A, smiles_B, length_B, rewards, lr=1e-4, gamma=0.995):
        '''
        Params
        ------
        smiles_A.shape = (batch, seq)
        length_A.shape = (batch, )
        smiles_B.shape = (batch, seq)
        length_B.shape = (batch, )
        rewards.shape = (batch, )
        '''
        batch_size = smiles_B.size(0)
        seqlen = smiles_B.size(1)
        
        ## Training phase
        self.encoder.eval()
        self.decoder.train()
        
        ## Optimizer
        optim_decoder = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        optim_decoder.zero_grad()
        
        ## Encoder
        with torch.no_grad():
            mean_A, logvar_A = self.encoder(smiles_A, length_A) # mean_A.shape = (batch, latent), logvar_A.shape = (batch, latent)
        
        ## Sampling
        z_A = self.encoder.sampling(mean_A, logvar_A) # z_A.shape = (batch, latent)
        
        ## Decode
        logits_B = self.decoder(smiles_B, z_A) # logits_B.shape = (batch, seq, vocab)
        logp_B = torch.nn.functional.log_softmax(logits_B, dim=-1) # logp.shape = (batch, seq, vocab)
        
        ## Returns (= cummulative rewards = discounted rewards)
        G_B = torch.zeros(batch_size, seqlen, device=self.device) # G_B.shape = (batch, seq)
        G_B[torch.arange(batch_size), length_B-1] = rewards
        for t in range(1, seqlen):
            G_B[:,-t-1] = G_B[:,-t-1] + G_B[:,-t] * gamma
        
        ## Loss
        glogp_B = G_B.unsqueeze(-1) * logp_B # glogp_B.shape = (batch, seq, vocab)
        target_ravel = smiles_B[:,1:].contiguous().view(-1) # target_ravel.shape = (batch*(seq-1), )
        glogp_ravel = glogp_B[:,:-1,:].contiguous().view(-1, glogp_B.size(-1)) # logp_ravel.shape = (batch*(seq-1), vocab)
        rl_loss = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(glogp_ravel, target_ravel)
        
        ## Backpropagation
        rl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.) # gradient clipping
        optim_decoder.step()
        
        return rl_loss.item()
    
    
    def partial_fit(self, smiles_A, length_A, smiles_B, length_B, smiles_C, length_C, lr=1e-3, beta=1., gamma=None):
        '''
        Params
        ------
        smiles.shape = (batch, seq)
        length.shape = (batch, )
        '''
        batch_size = smiles_A.size(0)
        assert batch_size == smiles_B.size(0)
        if gamma is None: gamma = beta
        
        ## Training phase
        self.train()
        
        ## Optimizer
        optim_encoder = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        optim_decoder = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        optim_encoder.zero_grad()
        optim_decoder.zero_grad()
        
        ## Encoder
        mean_A, logvar_A = self.encoder(smiles_A, length_A) # mean_A.shape = (batch, latent), logvar_A.shape = (batch, latent)
        mean_B, logvar_B = self.encoder(smiles_B, length_B)
        mean_C, logvar_C = self.encoder(smiles_C, length_C)

        ## Sampling
        z_A = self.encoder.sampling(mean_A, logvar_A) # z_A.shape = (batch, latent)
        z_B = self.encoder.sampling(mean_B, logvar_B)
        z_C = self.encoder.sampling(mean_C, logvar_C)

        ## Decoder
        logits_A = self.decoder(smiles_A, z_A) # logits_A.shape = (batch, seq, vocab)
        logits_B = self.decoder(smiles_B, z_B) # logits_B.shape = (batch, seq, vocab)
        logits_C = self.decoder(smiles_C, z_C) # logits_C.shape = (batch, seq, vocab)

        ## Reconstruction loss
        loss_recon_A = self._calc_reconstruction_loss(logits_A, smiles_A)
        loss_recon_B = self._calc_reconstruction_loss(logits_B, smiles_B)
        loss_recon_C = self._calc_reconstruction_loss(logits_C, smiles_C)
        
        ## Contraction loss
        loss_contraction = self._calc_frechet_distance(mean_A, logvar_A, mean_B, logvar_B)
        
        ## Relaxation loss
        loss_relaxation = 0.
        loss_relaxation = loss_relaxation + self._calc_relaxation_loss(mean_A, mean_C)
        loss_relaxation = loss_relaxation + self._calc_relaxation_loss(mean_B, mean_C)

        ## Total loss
        loss = 0.
        loss = loss + 0.3 * loss_recon_A
        loss = loss + 0.3 * loss_recon_B
        loss = loss + 0.4 * loss_recon_C
        loss = loss + beta * loss_contraction
        loss = loss + gamma * loss_relaxation
        
        ## Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.) # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.) # gradient clipping
        optim_encoder.step()
        optim_decoder.step()
        
        return loss.item(), loss_recon_A.item(), loss_recon_B.item(), loss_recon_C.item(), loss_contraction.item(), loss_relaxation.item()


    def _calc_reconstruction_loss(self, logit, target):
        '''
        Params
        ------
        target.shape = (batch, seq)
        logit.shape = (batch, seq, vocab)
        '''
        logp = torch.nn.functional.log_softmax(logit, dim=-1) # logp.shape = (batch, seq, vocab)
        target_ravel = target[:,1:].contiguous().view(-1) # target_ravel.shape = (batch*(seq-1), )
        logp_ravel = logp[:,:-1,:].contiguous().view(-1, logp.size(2)) # logp_ravel.shape = (batch*(seq-1), vocab)
        loss = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(logp_ravel, target_ravel)
        return loss


    def _calc_relaxation_loss(self, mean_A, mean_B):
        '''
        Params
        ------
        mean_A.shape = (batch, latent)
        mean_B.shape = (batch, latent)
        '''
        dist = (mean_A - mean_B).pow(2).sum(1) # dist.shape = (batch, )
        loss = torch.nn.functional.softplus(self.delta - dist) # loss.shape = (batch, )
        loss = loss.mean()
        return loss


    def _calc_frechet_distance(self, mean_A, logvar_A, mean_B, logvar_B):
        '''
        Frechet distance which is also called wasserstein-2 distance
        
        Params
        ------
        mean_A.shape = (batch, latent)
        logvar_A.shape = (batch, latent)
        mean_B.shape = (batch, latent)
        logvar_B.shape = (batch, latent)
        '''
        loss = (mean_A - mean_B).pow(2)
        loss = loss + torch.exp(logvar_A) + torch.exp(logvar_B)
        loss = loss - 2. * torch.exp(0.5 * (logvar_A + logvar_B))
        loss = loss.sum(1).mean()
        return loss
    
    
    def transform(self, smiles, length):
        mean, logvar = self.encoder(smiles, length)
        return mean.cpu().detach().numpy(), logvar.cpu().detach().numpy()
    
    
    def predict(self, smiles, length, max_seqlen=128):
        '''
        Params
        ------
        smiles.shape = (batch, seq)
        length.shape = (batch, )
        '''
        ## evaluate phase
        self.eval()
        ## Params
        batch_size = smiles.size(0)
        ## Generation
        with torch.no_grad():
            ## Encoder
            mean, logvar = self.encoder(smiles, length) # mean.shape = (batch, latent), logvar.shape = (batch, latent)
            ## Sampling
            z = self.encoder.sampling(mean, logvar) # z.shape = (batch, latent)
            ## Decoder
            generated = []
            for i in range(batch_size):
                z_i = z[i].unsqueeze(0) # z_i.shape = (1, latent)
                seq = self._generate(z_i, max_seqlen) # seq.shape = (1, max_seqlen)
                seq = seq.cpu().numpy()
                generated.append(seq)
        generated = np.concatenate(generated) # generated.shape = (batch, max_seqlen)
        return generated
            
            
    def _generate(self, z, max_seqlen, greedy=False):
        '''
        Params
        ------
        z.shape = (1, latent)
        '''
        batch_size = z.size(0)
        
        ## Initialize outs
        outs = torch.full(size=(batch_size, max_seqlen), fill_value=self.pad_idx, dtype=torch.long, device=self.device) # outs.shape = (batch, max_seqlen)
        
        ## Initial hidden
        hiddens = self.decoder.init_hidden(batch_size) # hiddens.shape = (numlayer, batch, hidden)
        
        ## Start token
        inps_sos = torch.full(size=(batch_size, 1), fill_value=self.sos_idx, dtype=torch.long, device=self.device) # inps_sos.shape = (batch, 1)
        
        ## Recursive
        inps = inps_sos # inps.shape = (batch, 1)
        for i in range(max_seqlen):
            ## Terminal condition
            if inps[0][0] == self.eos_idx or inps[0][0] == self.pad_idx:
                outs[:,i] = self.eos_idx
                break
            else:
                outs[:,i] = inps
                
            ## Decode
            logits, hiddens = self.decoder(inps, z, hiddens) # logits.shape = (batch, 1, vocab), hiddens.shape = (numlayer, batch, hidden)

            ## Next word
            if greedy:
                _, top_idx = torch.topk(logits, 1, dim=-1) # top_idx.shape = (batch, 1, 1)
                inps = top_idx.contiguous().view(batch_size, 1) # inps.shape = (batch, 1)
            else:
                probs = torch.softmax(logits, dim=-1) # probs.shape = (batch, 1, vocab)
                probs = probs.view(probs.size(0), probs.size(2)) # probs.shape = (batch, vocab)
                inps = torch.multinomial(probs, 1) # inps.shape = (batch, 1)
        return outs


    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    
    def save_config(self, path):
        with open(path, 'w') as fout:
            fout.write(f"VOCAB_SIZE,{self.vocab_size}\n")
            fout.write(f"HIDDEN_SIZE,{self.hidden_size}\n")
            fout.write(f"LATENT_SIZE,{self.latent_size}\n")
            fout.write(f"NUM_LAYERS,{self.num_layers}\n")
            fout.write(f"DROPOUT,{self.dropout}\n")
            fout.write(f"DELTA,{self.delta}\n")
            fout.write(f"SOS_IDX,{self.sos_idx}\n")
            fout.write(f"EOS_IDX,{self.eos_idx}\n")
            fout.write(f"PAD_IDX,{self.pad_idx}\n")
            
            
    def load_config(self, path):
        with open(path) as fin:
            lines = fin.readlines()
        lines = [l.rstrip().split(",") for l in lines]
        ## parsing
        params = dict()
        for k, v in lines:
            params[k] = v
        ## update configs
        self.vocab_size = int(params["VOCAB_SIZE"])
        self.hidden_size = int(params["HIDDEN_SIZE"])
        self.latent_size = int(params["LATENT_SIZE"])
        self.num_layers = int(params["NUM_LAYERS"])
        self.dropout = float(params["DROPOUT"])
        self.delta = float(params["DELTA"])
        self.sos_idx = int(params["SOS_IDX"])
        self.eos_idx = int(params["EOS_IDX"])
        self.pad_idx = int(params["PAD_IDX"])
        