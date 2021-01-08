import torch
from torch.nn import functional as F
from torch.nn import Linear, Parameter, Conv1d
import math


class AttentionBase(torch.nn.Module):
    """Abstract attention class.
    
    Arguments:
        representation_dim -- size of the hidden representation
        query_dim -- size of the attention query input (probably decoder hidden state)
        memory_dim -- size of the attention memory input (probably encoder outputs)
    """

    def __init__(self, representation_dim, query_dim, memory_dim):
        super(AttentionBase, self).__init__()
        self._bias = Parameter(torch.zeros(1, representation_dim))
        self._energy = Linear(representation_dim, 1, bias=False)     
        self._query = Linear(query_dim, representation_dim, bias=False)          
        self._memory = Linear(memory_dim, representation_dim, bias=False)
        self._memory_dim = memory_dim
    
    def reset(self, encoded_input, batch_size, max_len, device):
        """Initialize previous attention weights & prepare attention memory."""
        self._memory_transform = self._memory(encoded_input)   
        self._prev_weights = torch.zeros(batch_size, max_len, device=device)
        self._prev_context = torch.zeros(batch_size, self._memory_dim, device=device)
        return self._prev_context

    def _attent(self, query, memory_transform, weights):      
        raise NotImplementedError

    def _combine_weights(self, previsous_weights, weights):      
        raise NotImplementedError

    def _normalize(self, energies, mask):
        raise NotImplementedError

    def forward(self, query, memory, mask, prev_decoder_output):
        energies = self._attent(query, self._memory_transform, self._prev_weights)
        attention_weights = self._normalize(energies, mask)
        self._prev_weights = self._combine_weights(self._prev_weights, attention_weights)
        attention_weights = attention_weights.unsqueeze(1)
        self._prev_context = torch.bmm(attention_weights, memory).squeeze(1)
        return self._prev_context, attention_weights.squeeze(1)


class LocationSensitiveAttention(AttentionBase):
    """
    Location Sensitive Attention:
        Location-sensitive attention: https://arxiv.org/abs/1506.07503.
        Extends additive attention (here https://arxiv.org/abs/1409.0473)  
        to use cumulative attention weights from previous decoder time steps.
        
    Arguments:
        kernel_size -- kernel size of the convolution calculating location features 
        channels -- number of channels of the convolution calculating location features
        smoothing -- to normalize weights using softmax, use False (default) and True to use sigmoids
    """

    def __init__(self, kernel_size, channels, smoothing, representation_dim, query_dim, memory_dim):
        super(LocationSensitiveAttention, self).__init__(representation_dim, query_dim, memory_dim)
        self._location = Linear(channels, representation_dim, bias=False)
        self._loc_features = Conv1d(1, channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self._smoothing = smoothing

    def _attent(self, query, memory_transform, cum_weights):
        query = self._query(query.unsqueeze(1))
        cum_weights = cum_weights.unsqueeze(-1)
        loc_features = self._loc_features(cum_weights.transpose(1,2))
        loc_features = self._location(loc_features.transpose(1,2))
        energy = query + memory_transform + loc_features
        energy = self._energy(torch.tanh(energy + self._bias))
        return energy.squeeze(-1)

    def _normalize(self, energies, mask):
        energies[~mask] = float('-inf')
        if self._smoothing:
            sigmoid = torch.sigmoid(energies)
            total = torch.sum(sigmoid, dim=-1)
            return sigmoid / total
        else:
            return F.softmax(energies, dim=1)

    def _combine_weights(self, previous_weights, weights):      
        return previous_weights + weights


class ForwardAttention(AttentionBase):
    """
    Forward Attention:
        Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis
        without the transition agent: https://arxiv.org/abs/1807.06736.
        However, the attention with convolutional features should have a negative effect 
        on the naturalness of synthetic speech.
    """

    def __init__(self, *args, **kwargs):
        super(ForwardAttention, self).__init__(*args, **kwargs)
        
    def reset(self, encoded_input, batch_size, max_len, device):
        super(ForwardAttention, self).reset(encoded_input, batch_size, max_len, device)
        self._prev_weights[:,0] = 1
        return self._prev_context

    def _prepare_transition(self, query, memory_transform, weights):
        query = self._query(query.unsqueeze(1))
        energy = query + memory_transform
        energy = self._energy(torch.tanh(energy + self._bias)).squeeze(-1)
        energy = F.softmax(energy, dim=1)
        shifted_weights = F.pad(weights, (1, 0))[:, :-1]
        return energy, shifted_weights

    def _attent(self, query, memory_transform, cum_weights):
        energy, shifted_weights = self._prepare_transition(query, memory_transform, self._prev_weights)
        self._prev_weights = (self._prev_weights + shifted_weights) * energy
        return self._prev_weights

    def _normalize(self, energies, mask):
        energies[~mask] = float(0)
        return F.normalize(torch.clamp(energies, 1e-6), p=1)
        
    def _combine_weights(self, previous_weights, weights):      
        return weights


class ForwardAttentionWithTransition(ForwardAttention):
    """
    Forward Attention:
        Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis
        with the transition agent: https://arxiv.org/abs/1807.06736.
    
    Arguments:
        decoder_output_dim -- size of the decoder output (from previous step)
    """

    def __init__(self, decoder_output_dim, representation_dim, query_dim, memory_dim):
        super(ForwardAttentionWithTransition, self).__init__(representation_dim, query_dim, memory_dim)
        self._transition_agent = Linear(memory_dim + query_dim + decoder_output_dim, 1)
        
    def reset(self, encoded_input, batch_size, max_len):
        super(ForwardAttentionWithTransition, self).reset(encoded_input, batch_size, max_len)
        self._t_prob = 0.5
        return self._prev_context

    def _attent(self, query, memory_transform, cum_weights):
        energy, shifted_weights = self._prepare_transition(query, memory_transform, self._prev_weights)
        self._prev_weights = ((1 - self._t_prob) * self._prev_weights + self._t_prob * shifted_weights) * energy
        return self._prev_weights

    def forward(self, query, memory, mask, prev_decoder_output):
        context, weights = super(ForwardAttentionWithTransition, self).forward(query, memory, mask, None)
        transtition_input = torch.cat([self._prev_context, query, prev_decoder_output], dim=1)
        t_prob = self._transition_agent(transtition_input)
        self._t_prob = torch.sigmoid(t_prob)
        return context, weights


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_normal_(self.linear_layer.weight,
                                     gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class SMAAttention(torch.nn.Module):
    def __init__(self, representation_dim, query_dim, memory_dim, sigmoid_noise=2.0,
                 sigmoid_noise_seed=None, score_bias_init=3.5, mode='parallel'):
        '''
        Arguments:
        representation_dim -- size of the hidden representation
        query_dim -- size of the attention query input (probably decoder hidden state)
        memory_dim -- size of the attention memory input (probably encoder outputs)
        '''
        super(SMAAttention, self).__init__()
        self.query_layer = LinearNorm(query_dim, representation_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(memory_dim, representation_dim, bias=False, w_init_gain='tanh')

        v = torch.empty(representation_dim)
        torch.nn.init.uniform_(v)
        self.v = torch.nn.Parameter(v)
        self.register_parameter('attention_v', self.v)

        g = torch.ones(representation_dim) * math.sqrt(1. / representation_dim)
        self.g = torch.nn.Parameter(g)
        self.register_parameter('attention_g', self.g)

        b = torch.zeros(representation_dim)
        self.b = torch.nn.Parameter(b)
        self.register_parameter('attention_b', self.b)

        self.score_mask_value = -float('inf')
        self.sigmoid_noise = sigmoid_noise
        self.sigmoid_noise_seed = sigmoid_noise_seed

        self.mode = mode

    def get_alignment_energies(self, query, processed_memory):
        '''
        query: decoder output [batch_size,attention_rnn_dim]
        processed_memory: processed encoder outputs [batch_size,seq_len,attention_dim]

        RETURNS
        alignment [batch_size,max_time]
        '''
        processed_query = self.query_layer(query)
        normed_v = self.g * self.v * torch.rsqrt(torch.sum(torch.pow(self.v, 2)))
        energies = torch.sum(normed_v * (torch.tanh(processed_query + processed_memory + self.b)), dim=2)
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory, prev_attention, mask):
        '''
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        prev_attention: previous attention weights
        mask: binary mask for padded data
        '''
        prev_attention = prev_attention.float()
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory)
        if self.sigmoid_noise > 0:
            noise = torch.randn(alignment.size(), dtype=alignment.dtype, device=alignment.device)
            alignment += self.sigmoid_noise * noise

        if self.mode == 'hard':
            p_choose_i = torch.gt(alignment, 0).type(alignment.dtype)
        else:
            p_choose_i = torch.sigmoid(alignment)

        if self.mode == 'parallel':
            pad = torch.zeros([p_choose_i.size(0), 1], dtype=p_choose_i.dtype, device=p_choose_i.device)
            attention_weights = prev_attention * p_choose_i + \
                                torch.cat(
                                    (pad, prev_attention[:, :-1] * (1. - p_choose_i[:, :-1])),
                                    1)
        elif self.mode == 'hard':
            move_next_mask = torch.cat(
                (torch.zeros_like(prev_attention[:, :1]), prev_attention[:, :-1]),
                1
            )
            stay_prob = torch.sum(p_choose_i * prev_attention.type(alignment.dtype), 1)
            stay_prob = stay_prob.unsqueeze(1)
            attention_weights = torch.where(stay_prob > 0.5, prev_attention, move_next_mask).float()

        else:
            raise ValueError('mode must be `parallel` or `hard`')

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
