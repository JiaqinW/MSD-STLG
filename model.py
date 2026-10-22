import torch
from torch import nn
from torch.nn import functional as F
from util import INPUT_KEYS


class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = nn.LSTM(args.input_dim, args.hidden_dim, args.num_layers, batch_first=True)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Encoder input must be [batch, time, nodes, features].')
        batch, steps, nodes, features = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch*nodes, steps, features)
        _, (hidden, _) = self.lstm(sequence)
        return F.leaky_relu(hidden[-1], 0.1).reshape(batch, nodes, -1)


class SimpleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(1, args.degree_dim)

    def forward(self, degree):
        if degree.ndim != 2 or degree.shape[-1] != 1:
            raise ValueError('Node degrees must have shape [nodes, 1].')
        return F.leaky_relu(self.fc(degree), 0.1)


class NCSMGAT(nn.Module):
    def __init__(self, input_dim, out_channels, heads):
        super().__init__()
        self.heads, self.channels = heads, out_channels
        self.projection = nn.Linear(input_dim, heads*out_channels, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, out_channels))
        self.attention_target = nn.Parameter(torch.empty(heads, out_channels))
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, x, edge_index, edge_weight, return_attention=False):
        batch, nodes, _ = x.shape
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError('edge_index must be a long tensor of shape [2, edges].')
        source, target = edge_index
        weight = edge_weight.reshape(-1)
        if weight.numel() != source.numel() or not weight.numel():
            raise ValueError('Edge indices and weights must be nonempty and aligned.')
        if edge_index.min() < 0 or edge_index.max() >= nodes:
            raise ValueError('Edge indices fall outside the node dimension.')
        if not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError('NCSM weights must be finite and positive on every edge.')
        if torch.bincount(target, minlength=nodes).min() == 0:
            raise ValueError('Each target needs an incoming edge, including its self-loop.')
        h = self.projection(x).reshape(batch, nodes, self.heads, self.channels)
        score_source = (h*self.attention_source).sum(-1)
        score_target = (h*self.attention_target).sum(-1)
        logits = F.leaky_relu(score_source[:, source] + score_target[:, target], 0.2)
        logits = logits + weight.log()[None, :, None]
        index = target[None, :, None].expand(batch, -1, self.heads)
        maxima = x.new_full((batch, nodes, self.heads), -torch.inf)
        maxima.scatter_reduce_(1, index, logits.detach(), reduce='amax', include_self=True)
        unnormalized = (logits-maxima.gather(1, index)).exp()
        denominator = x.new_zeros((batch, nodes, self.heads)).scatter_add_(1, index, unnormalized)
        attention = unnormalized/denominator.gather(1, index)
        messages = h[:, source]*attention[..., None]
        result = x.new_zeros((batch, nodes, self.heads, self.channels))
        result.scatter_add_(1, index[..., None].expand_as(messages), messages)
        result = result.reshape(batch, nodes, -1)
        return (result, attention) if return_attention else result


class LSTMDecoderWithLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.horizon = args.out_length
        if self.horizon < 1:
            raise ValueError('The forecast horizon must be positive.')
        self.gat = NCSMGAT(args.hidden_dim*4 + args.degree_dim, args.out_channels,
                          args.num_heads)
        self.lstm = nn.LSTM(args.out_channels*args.num_heads, args.decoderout,
                            args.decoder_num_layers, batch_first=True)
        self.linear = nn.Linear(args.decoderout, 1)

    def forward(self, x, edge_index, edge_attr):
        context = F.leaky_relu(self.gat(x, edge_index, edge_attr), 0.1)
        batch, nodes, features = context.shape
        sequence = context.reshape(batch*nodes, 1, features).expand(-1, self.horizon, -1).contiguous()
        state_shape = (self.lstm.num_layers, batch*nodes, self.lstm.hidden_size)
        decoded, _ = self.lstm(sequence, (context.new_zeros(state_shape),
                                        context.new_zeros(state_shape)))
        output = self.linear(decoded)
        return output.reshape(batch, nodes, self.horizon, 1).permute(0, 2, 1, 3).contiguous()


class MSDSTLG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoders = nn.ModuleDict({key: LSTMEncoder(args) for key in INPUT_KEYS})
        self.degree_encoder = SimpleModel(args)
        self.decoder = LSTMDecoderWithLinear(args)

    def forward(self, inputs, degrees, edge_index, edge_attr):
        temporal = [self.encoders[key](inputs[key]) for key in INPUT_KEYS]
        degree = self.degree_encoder(degrees)[None].expand(temporal[0].shape[0], -1, -1)
        return self.decoder(torch.cat((*temporal, degree), dim=-1), edge_index, edge_attr)
