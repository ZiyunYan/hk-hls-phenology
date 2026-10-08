import torch
import torch.nn as nn
from layers.Embed import TemporalEmbedding, TimeFeatureEmbedding

class PositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: int, n_positions: int = 1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (torch.arange(0, d_hid, 2).float() * -(torch.log(torch.tensor(10000)) / d_hid)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pos_table", pe)

    def forward(self, x: torch.Tensor, return_only_pos: bool = False) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            x_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pos_table[:, : x.size(1)].clone().detach()

        if return_only_pos:
            return pos_enc

        x_with_pos = x + pos_enc
        return x_with_pos



class Embedding(nn.Module):
    """The embedding method from the SAITS paper :cite:`du2023SAITS`.

    Parameters
    ----------
    d_in :
        The input dimension.

    d_out :
        The output dimension.

    with_pos :
        Whether to add positional encoding.

    n_max_steps :
        The maximum number of steps.
        It only works when ``with_pos`` is True.

    dropout :
        The dropout rate.

    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        with_pos: bool,
        embed_type: str = 'fixed',
        freq: str = 'd',
        n_max_steps: int = 1000,
        dropout: float = 0,

    ):
        super().__init__()
        self.with_pos = with_pos
        self.dropout_rate = dropout

        self.embedding_layer = nn.Linear(d_in, d_model)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_max_steps) if with_pos else None
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, X, missing_mask=None, time_mark=None):
        if missing_mask is not None:
            X = torch.cat([X, missing_mask], dim=2)
        if time_mark is not None:
            X = torch.cat([X, time_mark], dim=2)
        X_embedding = self.embedding_layer(X)
        if self.with_pos:
            X_embedding = self.position_enc(X_embedding)
        if self.dropout_rate > 0:
            X_embedding = self.dropout(X_embedding)

        return X_embedding

