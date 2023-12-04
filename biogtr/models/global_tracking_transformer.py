"""Module containing GTR model used for training."""
from biogtr.models.transformer import Transformer
from biogtr.models.visual_encoder import VisualEncoder
from biogtr.data_structures import Frame
from torch import nn

# todo: do we want to handle params with configs already here?


class GlobalTrackingTransformer(nn.Module):
    """Modular GTR model composed of visual encoder + transformer used for tracking."""

    def __init__(
        self,
        encoder_model: str = "resnet18",
        encoder_cfg: dict = {},
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: int = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        feature_dim_attn_head: int = 1024,
        norm: bool = False,
        num_layers_attn_head: int = 2,
        dropout_attn_head: int = 0.1,
        embedding_meta: dict = None,
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
        **kwargs,
    ):
        """Initialize GTR.

        Args:
            encoder_model: Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            encoder_cfg: Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"weights": "ResNet18_Weights.DEFAULT"}`
            d_model: The number of features in the encoder/decoder inputs.
            nhead: The number of heads in the transfomer encoder/decoder.
            num_encoder_layers: The number of encoder-layers in the encoder.
            num_decoder_layers: The number of decoder-layers in the decoder.
            dim_feedforward: The dimension of the feedforward layers of the transformer.
            dropout: Dropout value applied to the output of transformer layers.
            activation: Activation function to use.
            return_intermediate_dec: Return intermediate layers from decoder.
            norm: If True, normalize output of encoder and decoder.
            feature_dim_attn_head: The number of features in the attention head.
            num_layers_attn_head: The number of layers in the attention head.
            dropout_attn_head: Dropout value for the attention_head.
            embedding_meta: Metadata for positional embeddings. See below.
            return_embedding: Whether to return the positional embeddings
            decoder_self_attn: If True, use decoder self attention.
            embedding_meta: By default this will be an empty dict and indicate
                that no positional embeddings should be used. To use positional
                embeddings, a dict should be passed with the type of embedding to
                use. Valid options are:
                    * learned_pos: only learned position embeddings
                    * learned_temp: only learned temporal embeddings
                    * learned_pos_temp: learned position and temporal embeddings
                    * fixed_pos: fixed sine position embeddings
                    * fixed_pos_temp: fixed sine position and learned temporal embeddings
                You can additionally pass kwargs to override the default
                embedding values (see embedding.py function methods for relevant
                embedding parameters). Example:
                    embedding_meta = {
                        'embedding_type': 'learned_pos_temp',
                        'kwargs': {
                            'learn_pos_emb_num': 16,
                            'learn_temp_emb_num': 16,
                            'over_boxes': False
                        }
                    }
                Note: Embedding features are handled directly in the forward
                pass for each case. Overriding the features through kwargs will
                likely throw errors due to incorrect tensor shapes.
        """
        super().__init__()

        self.visual_encoder = VisualEncoder(encoder_model, encoder_cfg, d_model)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            feature_dim_attn_head=feature_dim_attn_head,
            norm=norm,
            num_layers_attn_head=num_layers_attn_head,
            dropout_attn_head=dropout_attn_head,
            embedding_meta=embedding_meta,
            return_embedding=return_embedding,
            decoder_self_attn=decoder_self_attn,
        )

    def forward(self, frames: list[Frame], query_frame: int = None):
        """Execute forward pass of GTR Model to get asso matrix.

        Args:
            frames: List of Frames from chunk containing crops of objects + gt label info
            query_frame: Frame index used as query for self attention. Only used in sliding inference where query frame is the last frame in the window.

        Returns:
            An N_T x N association matrix
        """
        # Extract feature representations with pre-trained encoder.
        for frame in frames:
            if frame.has_instances():
                if not frame.has_features():
                    crops = frame.get_crops()
                    z = self.visual_encoder(crops)

                    for i, z_i in enumerate(z):
                        frame.instances[i].features = z_i

        asso_preds, emb = self.transformer(frames, query_frame=query_frame)

        return asso_preds, emb
