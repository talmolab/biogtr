"""Module containing GTR model used for training."""

from biogtr.models.transformer import Transformer
from biogtr.models.visual_encoder import VisualEncoder
from biogtr.io.instance import Instance
import torch

# todo: do we want to handle params with configs already here?


class GlobalTrackingTransformer(torch.nn.Module):
    """Modular GTR model composed of visual encoder + transformer used for tracking."""

    def __init__(
        self,
        encoder_cfg: dict = None,
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: int = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
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
            encoder_cfg: Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"model_name": "resnet18", "pretrained": False, "in_chans": 3}`
            d_model: The number of features in the encoder/decoder inputs.
            nhead: The number of heads in the transfomer encoder/decoder.
            num_encoder_layers: The number of encoder-layers in the encoder.
            num_decoder_layers: The number of decoder-layers in the decoder.
            dropout: Dropout value applied to the output of transformer layers.
            activation: Activation function to use.
            return_intermediate_dec: Return intermediate layers from decoder.
            norm: If True, normalize output of encoder and decoder.
            num_layers_attn_head: The number of layers in the attention head.
            dropout_attn_head: Dropout value for the attention_head.
            embedding_meta: Metadata for positional embeddings. See below.
            return_embedding: Whether to return the positional embeddings
            decoder_self_attn: If True, use decoder self attention.

            More details on `embedding_meta`:
                By default this will be an empty dict and indicate
                that no positional embeddings should be used. To use the positional embeddings
                pass in a dictionary containing a "pos" and "temp" key with subdictionaries for correct parameters ie:
                {"pos": {'mode': 'learned', 'emb_num': 16, 'over_boxes: 'True'},
                "temp": {'mode': 'learned', 'emb_num': 16}}. (see `biogtr.models.embeddings.Embedding.EMB_TYPES`
                and `biogtr.models.embeddings.Embedding.EMB_MODES` for embedding parameters).
        """
        super().__init__()

        if encoder_cfg is not None:
            self.visual_encoder = VisualEncoder(d_model=d_model, **encoder_cfg)
        else:
            self.visual_encoder = VisualEncoder(d_model=d_model)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            norm=norm,
            num_layers_attn_head=num_layers_attn_head,
            dropout_attn_head=dropout_attn_head,
            embedding_meta=embedding_meta,
            return_embedding=return_embedding,
            decoder_self_attn=decoder_self_attn,
        )

    def forward(
        self, ref_instances: list[Instance], query_instances: list[Instance] = None
    ) -> list["AssociationMatrix"]:
        """Execute forward pass of GTR Model to get asso matrix.

        Args:
            frames: List of Frames from chunk containing crops of objects + gt label info
            query_frame: Frame index used as query for self attention. Only used in sliding inference where query frame is the last frame in the window.

        Returns:
            An N_T x N association matrix
        """
        # Extract feature representations with pre-trained encoder.
        if any(
            [
                (not instance.has_features()) and instance.has_crop()
                for instance in ref_instances
            ]
        ):
            ref_crops = torch.concat(
                [instance.crop for instance in ref_instances], axis=0
            )
            ref_z = self.visual_encoder(ref_crops)
            for i, z_i in enumerate(ref_z):
                ref_instances[i].features = z_i

        if query_instances:
            if any(
                [
                    (not instance.has_features()) and instance.has_crop()
                    for instance in query_instances
                ]
            ):
                query_crops = torch.concat(
                    [instance.crop for instance in query_instances], axis=0
                )
                query_z = self.visual_encoder(query_crops)
                for i, z_i in enumerate(query_z):
                    query_instances[i].features = z_i

        asso_preds = self.transformer(ref_instances, query_instances)

        return asso_preds
