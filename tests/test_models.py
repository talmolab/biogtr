"""Test model modules."""

import pytest
import torch
from biogtr.data_structures import Frame, Instance
from biogtr.models.attention_head import MLP, ATTWeightHead
from biogtr.models.embedding import Embedding
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models.transformer import (
    Transformer,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from biogtr.models.visual_encoder import VisualEncoder


# todo: add named tensor tests


def test_mlp():
    """Test MLP logic."""
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    mlp = MLP(input_dim=f, hidden_dim=f, output_dim=f, num_layers=2, dropout=0.1)

    output_tensor = mlp(torch.rand(size=(b, n, f)))

    assert output_tensor.shape == (b, n, f)


def test_att_weight_head():
    """Test self-attention head logic."""
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    att_weight_head = ATTWeightHead(feature_dim=f, num_layers=2, dropout=0.1)

    q = k = torch.rand(size=(b, n, f))

    attn_weights = att_weight_head(q, k)

    assert attn_weights.shape == (b, n, n)


def test_encoder():
    """Test feature extractor logic."""
    b, c, h, w = 1, 1, 100, 100  # batch size, channels, height, width

    features = 512
    input_tensor = torch.rand(b, c, h, w)

    for model_name, weights_name in [
        ("resnet18", "ResNet18_Weights.DEFAULT"),
        ("resnet50", "ResNet50_Weights.DEFAULT"),
    ]:
        cfg = {"weights": weights_name}

        encoder = VisualEncoder(model_name, cfg, features)

        output = encoder(input_tensor)

        assert output.shape == (b, features)


def test_embedding_validity():
    """Test embedding usage."""

    # this would throw assertion since embedding should be "pos"
    with pytest.raises(Exception):
        _ = Embedding(type="position", mode="learned")
    with pytest.raises(Exception):
        _ = Embedding(type="position", mode="fixed")

    with pytest.raises(Exception):
        _ = Embedding(type="temporal", mode="learned")
    with pytest.raises(Exception):
        _ = Embedding(type="position", mode="fixed")

    with pytest.raises(Exception):
        _ = Embedding(type="pos", mode="learn")
    with pytest.raises(Exception):
        _ = Embedding(type="temp", mode="learn")

    with pytest.raises(Exception):
        _ = Embedding(type="pos", mode="fix")
    with pytest.raises(Exception):
        _ = Embedding(type="temp", mode="fix")

    with pytest.raises(Exception):
        _ = Embedding(type="position", mode="learn")
    with pytest.raises(Exception):
        _ = Embedding(type="temporal", mode="learn")
    with pytest.raises(Exception):
        _ = Embedding(type="position", mode="fix")
    with pytest.raises(Exception):
        _ = Embedding(type="temporal", mode="learn")

    with pytest.raises(Exception):
        _ = Embedding(type="temp", mode="fixed")

    _ = Embedding(type="temp", mode="learned")
    _ = Embedding(type="pos", mode="learned")

    _ = Embedding(type="pos", mode="learned")


def test_embedding():
    """Test embedding logic."""

    frames = 32
    objects = 10
    d_model = 256

    N = frames * objects

    boxes = torch.rand(size=(N, 4))
    times = torch.rand(size=(N,))

    pos_emb = Embedding(
        type="pos",
        mode="fixed",
        features=d_model,
        temperature=objects,
        normalize=True,
        scale=10,
    )

    sine_pos_emb = pos_emb(boxes)

    pos_emb = Embedding(type="pos", mode="learned", features=d_model, emb_num=100)
    learned_pos_emb = pos_emb(boxes)

    temp_emb = Embedding(type="temp", mode="learned", features=d_model, emb_num=16)
    learned_temp_emb = temp_emb(times)

    assert sine_pos_emb.size() == (N, d_model)
    assert learned_pos_emb.size() == (N, d_model)
    assert learned_temp_emb.size() == (N, d_model)

    assert not torch.equal(sine_pos_emb, learned_pos_emb)
    assert not torch.equal(sine_pos_emb, learned_temp_emb)
    assert not torch.equal(learned_pos_emb, learned_temp_emb)


def test_embedding_kwargs():
    """Test embedding config logic."""

    frames = 32
    objects = 10

    N = frames * objects

    boxes = torch.rand(size=(N, 4))
    times = torch.rand(size=(N,))

    # sine embedding

    sine_no_args = Embedding("pos", "fixed")(boxes)

    sine_args = {
        "temperature": objects,
        "scale": frames,
        "normalize": True,
    }

    sine_with_args = Embedding("pos", "fixed", **sine_args)(boxes)

    assert not torch.equal(sine_no_args, sine_with_args)

    # learned pos embedding

    lp_no_args = Embedding("pos", "learned")(boxes)

    lp_args = {"emb_num": 100, "over_boxes": False}

    lp_with_args = Embedding("pos", "learned", **lp_args)(boxes)

    assert not torch.equal(lp_no_args, lp_with_args)
    assert not torch.equal(lp_no_args, sine_no_args)
    assert not torch.equal(lp_no_args, sine_with_args)
    assert not torch.equal(lp_with_args, sine_no_args)
    assert not torch.equal(lp_with_args, sine_with_args)
    # learned temp embedding

    lt_no_args = Embedding("temp", "learned")(times)

    lt_args = {"emb_num": 100}

    lt_with_args = Embedding("temp", "learned", **lt_args)(times)

    assert not torch.equal(lt_no_args, lt_with_args)

    assert not torch.equal(lt_no_args, lp_no_args)
    assert not torch.equal(lt_no_args, lp_with_args)

    assert not torch.equal(lt_no_args, sine_no_args)
    assert not torch.equal(lt_no_args, sine_with_args)

    assert not torch.equal(lt_with_args, lp_no_args)
    assert not torch.equal(lt_with_args, lp_with_args)

    assert not torch.equal(lt_with_args, sine_no_args)
    assert not torch.equal(lt_with_args, sine_with_args)


def test_transformer_encoder():
    """Test transformer encoder layer logic."""
    feats = 256

    transformer_encoder = TransformerEncoderLayer(
        d_model=feats, nhead=1, dim_feedforward=feats, norm=True
    )

    N, B, D = 10, 1, feats

    # no position
    src = torch.rand(size=(N, B, D))

    out = transformer_encoder(src)

    assert out.size() == src.size()

    # with position
    pos = torch.ones_like(src)

    out = transformer_encoder(src, pos=pos)

    assert out.size() == src.size()


def test_transformer_decoder():
    """Test transformer decoder layer logic."""
    feats = 512

    transformer_decoder = TransformerDecoderLayer(
        d_model=feats,
        nhead=2,
        dim_feedforward=feats,
        dropout=0.2,
        norm=False,
        decoder_self_attn=True,
    )

    N, B, D = 10, 1, feats

    # no position
    tgt = memory = torch.rand(size=(N, B, D))

    out = transformer_decoder(tgt, memory)

    assert out.size() == tgt.size()

    # with position
    pos = tgt_pos = torch.ones_like(memory)

    out = transformer_decoder(tgt, memory, pos=pos, tgt_pos=tgt_pos)

    assert out.size() == pos.size()


def test_transformer_basic():
    """Test full transformer logic."""
    feats = 256
    num_frames = 32
    num_detected = 10
    img_shape = (1, 100, 100)

    transformer = Transformer(d_model=feats, num_encoder_layers=1, num_decoder_layers=1)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats))
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    asso_preds, _ = transformer(frames)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2


def test_transformer_embedding():
    """Test transformer using embedding."""
    feats = 256
    num_frames = 3
    num_detected = 10
    img_shape = (1, 50, 50)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats))
                )
            )
        frames.append(Frame(video_id=0, frame_id=i, instances=instances))

    embedding_meta = {
        "pos": {"mode": "learned", "emb_num": 16, "normalize": True},
        "temp": {"mode": "learned", "emb_num": 16, "normalize": True},
    }

    transformer = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    asso_preds, embedding = transformer(frames)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2
    assert embedding.size() == (num_detected * num_frames, 1, feats)


def test_tracking_transformer():
    """Test GTR logic."""
    feats = 512
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), crop=torch.rand(size=(1, 1, 64, 64))
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "pos": {
            "mode": "fixed",
            "temperature": num_detected,
            "scale": num_frames,
            "normalize": True,
        },
        "temp": None,
    }

    cfg = {"resnet18", "ResNet18_Weights.DEFAULT"}

    tracking_transformer = GlobalTrackingTransformer(
        encoder_model="resnet18",
        encoder_cfg={"weights": "ResNet18_Weights.DEFAULT"},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    asso_preds, embedding = tracking_transformer(frames)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2
    assert embedding.size() == (num_detected * num_frames, 1, feats)
