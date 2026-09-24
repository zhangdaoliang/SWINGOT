import numpy as np
import pytest

from swingot import SWINGOT, pp


def test_swingot_rejects_unpaired_modalities(paired_adatas):
    paired_adatas["adt"] = paired_adatas["adt"][:-1].copy()
    with pytest.raises(ValueError, match="row-paired modalities"):
        SWINGOT(adatas=paired_adatas, vertices=[])


def test_swingot_encodes_paired_spots(paired_adatas):
    for adata in paired_adatas.values():
        pp.setup_data(adata, prob_model="NB", use_highly_variable=False, use_spatial=False)
    vertices = sorted(set(paired_adatas["rna"].var_names) | set(paired_adatas["adt"].var_names))
    model = SWINGOT(
        adatas=paired_adatas,
        vertices=vertices,
        latent_dim=4,
        conv_layer="LIN",
        dropout=0.0,
        seed=0,
    )
    embedding = model.encode_fused(paired_adatas)
    assert embedding.shape == (8, 4)
    assert np.isfinite(embedding).all()
