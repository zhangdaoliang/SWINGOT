import pandas as pd

from swingot import pp


def test_spatial_network(paired_adatas):
    rna = paired_adatas["rna"]
    result = pp.cal_spatial_net(rna, cutoff=1.1, model="Radius", verbose=False, copy=True)
    assert isinstance(result, pd.DataFrame)
    assert {"Cell1", "Cell2", "Distance"} <= set(result.columns)
    assert not result.empty
    assert (result["Cell1"] != result["Cell2"]).all()


def test_setup_data_uses_swingot_key(paired_adatas):
    rna = paired_adatas["rna"]
    pp.setup_data(rna, prob_model="NB", use_highly_variable=False, use_spatial=False)
    config = rna.uns["SWINGOT_config"]
    assert config["prob_model"] == "NB"
    assert config["features"] == list(rna.var_names)
    assert config["use_spatial"] is False


