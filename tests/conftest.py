import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def paired_adatas():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(index=[f"spot_{i}" for i in range(8)])
    coordinates = np.array([[i % 4, i // 4] for i in range(8)], dtype=float)
    rna = AnnData(
        X=rng.poisson(3, size=(8, 5)).astype("float32"),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
        obsm={"spatial": coordinates.copy()},
    )
    adt = AnnData(
        X=rng.poisson(2, size=(8, 3)).astype("float32"),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"protein_{i}" for i in range(3)]),
        obsm={"spatial": coordinates.copy()},
    )
    return {"rna": rna, "adt": adt}
