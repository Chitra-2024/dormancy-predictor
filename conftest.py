import pytest
from src.data_generator import SyntheticBrokerData
from src.pipeline import Pipeline


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A small Bronze -> Silver pipeline in a temp folder."""
    root = tmp_path_factory.mktemp("lake")
    bronze, silver, gold = (str(root / d) for d in ["bronze", "silver", "gold"])
    SyntheticBrokerData(n_clients=500, seed=1).generate().write_bronze(bronze)
    pipe = Pipeline(bronze, silver, gold)
    counts = pipe.bronze_to_silver()
    return pipe, counts, silver