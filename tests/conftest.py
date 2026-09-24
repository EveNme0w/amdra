import pytest

from amdra.config import Settings
from amdra.data.generate import generate


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("synthetic")
    generate(data_dir, per_scenario=3, seed=7)
    return Settings(llm="offline", vector_backend="memory", embedder="hashing",
                    ocr="auto", data_dir=data_dir)


@pytest.fixture(scope="session")
def cases(settings):
    from amdra.data.generate import load_cases

    return load_cases(settings.cases_path)
