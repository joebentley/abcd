import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


@pytest.fixture
def run_slow(request):
    return request.config.getoption("--runslow")
