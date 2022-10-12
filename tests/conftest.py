def pytest_addoption(parser):
    parser.addoption(
        "--reset",
        action="store_true",
        default=False,
        help="Reset a Simulator's state after every benchmark test iteration",
    )
