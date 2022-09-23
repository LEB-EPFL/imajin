import pytest


def test_simulator_processors_run_each_step(simulator, preprocessor, post_processor):
    simulator.preprocessors = [preprocessor]
    simulator.post_processors = [post_processor]

    simulator.step()

    preprocessor.assert_called_once()
    post_processor.assert_called_once()
