from training.config import TrainingExperimentConfig
from training.dataset import TrainingDatasetBuilder
from training.trainer import TftTrainer


def test_training_factory_exposes_core_training_interfaces() -> None:
    assert TrainingExperimentConfig.__module__ == "training.config"
    assert TrainingDatasetBuilder.__module__ == "training.dataset"
    assert TftTrainer.__module__ == "training.trainer"
