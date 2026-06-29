from training.catalog import TrainingCatalog
from training.config import ExperimentTemplate, ResolvedTrialSnapshot, StudySpec
from training.dataset import TrainingDatasetBuilder
from training.experiment import FinalEvaluationApproval
from training.trainer import TftTrainer


def test_training_factory_exposes_core_training_interfaces() -> None:
    assert ExperimentTemplate.__module__ == "training.config"
    assert StudySpec.__module__ == "training.config"
    assert ResolvedTrialSnapshot.__module__ == "training.config"
    assert TrainingCatalog.__module__ == "training.catalog"
    assert FinalEvaluationApproval.__module__ == "training.experiment"
    assert TrainingDatasetBuilder.__module__ == "training.dataset"
    assert TftTrainer.__module__ == "training.trainer"
