"""Experiment configuration for reproducible training-factory runs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_TEMPLATE_SCHEMA_VERSION = 1
_SUPPORTED_TEMPLATE_SCHEMA_VERSIONS = {1, 2}
_FIXED_QUANTILES = (0.1, 0.5, 0.9)
_SEARCHABLE_PARAMETERS = {
    "hidden_size",
    "attention_head_size",
    "hidden_continuous_size",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_clip_val",
}


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.20
    hidden_continuous_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    reduce_lr_patience: int = 3
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    target_winsor: float = 8.0
    max_encoder_length: int = 72
    max_prediction_length: int = 48
    use_history_target: bool = False


@dataclass(frozen=True)
class TrainerConfig:
    min_epochs: int = 5
    max_epochs: int = 20
    batch_size: int = 32
    seed: int = 42
    deterministic: bool = True
    precision: str = "32"
    num_workers: int = 0
    early_stop_patience: int = 2
    gradient_clip_val: float = 0.1


@dataclass(frozen=True)
class SearchParameter:
    kind: str
    choices: tuple[Any, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None
    log: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any], name: str) -> "SearchParameter":
        if not isinstance(payload, dict):
            raise ValueError(f"study.search_space.{name} must be an object")
        kind = str(payload.get("type", ""))
        if kind == "categorical":
            _require_exact_fields(
                payload,
                required={"type", "choices"},
                context=f"study.search_space.{name}",
            )
            raw_choices = payload["choices"]
            if not isinstance(raw_choices, list) or not raw_choices:
                raise ValueError(
                    f"study.search_space.{name}.choices must be a non-empty list"
                )
            return cls(kind=kind, choices=tuple(raw_choices))
        if kind in {"float", "int"}:
            _require_exact_fields(
                payload,
                required={"type", "min", "max"},
                optional={"log"},
                context=f"study.search_space.{name}",
            )
            minimum = payload["min"]
            maximum = payload["max"]
            if not isinstance(minimum, (int, float)) or not isinstance(
                maximum, (int, float)
            ):
                raise ValueError(
                    f"study.search_space.{name} bounds must be numeric"
                )
            if minimum >= maximum:
                raise ValueError(
                    f"study.search_space.{name} min must be less than max"
                )
            if kind == "int" and (
                not isinstance(minimum, int) or not isinstance(maximum, int)
            ):
                raise ValueError(
                    f"study.search_space.{name} integer bounds must be integers"
                )
            if bool(payload.get("log", False)) and minimum <= 0:
                raise ValueError(
                    f"study.search_space.{name} log bounds must be positive"
                )
            return cls(
                kind=kind,
                minimum=minimum,
                maximum=maximum,
                log=bool(payload.get("log", False)),
            )
        raise ValueError(
            f"study.search_space.{name}.type must be categorical, float, or int"
        )

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "categorical":
            return {"type": self.kind, "choices": list(self.choices)}
        payload: dict[str, Any] = {
            "type": self.kind,
            "min": self.minimum,
            "max": self.maximum,
        }
        if self.log:
            payload["log"] = True
        return payload


@dataclass(frozen=True)
class SamplerConfig:
    name: str
    seed: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SamplerConfig":
        _require_exact_fields(
            payload,
            required={"name", "seed"},
            context="study.sampler",
        )
        if payload["name"] != "tpe":
            raise ValueError("study.sampler.name must be tpe in schema v1")
        return cls(name="tpe", seed=int(payload["seed"]))


@dataclass(frozen=True)
class PrunerConfig:
    name: str
    startup_trials: int
    warmup_epochs: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrunerConfig":
        _require_exact_fields(
            payload,
            required={"name", "startup_trials", "warmup_epochs"},
            context="study.pruner",
        )
        if payload["name"] != "median":
            raise ValueError("study.pruner.name must be median in schema v1")
        startup_trials = int(payload["startup_trials"])
        warmup_epochs = int(payload["warmup_epochs"])
        if startup_trials < 0 or warmup_epochs < 0:
            raise ValueError("study.pruner counts must be non-negative")
        return cls(
            name="median",
            startup_trials=startup_trials,
            warmup_epochs=warmup_epochs,
        )


@dataclass(frozen=True)
class StudySpec:
    schema_version: int
    objective: str
    direction: str
    budget_gpu_hours: float
    top_k: int
    sampler: SamplerConfig
    pruner: PrunerConfig
    search_space: Mapping[str, SearchParameter]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudySpec":
        _require_exact_fields(
            payload,
            required={
                "schema_version",
                "objective",
                "direction",
                "budget_gpu_hours",
                "top_k",
                "sampler",
                "pruner",
                "search_space",
            },
            context="study",
        )
        if int(payload["schema_version"]) != _TEMPLATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported study schema_version={payload['schema_version']}"
            )
        if payload["objective"] != "validation_quantile_loss":
            raise ValueError(
                "study.objective must be validation_quantile_loss in schema v1"
            )
        if payload["direction"] != "minimize":
            raise ValueError("study.direction must be minimize in schema v1")
        budget = float(payload["budget_gpu_hours"])
        top_k = int(payload["top_k"])
        if budget <= 0:
            raise ValueError("study.budget_gpu_hours must be positive")
        if top_k <= 0:
            raise ValueError("study.top_k must be positive")
        raw_space = payload["search_space"]
        if not isinstance(raw_space, dict) or not raw_space:
            raise ValueError("study.search_space must be a non-empty object")
        unknown_parameters = sorted(set(raw_space) - _SEARCHABLE_PARAMETERS)
        if unknown_parameters:
            raise ValueError(
                "study.search_space contains unsupported parameter(s): "
                + ", ".join(unknown_parameters)
            )
        return cls(
            schema_version=1,
            objective=str(payload["objective"]),
            direction=str(payload["direction"]),
            budget_gpu_hours=budget,
            top_k=top_k,
            sampler=SamplerConfig.from_dict(payload["sampler"]),
            pruner=PrunerConfig.from_dict(payload["pruner"]),
            search_space=MappingProxyType(
                {
                    name: SearchParameter.from_dict(spec, name)
                    for name, spec in sorted(raw_space.items())
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "direction": self.direction,
            "budget_gpu_hours": self.budget_gpu_hours,
            "top_k": self.top_k,
            "sampler": asdict(self.sampler),
            "pruner": asdict(self.pruner),
            "search_space": {
                name: spec.to_dict()
                for name, spec in sorted(self.search_space.items())
            },
        }


@dataclass(frozen=True)
class SmokePolicy:
    max_epochs: int
    batch_size: int
    precision: str
    num_workers: int
    limit_train_batches: int
    limit_val_batches: int
    timeout_minutes: int
    evaluate_test: bool
    publish_candidate: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SmokePolicy":
        required = {item.name for item in fields(cls)}
        _require_exact_fields(payload, required=required, context="execution.smoke")
        policy = cls(**payload)
        if (
            policy.max_epochs != 1
            or policy.batch_size <= 0
            or policy.num_workers < 0
            or policy.limit_train_batches <= 0
            or policy.limit_val_batches <= 0
            or policy.timeout_minutes <= 0
        ):
            raise ValueError("execution.smoke limits are invalid")
        if policy.evaluate_test or policy.publish_candidate:
            raise ValueError(
                "execution.smoke cannot evaluate test or publish candidate"
            )
        return policy


@dataclass(frozen=True)
class CalibrationPolicy:
    max_epochs: int
    max_gpu_hours: float
    evaluate_test: bool
    publish_candidate: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationPolicy":
        required = {item.name for item in fields(cls)}
        _require_exact_fields(
            payload,
            required=required,
            context="execution.calibration",
        )
        policy = cls(**payload)
        if policy.max_epochs != 1 or policy.max_gpu_hours <= 0:
            raise ValueError("execution.calibration limits are invalid")
        if policy.evaluate_test or policy.publish_candidate:
            raise ValueError(
                "execution.calibration cannot evaluate test or publish candidate"
            )
        return policy


@dataclass(frozen=True)
class ExecutionPolicy:
    smoke: SmokePolicy
    calibration: CalibrationPolicy

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionPolicy":
        _require_exact_fields(
            payload,
            required={"smoke", "calibration"},
            context="execution",
        )
        return cls(
            smoke=SmokePolicy.from_dict(payload["smoke"]),
            calibration=CalibrationPolicy.from_dict(payload["calibration"]),
        )


@dataclass(frozen=True)
class ExperimentTemplate:
    schema_version: int
    template_id: str
    station: str
    splits: Mapping[str, tuple[str, str]]
    model: ModelConfig
    trainer: TrainerConfig
    runtime_profile: str
    study: StudySpec
    execution: ExecutionPolicy

    @classmethod
    def from_json(cls, path: Path | str) -> "ExperimentTemplate":
        payload = json.loads(Path(path).read_text(encoding="ascii"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentTemplate":
        _require_exact_fields(
            payload,
            required={
                "schema_version",
                "template_id",
                "station",
                "splits",
                "model",
                "trainer",
                "runtime_profile",
                "study",
                "execution",
            },
            context="experiment template",
        )
        schema_version = int(payload["schema_version"])
        if schema_version not in _SUPPORTED_TEMPLATE_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported experiment template schema_version={schema_version}"
            )
        template_id = str(payload["template_id"]).strip()
        station = str(payload["station"]).strip().upper()
        if not template_id:
            raise ValueError("template_id must be non-empty")
        if not station:
            raise ValueError("station must be non-empty")
        raw_splits = payload["splits"]
        _require_exact_fields(
            raw_splits,
            required={"train", "val", "test"},
            context="splits",
        )
        splits = {
            name: _split_tuple(raw_splits[name], name)
            for name in ("train", "val", "test")
        }
        template = cls(
            schema_version=schema_version,
            template_id=template_id,
            station=station,
            splits=MappingProxyType(splits),
            model=_strict_dataclass(ModelConfig, payload["model"], "model"),
            trainer=_strict_dataclass(TrainerConfig, payload["trainer"], "trainer"),
            runtime_profile=str(payload["runtime_profile"]),
            study=StudySpec.from_dict(payload["study"]),
            execution=ExecutionPolicy.from_dict(payload["execution"]),
        )
        template.validate()
        return template

    def validate(self) -> None:
        _validate_common_config(self.station, self.splits, self.model, self.trainer)
        if self.model.quantiles != _FIXED_QUANTILES:
            raise ValueError(
                "schema v1 fixes model.quantiles to [0.1, 0.5, 0.9]"
            )
        if (
            self.model.max_encoder_length != 72
            or self.model.max_prediction_length != 48
        ):
            raise ValueError(
                "schema v1 fixes encoder/prediction lengths to 72/48"
            )
        batch_space = self.study.search_space.get("batch_size")
        if (
            batch_space is not None
            and (
                batch_space.kind != "categorical"
                or not set(batch_space.choices)
                <= {4, 8, 16, 32, 64, 128, 256}
            )
        ):
            raise ValueError(
                "study.search_space.batch_size choices must be powers of two "
                "no larger than 256"
            )

    def to_dict(self) -> dict[str, Any]:
        model_payload = asdict(self.model)
        if self.schema_version == 1:
            model_payload.pop("use_history_target")
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "station": self.station,
            "splits": {key: list(value) for key, value in self.splits.items()},
            "model": model_payload,
            "trainer": asdict(self.trainer),
            "runtime_profile": self.runtime_profile,
            "study": self.study.to_dict(),
            "execution": {
                "smoke": asdict(self.execution.smoke),
                "calibration": asdict(self.execution.calibration),
            },
        }

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="ascii",
        )

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def apply_to_settings(self, settings: Any) -> None:
        _apply_to_settings(self, settings)

    def validate_audit(self, audit_summary: dict[str, Any]) -> str:
        if not audit_summary.get("ready", False):
            raise ValueError("experiment template requires a ready audit")
        audit_station = str(audit_summary.get("station_id", ""))
        if audit_station != self.station:
            raise ValueError(
                f"audit station {audit_station!r} does not match "
                f"template station {self.station!r}"
            )
        data_order_hash = str(audit_summary.get("data_order_hash", ""))
        if not data_order_hash:
            raise ValueError("ready audit must contain data_order_hash")
        return data_order_hash


@dataclass(frozen=True)
class ResolvedTrialSnapshot:
    schema_version: int
    study_id: str
    trial_number: int
    execution_type: str
    template_hash: str
    data_order_hash: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResolvedTrialSnapshot":
        _require_exact_fields(
            payload,
            required={
                "schema_version",
                "study_id",
                "trial_number",
                "execution_type",
                "template_hash",
                "data_order_hash",
                "parameters",
            },
            context="resolved trial snapshot",
        )
        schema_version = int(payload["schema_version"])
        if schema_version != _TEMPLATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported trial snapshot schema_version={schema_version}"
            )
        execution_type = str(payload["execution_type"])
        if execution_type not in {"smoke", "calibration", "trial", "final"}:
            raise ValueError("resolved trial snapshot execution_type is invalid")
        trial_number = int(payload["trial_number"])
        if trial_number < 0:
            raise ValueError("resolved trial snapshot trial_number must be non-negative")
        parameters = payload["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("resolved trial snapshot parameters must be an object")
        required_text = {
            name: str(payload[name]).strip()
            for name in ("study_id", "template_hash", "data_order_hash")
        }
        if any(not value for value in required_text.values()):
            raise ValueError("resolved trial snapshot identifiers must be non-empty")
        return cls(
            schema_version=schema_version,
            study_id=required_text["study_id"],
            trial_number=trial_number,
            execution_type=execution_type,
            template_hash=required_text["template_hash"],
            data_order_hash=required_text["data_order_hash"],
            parameters=MappingProxyType(dict(parameters)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "trial_number": self.trial_number,
            "execution_type": self.execution_type,
            "template_hash": self.template_hash,
            "data_order_hash": self.data_order_hash,
            "parameters": dict(self.parameters),
        }

    def snapshot_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _split_tuple(value: Any, name: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"splits.{name} must be a [start, end] pair")
    return str(value[0]), str(value[1])


def _strict_dataclass(cls: type, payload: dict[str, Any], context: str) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{context} contains unknown field(s): " + ", ".join(unknown)
        )
    values = dict(payload)
    if cls is ModelConfig and "quantiles" in values:
        quantiles = values["quantiles"]
        if not isinstance(quantiles, (list, tuple)):
            raise ValueError("model.quantiles must be a list")
        values["quantiles"] = tuple(float(value) for value in quantiles)
    return cls(**values)


def _require_exact_fields(
    payload: Any,
    *,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    optional = optional or set()
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"{context} missing required field(s): " + ", ".join(missing)
        )
    unknown = sorted(set(payload) - required - optional)
    if unknown:
        raise ValueError(
            f"{context} contains unknown field(s): " + ", ".join(unknown)
        )


def _validate_common_config(
    station: str,
    splits: Mapping[str, tuple[str, str]],
    model: ModelConfig,
    trainer: TrainerConfig,
) -> None:
    if not station:
        raise ValueError("station must be non-empty")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("splits must contain train, val, and test")
    parsed: dict[str, tuple[Any, Any]] = {}
    for name, (start, end) in splits.items():
        try:
            start_date = date.fromisoformat(start)
            end_date = date.fromisoformat(end)
        except ValueError as exc:
            raise ValueError(f"splits.{name} dates must use YYYY-MM-DD") from exc
        if start_date > end_date:
            raise ValueError(f"splits.{name} start must not exceed end")
        parsed[name] = (start_date, end_date)
    if not (
        parsed["train"][1] < parsed["val"][0]
        and parsed["val"][1] < parsed["test"][0]
    ):
        raise ValueError("train, val, and test splits must be ordered and disjoint")
    quantiles = model.quantiles
    if not quantiles or quantiles != tuple(sorted(quantiles)):
        raise ValueError("model.quantiles must be sorted and non-empty")
    if any(q <= 0 or q >= 1 for q in quantiles):
        raise ValueError("model.quantiles must be between 0 and 1")
    if model.hidden_size <= 0 or model.attention_head_size <= 0:
        raise ValueError("model hidden and attention sizes must be positive")
    if model.hidden_continuous_size <= 0:
        raise ValueError("model.hidden_continuous_size must be positive")
    if not 0 <= model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if model.learning_rate <= 0 or model.weight_decay < 0:
        raise ValueError("model optimizer values are invalid")
    if model.max_encoder_length <= 0 or model.max_prediction_length <= 0:
        raise ValueError("encoder and prediction lengths must be positive")
    if trainer.min_epochs < 0 or trainer.max_epochs < trainer.min_epochs:
        raise ValueError("trainer epoch bounds are invalid")
    if trainer.batch_size <= 0:
        raise ValueError("trainer.batch_size must be positive")
    if trainer.num_workers < 0 or trainer.gradient_clip_val < 0:
        raise ValueError("trainer runtime values are invalid")


def _apply_to_settings(config: Any, settings: Any) -> None:
    settings.train_start, settings.train_end = config.splits["train"]
    settings.val_start, settings.val_end = config.splits["val"]
    settings.test_start, settings.test_end = config.splits["test"]
    settings.tft_hidden_size = config.model.hidden_size
    settings.tft_attention_head_size = config.model.attention_head_size
    settings.tft_dropout = config.model.dropout
    settings.tft_hidden_continuous_size = config.model.hidden_continuous_size
    settings.tft_learning_rate = config.model.learning_rate
    settings.tft_weight_decay = config.model.weight_decay
    settings.tft_reduce_lr_patience = config.model.reduce_lr_patience
    settings.tft_quantiles = list(config.model.quantiles)
    settings.tft_target_winsor = config.model.target_winsor
    settings.tft_max_encoder_length = config.model.max_encoder_length
    settings.tft_max_prediction_length = config.model.max_prediction_length
    settings.tft_use_history_target = config.model.use_history_target
    settings.tft_min_epochs = config.trainer.min_epochs
    settings.tft_max_epochs = config.trainer.max_epochs
    settings.tft_batch_size = config.trainer.batch_size
    settings.tft_early_stop_patience = config.trainer.early_stop_patience
