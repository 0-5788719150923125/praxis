import lighteval.tasks.default_prompts as prompt
import numpy as np
from aenum import extend_enum
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

praxis_hellaswag = LightevalTaskConfig(
    name="praxis_hellaswag",
    suite=["helm", "helm_general"],
    prompt_function=prompt.hellaswag_generative,
    hf_repo="hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
    ],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)
