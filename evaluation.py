import logging

import tqdm


def nop(it, *a, **k):
    return it


tqdm.tqdm = nop

import os

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
# from lighteval.models.transformers.transformers_model import TransformersModel


# class QuietTransformersModel(TransformersModel):
#     @property
#     def disable_tqdm(self) -> bool:
#         return True


# lighteval.models.transformers.transformers_model.TransformersModel = (
#     QuietTransformersModel
)
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# import warnings


# ignored_warnings = [
#     ".*Creating parquet from Arrow format*",
#     ".*Token indices sequence length is longer than the specified maximum sequence length*",
# ]
# for pattern in ignored_warnings:
#     warnings.filterwarnings("ignore", pattern)


AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)


# Global configuration
cache_dir = "data"
device = "cpu"
vocab_size = 16384


def evaluate_model(model=None, model_config=None, max_samples=None, verbose=True):

    if not verbose:

        logging.getLogger("lighteval").setLevel(logging.CRITICAL)

    evaluation_tracker = EvaluationTracker(
        output_dir=os.path.join(cache_dir, "eval"),
        save_details=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=max_samples,
    )

    if model is None and model_config is None:
        model_config = TransformersModelConfig(
            model_name=os.path.join(cache_dir, "praxis"),
            device=device,
            tokenizer=f"UNSAFE/praxis-{vocab_size}",
            model_parallel=False,
            batch_size=1,
        )

    task = "helm|mmlu|5|1"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()

    if verbose:
        pipeline.show_results()

    return pipeline.get_results()["results"]["all"]["pqem"]


if __name__ == "__main__":
    evaluate_model(max_samples=10, verbose=False)
