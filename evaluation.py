import logging
import os
from pprint import pprint

import datasets
import lighteval
from lighteval.models.transformers.transformers_model import TransformersModel


class QuietTransformersModel(TransformersModel):
    @property
    def disable_tqdm(self) -> bool:
        return True

    # def _create_auto_model(self, model=None) -> transformers.PreTrainedModel:
    #     if model:
    #         pass
    #     else:
    #         return super()._create_auto_model()


lighteval.models.transformers.transformers_model.TransformersModel = (
    QuietTransformersModel
)

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)


def evaluate_model(
    model=None,
    model_config=None,
    max_samples=None,
    task="helm|mmlu|5|1",
    device="cpu",
    vocab_size=16384,
    verbose=True,
):

    cache_dir = "data"

    if not verbose:
        logging.getLogger("lighteval").setLevel(logging.CRITICAL)
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        datasets.disable_progress_bars()

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
            max_length=4096,
        )
    elif model is not None:
        model = TransformersModel.from_model(
            model, tokenizer_name=f"UNSAFE/praxis-{vocab_size}"
        )

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

    return pipeline.get_results()


if __name__ == "__main__":
    evaluate_model(max_samples=10, verbose=True)
