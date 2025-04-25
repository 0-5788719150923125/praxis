import logging
import os
from pprint import pprint

import datasets
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["praxis"] = "PraxisForCausalLM"
import lighteval
from lighteval.models.transformers.transformers_model import TransformersModel


class QuietTransformersModel(TransformersModel):
    @property
    def disable_tqdm(self) -> bool:
        return True


lighteval.models.transformers.transformers_model.TransformersModel = (
    QuietTransformersModel
)

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


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

    # if model is None and model_config is None:
    model_config = TransformersModelConfig(
        model_name=os.path.join(cache_dir, "praxis"),
        device=device,
        tokenizer=f"UNSAFE/praxis-{vocab_size}",
        model_parallel=False,
        batch_size=1,
        max_length=4096,
    )
    if model is not None:
        # model = TransformersModel(model_config)
        # print(model_config)
        model = TransformersModel.from_model(
            model, config=model_config, tokenizer_name=f"UNSAFE/praxis-{vocab_size}"
        )
        model_config = None

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
