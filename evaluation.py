import os

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)


# Global configuration
cache_dir = "data"
device = "cpu"
vocab_size = 16384


def evaluate_model(model=None, max_samples=None):
    evaluation_tracker = EvaluationTracker(
        output_dir=os.path.join(cache_dir, "eval"),
        save_details=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=max_samples,
    )

    if model is None:
        # from transformers import GenerationConfig

        model_config = TransformersModelConfig(
            model_name=os.path.join(cache_dir, "praxis"),
            device=device,
            tokenizer=f"UNSAFE/praxis-{vocab_size}",
            model_parallel=False,
            batch_size=1,
            # generation_config=GenerationConfig(use_cache=False),
        )
    else:
        model_config = None

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
    pipeline.show_results()


if __name__ == "__main__":
    evaluate_model(max_samples=10)
