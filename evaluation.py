import os

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker

# from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from lighteval.utils.utils import EnvConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

# if is_accelerate_available():
#     from accelerate import Accelerator, InitProcessGroupKwargs

#     accelerator = Accelerator(
#         kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
#     )
# else:
#     accelerator = None


# Global configuration
cache_dir = "data"
device = "cpu"
vocab_size = 16384


def evaluate_model(model=None, max_samples=None):
    evaluation_tracker = EvaluationTracker(
        output_dir=os.path.join(cache_dir, "eval"),
        save_details=True,
        # push_to_hub=False,
        # hub_results_org="UNSAFE",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir=os.path.join(cache_dir, "tmp")),
        override_batch_size=1,
        max_samples=max_samples,
    )

    if model is None:
        model_config = TransformersModelConfig(
            pretrained=os.path.join(cache_dir, "praxis"),
            device=device,
            tokenizer=f"UNSAFE/praxis-{vocab_size}",
            accelerator=None,
            model_parallel=False,
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
        # custom_task_directory=None,  # if using a custom task
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    evaluate_model(max_samples=10)

# import os

# import lm_eval
# from lm_eval.models.huggingface import HFLM
# from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

# from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# AutoConfig.register("praxis", PraxisConfig)
# AutoModel.register(PraxisConfig, PraxisModel)
# AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

# # Global configuration
# cache_dir = "data"
# device = "cuda:0"
# vocab_size = 8192

# # Tokenizer initialization
# tokenizer = AutoTokenizer.from_pretrained(
#     f"UNSAFE/praxis-{vocab_size}", cache_dir=cache_dir
# )

# # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
# # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
# eval_model = HFLM(
#     os.path.join(cache_dir, "praxis"),
#     backend="causal",
#     batch_size=1,
#     tokenizer=tokenizer,
#     device=device,
# )

# task_manager = lm_eval.tasks.TaskManager()
# results = lm_eval.simple_evaluate(
#     model=eval_model,
#     tasks=["arc_easy"],
#     # tasks=[
#     #     "arc_easy",
#     #     "arc_challenge",
#     #     "arithmetic",
#     #     "glue",
#     #     "hellaswag",
#     #     "openbookqa",
#     #     "piqa",
#     #     "sciq",
#     #     "squadv2",
#     #     "tinyMMLU",
#     #     "winogrande",
#     # ],
#     num_fewshot=0,
#     task_manager=task_manager,
# )
