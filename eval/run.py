import logging
import os

os.environ["LIGHTEVAL_DISABLE_TQDM"] = "1"

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
from lighteval.models.model_input import GenerationParameters
from lighteval.models.transformers.transformers_model import TransformersModel


class QuietTransformersModel(TransformersModel):
    @property
    def disable_tqdm(self) -> bool:
        return True


lighteval.models.transformers.transformers_model.TransformersModel = (
    QuietTransformersModel
)

import tasks
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


def evaluate_model(
    model=None,
    max_samples=None,
    tasks="helm|hellaswag|5|1",
    device="cpu",
    vocab_size=16384,
    verbose=True,
):

    cache_dir = "data"

    if not verbose:
        logging.getLogger("lighteval").setLevel(logging.CRITICAL)
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        # datasets.disable_progress_bars()

    evaluation_tracker = EvaluationTracker(
        output_dir=os.path.join(cache_dir, "eval"),
        save_details=True,
    )

    model_config = TransformersModelConfig(
        model_name=os.path.join(cache_dir, "praxis"),
        device=device,
        tokenizer=f"UNSAFE/praxis-{vocab_size}",
        model_parallel=False,
        batch_size=1,
        max_length=4096,
        generation_size=256,
        generation_parameters=GenerationParameters(
            # temperature=0.5,
            repetition_penalty=1.2,
            max_new_tokens=256,
        ),
    )
    if model is not None:
        model = TransformersModel.from_model(model, config=model_config)
        model_config = None

    pipeline = Pipeline(
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            max_samples=max_samples,
            use_chat_template=True,
            system_prompt="You are an intelligent chatbot. Please answer the following questions accurately.",
            custom_tasks_directory="tasks",
        ),
        evaluation_tracker=evaluation_tracker,
        model=model,
        model_config=model_config,
        tasks=tasks,
        # tasks="helm|praxis_hellaswag|5|1",
    )

    pipeline.evaluate()

    if verbose:
        pipeline.show_results()

    return pipeline.get_results()


def get_all_task_metrics(results_dict):
    """
    Extract all available metrics for each task in the results dictionary.

    Args:
        results_dict: The lighteval results dictionary returned by pipeline.get_results()

    Returns:
        List of dictionaries, each containing task name and all available metrics with their values
    """
    task_metrics = []

    # Skip the 'all' entry which is just an aggregate
    for task_key, task_results in results_dict["results"].items():
        if task_key == "all":
            continue

        # Clean the task name if needed
        clean_task = task_key

        for prefix in ["lighteval", "helm"]:
            prefix += ":"
            if task_key.startswith(prefix):
                clean_task = task_key.replace(prefix, "")

        # Create a dictionary for this task
        task_data = {"original_task": task_key, "task": clean_task}

        # Get the version from the versions dictionary
        if task_key in results_dict.get("versions", {}):
            task_data["version"] = results_dict["versions"][task_key]

        # Add all metrics available for this task
        for metric_key, metric_value in task_results.items():
            # Skip stderr entries as we'll handle them separately
            if metric_key.endswith("_stderr"):
                continue

            # Add the metric value
            task_data[metric_key] = metric_value

            # Add stderr if available
            stderr_key = f"{metric_key}_stderr"
            if stderr_key in task_results:
                task_data[stderr_key] = task_results[stderr_key]

        task_metrics.append(task_data)

    return task_metrics


if __name__ == "__main__":
    tasks = "lighteval|glue:sst2|2|1"
    evaluate_model(tasks=tasks, max_samples=100, verbose=True)
