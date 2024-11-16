import os

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

# Global configuration
cache_dir = "data"
device = "cuda:0"
vocab_size = 8192

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(
    f"UNSAFE/praxis-{vocab_size}", cache_dir=cache_dir
)

# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
eval_model = HFLM(
    os.path.join(cache_dir, "praxis"),
    backend="causal",
    batch_size=1,
    tokenizer=tokenizer,
    device=device,
)

task_manager = lm_eval.tasks.TaskManager()
results = lm_eval.simple_evaluate(
    model=eval_model,
    tasks=["arc_easy"],
    # tasks=[
    #     "arc_easy",
    #     "arc_challenge",
    #     "arithmetic",
    #     "glue",
    #     "hellaswag",
    #     "openbookqa",
    #     "piqa",
    #     "sciq",
    #     "squadv2",
    #     "tinyMMLU",
    #     "winogrande",
    # ],
    num_fewshot=0,
    task_manager=task_manager,
)
