# praxis

<!-- Triangulated Human Observation for Reasoning in the Natural Sciences -->

*Praxis is the process by which a theory, lesson, or skill is enacted, embodied, realized, applied, or put into practice.*

![Terminal](./static/terminal.webp)

## what we're building

The Praxis swarm is a decentralized, peer-to-peer, always online, and continuously-learning AI intelligence - with Hivemind integrated into core layers of the model itself. We will do this via a mixture of experts, sparse routing, algorithmic switching and self-modeling of remotely-hosted peers.

## install

From a Linux shell, run these commands:

```sh
# Setup a virtual environment
source make-venv.sh

# Install core model dependencies
pip install .

# Install training dependencies
pip install .[all]
```

## contribute to the swarm

To donate your compute:

```sh
python server.py
```

## do inference

Send a JSON-encoded payload via POST to:

```
http://localhost:5000/generate
```

This payload should support all arguments in the [Transformers text generation API](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

## to register with transformers

```py
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

config = PraxisConfig(
    n_positions=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    device_map="cuda:0",
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

tokenizer_model = "NousResearch/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

model = AutoModelForCausalLM.from_config(config)

input_ids = tokenizer.encode("The quick brown fox ")

outputs = model.generate(input_ids, do_sample=True)

print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
# --> The quick brown fox jumped over a lazy dog.
```

## ideas

- a GLOBAL swarm, by default
- replace dense layers with remote experts, then leverage [self-modeling](https://arxiv.org/abs/2407.10188) to train the network to model those peers
- make experts deeper; if an expert is comprised of multiple transformer blocks, rather than a single dense layer, then the network could learn to dynamically-route through deeper networks, or it could learn to relay information across multiple peers, or it could learn that "no relay is needed" at all, simply returning simple predictions back to the requestee. For example, let's say that every peer runs a 6-layer transformer, comprised of three experts - with 2 layers each. By routing through multiple peers, one could theoretically turn a 6-layer transformer into 600 layers (if routing between tons of peers). Obviously, 600 layers would be tough to optimize, but if each peer kept optimization local - that's just 6 layers to optimize. Each peer becomes highly specialized in "something."
- treat every peer as an experiment in a massive hyperparameter search; publish results to the DHT, and ensure that better-performing hparams are assigned more often
- ship with an API server; if you run a node, you also get a local API that you can use for inference
- build connectors, allowing people to specialize their nodes on their own data
- [Mixture of a Million Experts](https://arxiv.org/abs/2407.04153)?