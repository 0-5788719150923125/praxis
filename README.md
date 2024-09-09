# praxis

<!-- Triangulated Human Observation for Reasoning in the Natural Sciences -->

*Praxis is the process by which a theory, lesson, or skill is enacted, embodied, realized, applied, or put into practice.*

![Terminal](./static/terminal.webp)

## what we're building

The Praxis swarm is a decentralized, peer-to-peer, always online, and continuously-learning AI intelligence - with [Hivemind](https://github.com/learning-at-home/hivemind) directly-integrated into core layers of the model itself. The goal is to build an expert model that is small and simple, easy to parallelize and performant at a scale of hundreds/thousands of peers. We will do this via a sparse mixture of experts, curated routing, algorithmic switching and weighted self-modeling of remotely-hosted peers.

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

Supported arguments:
```sh
python server.py \
  --data_path /path/to/my/data \ # Train on a local directory of data.
  --device cuda:0 \              # Specify your GPU's index. Omit this argument to use CPU.
  --no_dashboard                 # Disables the CLI interface.
```

## join us

- [Discord](https://discord.gg/8ZmHP8CqUX)
- [The Source](https://src.eco)

## do inference

Send a JSON-encoded payload via POST to:

```
http://localhost:5000/generate
```

This payload should support all arguments in the [Transformers text generation API](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

Example request:
```py
import requests

url = "http://localhost:5000/generate"
payload = {"prompt": "Once upon a time, ", "do_sample": True, "temperature": 0.7}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

## to register with transformers

```py
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

config = PraxisConfig(
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

- a global swarm, exclusively
- leverage [self-modeling](https://arxiv.org/abs/2407.10188) to focus learning on remote peers
- experts with a stack of attention/feedforward blocks
- if an expert is comprised of multiple transformer blocks, rather than a single layer, then the network might learn to dynamically-route through deeper subnetworks, or it could learn to relay/ensemble information across multiple peers, or it could learn that "no relay is needed" at all, simply returning a simple prediction back to the requestee.
- treat every peer as an experiment in hyperparameter search; publish results to the DHT, and ensure that better-performing hparams are assigned more often
- build connectors, allowing people to integrate their nodes with personal data
- [Soft Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745)?
- [Mixture of a Million Experts](https://arxiv.org/abs/2407.04153)?
- [Mixture of Depths](https://arxiv.org/abs/2404.02258).

## todo

- a proper and robust DHT
- central and persistent relay peers, to act as global initial_peers
- a routing algorithm with multi-hop support (ping -> pang -> pong -> ping)
- multi-layer experts
- peer validation
- self-modeling of remote experts

## won't do

- cryptocurrency ([donations](https://www.patreon.com/fold) are appreciated, though!)