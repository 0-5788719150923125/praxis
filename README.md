# praxis

<!-- Triangulated Human Observation for Reasoning in the Natural Sciences -->

_Praxis is the process by which a theory, lesson, or skill is enacted, embodied, realized, applied, or put into practice._

![Terminal](./static/terminal.webp)

## what we're building

The Praxis swarm is a decentralized, peer-to-peer, always online, and continuously-learning AI intelligence - with [Hivemind](https://github.com/learning-at-home/hivemind) directly-integrated into core layers of the model itself. The goal is to build an expert model that is small and simple, easy to parallelize and performant at a scale of hundreds/thousands of peers. We will do this via a sparse mixture of experts, curated routing, algorithmic switching and weighted self-modeling of remotely-hosted peers.

## architecture

- A [Mixture of Depths](https://arxiv.org/abs/2404.02258) allows us to route just a subset of all tokens in a sequence to remote peers - reducing the time required for remote computation, and the amount of data transferred.
- [LayerShuffle](https://arxiv.org/abs/2407.04513) proved that transformers can maintain coherence, even when every layer is shuffled at every forward pass. ~~We take this a step further, and implement a controller that predicts an optimized layer order.~~ The ability to work with out-of-order layers is crucial in a decentralized architecture, where some peers may fail, others may disappear, some may be overloaded, or undertrained, or are otherwise penalized for some reason or another...
- [Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) for length extrapolation, because it's easy, it works well at sane contexts lengths, and it requires no trainable parameters.
- [Differential Attention](https://arxiv.org/abs/2410.05258) is used to improve hallucination performance, reduce parameter counts required for attention, and filter-out noise in attention maps.
- Parameter-Efficient Expert Retrieval (PEER) from the [Mixture of a Million Experts](https://arxiv.org/abs/2407.04153) paper. In this design, dense feedforward layers are replaced with singleton Multi-Layer Perceptron networks.

## join us

- [Discord](https://discord.gg/8ZmHP8CqUX)
- [The Source](https://src.eco)

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
python run.py
```

To view all supported command-line arguments:

```sh
python run.py --help
```

## do inference

Send a JSON-encoded payload via POST to:

```
http://localhost:2100/input
```

This payload should support all arguments in the [Transformers text generation API](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

Example request:

```py
import requests

url = "http://localhost:5000/input"
payload = {"prompt": "Once upon a time, ", "do_sample": True, "temperature": 0.7}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

## local web chat (coming soon!)

Chat and swarm management interface is available here:

```
http://localhost:2100
```

## to register with transformers

```py
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

config = PraxisConfig(
    num_embeds=512,
    num_dims=384,
    num_layers=6,
    num_heads=8,
    device_map="cuda:0",
)

tokenizer_model = "UNSAFE/praxis-4096"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

model = AutoModelForCausalLM.from_config(config)

input_ids = tokenizer.encode("The quick brown fox ")

outputs = model.generate(input_ids, do_sample=True)

print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
# --> The quick brown fox jumped over a lazy dog.
```

## goals

- a global swarm
- [self-modeling](https://arxiv.org/abs/2407.10188) makes peers less-complex, and easier to model (for other AI)
- layers as experts; a marketplace of expert, composable "blocks"
- commit to yourself
- cascade-style token routing (ping -> pang -> pong -> ping) via a Mixture of Depths; cyclical graph computation
- treat every peer as an experiment in hyperparameter search; publish results to the DHT, and ensure that well-performing hparams are assigned more often
- build adapters/connectors, allowing people to integrate their nodes with external data sources

## notes, ideas and random things I want to remember

- a proper and robust DHT
- central and persistent relay peers, to act as global bootstrap nodes
- helix, octopi, pyramids
- multi-block, heirarchical experts
- peer validation (zero knowledge proofs)
- self-modeling of remote experts
- [Soft Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745)
- [T-FREE Tokenizer](https://github.com/aleph-alpha/trigrams)
- [Mini-Sequence Transformer](https://github.com/wdlctc/mini-s/tree/main) (probably not worth it on the smaller scale)
- embed training code within the model architecture itself, such that loading a model automatically starts the training, as well
- cascading assistant models via hivemind (speculative decoding)
- [Human-like Episodic Memory for Infinite Context LLMs](https://arxiv.org/abs/2407.09450)
- [TokenMonster](https://github.com/alasdairforsythe/tokenmonster)
- [Linear Recurrent Units](https://arxiv.org/abs/2303.06349)
- [DualFormer](https://arxiv.org/html/2410.09918v1) (this one would be tough to do, because it seems to require detailed reasoning steps and structured trace dropping)

## won't do

- cryptocurrency ([donations](https://www.patreon.com/fold) are appreciated, though!)
- half-precision; [there is research](https://sambanova.ai/blog/alibi-interpolation-vs-extrapolation) that suggests a detrimental interaction between ALiBi and low-precision. So, until we have replaced ALiBi - operations use full (32-bit) precision for now.
