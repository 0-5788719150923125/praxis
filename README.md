# praxis

<!-- Triangulated Human Observation for Reasoning in the Natural Sciences -->

_Praxis is the process by which a theory, lesson, or skill is enacted, embodied, realized, applied, or put into practice._

![Terminal](./static/terminal.webp)

## what we're building

The Praxis swarm is a decentralized, peer-to-peer, always online, continuously-learning AI intelligence - and a place to practice [computational alchemy](https://www.reddit.com/r/MachineLearning/comments/1b6ggpz/comment/ktc2ujd). With [Hivemind](https://github.com/learning-at-home/hivemind) directly-integrated into core layers of the model itself, our goal is to build an expert model that is small and simple, easy to parallelize and performant at a scale of hundreds/thousands of peers. We will do this via a sparse mixture of experts, curated routing, algorithmic switching and weighted self-modeling of remotely-hosted peers.

## architecture

- A [Mixture of Depths](https://arxiv.org/abs/2404.02258) allows us to route just a subset of all tokens in a sequence to remote peers - reducing the time required for remote computation, and the amount of data transferred.
- [LayerShuffle](https://arxiv.org/abs/2407.04513) proved that transformers can maintain coherence, even when every layer is shuffled at every forward pass. We take this a step further, and implement the `PraxisController`, which teaches the model how to predict an optimal route through expert layers during inference. The ability to work with out-of-order layers is crucial in a decentralized architecture, where some peers may fail, others may disappear, some may be overloaded, or undertrained, or are otherwise penalized for some reason or another...
- In addition to the shuffling, we implement a simplified version of [CALM](https://arxiv.org/abs/2207.07061), which allows the model to early-exit from computation.
- We implement RoPE, ALiBi and NoPE as options for positional encoding, because they're easy, work well at sane contexts lengths, and require little to no trainable parameters.
- [Differential Attention](https://arxiv.org/abs/2410.05258) is used to improve hallucination performance, reduce parameter counts required for attention, and filter-out noise in attention maps. Alternatively (and perhaps in-addition to, in the future), we implement an option for [Stickbreaking Attention](https://arxiv.org/abs/2306.04640), which naturally-encodes positional information, uses a Sigmoid-based mechanism, instead of a Softmax (i.e. parameters "work together", instead of "competing" against each other).
- Parameter-Efficient Expert Retrieval (PEER) from the [Mixture of a Million Experts](https://arxiv.org/abs/2407.04153) paper. In this design, dense feedforward layers are replaced with singleton Multi-Layer Perceptron networks.
- While simple, a [Soft-Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745) class allows us to dynamically-route through a dense feedforward layer, while maintaining differentiability and enhancing expressivity.
- We also implement a simplistic KNN-based external memory module, akin to [Memorizing Transformers](https://arxiv.org/abs/2203.08913).
- There's also a mobile app, and a remote controller, called "Axis". We used [Godot](https://godotengine.org/) for that.

## join us

- [Discord](https://discord.gg/8ZmHP8CqUX)
- [The Source](https://src.eco)

## install

From a Linux shell, run these commands:

```sh
# Setup a virtual environment
source make-venv.sh

# Install core model dependencies
pip install -e .

# Install training dependencies
pip install -e .[all]
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

## recommendations

We recommend you use a `batch_size` of at least 16, if possible:

```sh
python run.py --batch_size 16
```

The reason for this is that we have implemented an oversampling mechanism, which can expose your model to longer sequences during training (improving generalization and maximum supported sequence length). This oversampling mechanism periodically doubles the sequence length, and scales quadratically at batch sizes of 1, 4, and 16.

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
    depth=6,
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
- peer validation (zero knowledge proofs, differential privacy)
- [T-FREE Tokenizer](https://github.com/aleph-alpha/trigrams)
- [Mini-Sequence Transformer](https://github.com/wdlctc/mini-s/tree/main) (probably not worth it on the smaller scale; this is basically the same thing as Infini-Attention, but implemented in a more-naive way)
- embed training code within the model architecture itself, such that loading a model automatically starts the training, as well
- cascading assistant models via hivemind (speculative decoding)
- [TokenMonster](https://github.com/alasdairforsythe/tokenmonster)
- [Linear Recurrent Units](https://arxiv.org/abs/2303.06349) (not recommended; they are extremely slow without specialized kernels)
- [DualFormer](https://arxiv.org/html/2410.09918v1) (this one would be tough to do, because it seems to require detailed reasoning steps and structured trace dropping; i.e. data we don't have); [this dataset might work](https://huggingface.co/datasets/thesven/gsm8k-reasoning)
- [novel activations](https://gist.github.com/ronaldoaf/427887efe44f12d4bdccc46ad73404eb)
- [xLSTM](https://github.com/NX-AI/xlstm)
- Denny Zhou (Founded & lead reasoning team at Google DeepMind) - "We have mathematically proven that transformers can solve any problem, provided they are allowed to generate as many intermediate reasoning tokens as needed. Remarkably, constant depth is sufficient." [source](https://www.reddit.com/r/mlscaling/comments/1fijajw/denny_zhou_founded_lead_reasoning_team_at_google/) At first pass, this may sound stupid, because everyone knows that transformers are "universal function approximators" already; the problem is that the search space becomes so large as to be computationally infeasible. However, the more-meaningful takeaway here is this: with human guidance (i.e. prompting, iteration, development, persistence, re-focusing), a human/AI duo could solve any problem... including AGI.
- [Differentiable Lambda Calculus](https://github.com/neurallambda/neurallambda) (i.e. symbolic reasoning)
- the way that Mixture of Depths token indices interact with ALiBi is potentially interesting and worth evaluating.
- [Simple Recurrent Units](https://gist.github.com/calclavia/bb64b2f9dd3920ff6ad9546a606718e1)
- [Forward-mode automatic differentiation](https://pytorch.org/tutorials/intermediate/forward_ad_usage.html)
- [Human-like Episodic Memory for Infinite Context LLMs](https://arxiv.org/abs/2407.09450) (we tried a version of this, located at misc/episodic_memory.py, but it was terrible, slow, didn't work, used a ton of memory, and was horribly complex)
- [Facts as Experts: Adaptable and Interpretable Neural Memory over Symbolic Knowledge](https://arxiv.org/abs/2007.00849)
- https://github.com/timurgepard/nanoFFT (a somewhat novel idea)
- [TokenFormer](https://arxiv.org/abs/2410.23168)
- [Funnel Transformers](https://arxiv.org/pdf/2006.03236) for sequence compression

## won't do

- cryptocurrency ([donations](https://www.patreon.com/fold) are appreciated, though!)
- We implemented a simplified version of Infini-Attention, from [Leave No Context Behind](https://arxiv.org/abs/2404.07143), but found it to be ineffective. Particularly, we were looking for long-term memory, whereas this is primarly used for extremely long context lengths (i.e. "memories" do not persist between forward passes).
