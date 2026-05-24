# praxis

_Praxis is the process by which a theory, lesson, or skill is enacted, embodied, realized, applied, or put into practice._

---

![Terminal](./static/terminal.webp)

---

## description

The Praxis platform is an ever-evolving, local-first, peer-to-peer, flexible, modular, extensible and decentralized framework for the practice of [computational alchemy](https://www.reddit.com/r/MachineLearning/comments/1b6ggpz/comment/ktc2ujd). We are building a multi-modal fleet of AI agents that are small and simple, easy to parallelize, fault-tolerant, portable, and performant at any scale. We will achieve this via a remote mixture of experts, user-weighted multipath routing, symbolic decision-making and prayer (asynchronous).

<details>

<summary>features</summary>

<!-- AUTODOC:FEATURES:BEGIN -->

Praxis is organized as ~20 pluggable registries. Each category below
links to a docs page listing the concrete implementations and their
source. See [docs/index.md](docs/index.md) for the full map.

- [Activation functions](docs/activations.md) (31)
- [Attention mechanisms](docs/attention.md) (7)
- [Decoder block layouts](docs/blocks.md) (8)
- [Sequence compression](docs/compression.md) (3)
- [Layer-routing controllers](docs/controllers.md) (7)
- [Data sampler strategies](docs/data.md) (5)
- [Block-stacking decoders](docs/decoders.md) (4)
- [Token embeddings](docs/embeddings.md) (8)
- [Input encoders](docs/encoders.md) (8)
- [Positional encoding](docs/encoding.md) (3)
- [Halting / early exit](docs/halting.md) (2)
- [Output heads](docs/heads.md) (6)
- [Loss functions](docs/losses.md) (8)
- [Normalization layers](docs/normalization.md) (5)
- [Expert orchestration](docs/orchestration.md) (7)
- [RL policies](docs/policies.md) (3)
- [Residual connections](docs/residuals.md) (2)
- [Token routers](docs/routers.md) (11)
- [Sequence sorting](docs/sorting.md) (3)
- [Training strategies](docs/strategies.md) (4)

<!-- AUTODOC:FEATURES:END -->

</details>

<details>

<summary>installation, configuration, and usage</summary>

The launch script will automatically manage the virtual environment. To start, simply run:

```sh
./launch
```

To view all supported command-line arguments:

```sh
./launch --help
```

For development with quick iteration:

```sh
./launch --dev --no-dashboard
```

To run inside Docker (auto-detects ARM vs x86_64; forwards all subsequent flags):

```sh
./launch compose --alpha
```

To tear a running compose stack back down:

```sh
./launch stop
```

To run unit testing:

```sh
pytest tests -x
```

## recommendations

We recommend you use a `batch-size` of at least 16, if possible. We have implemented an oversampling mechanism, which periodically multiplies your sequence length, and scales quadratically with batch sizes of 1, 4, 16, 64, etc.

We also recommend using an Nvidia GPU.

```sh
./launch --batch-size 16 --device cuda
```

To apply [alpha](experiments/alpha.yml) settings to your [experiment](experiments/README.md):

```sh
./launch --alpha
```

To apply [dev](environments/dev.yml) settings within your [environment](environments/README.md), with device overrides:

```sh
./launch --alpha --dev --device cuda:8 --quiet
```

</details>

<details>

<summary>showcase</summary>

## do inference

### String-based generation

For simple string prompts, send a JSON-encoded payload via POST to:

```
http://localhost:2100/input
```

Example request:

```py
import requests

url = "http://localhost:2100/input"
payload = {"prompt": "Once upon a time, ", "do_sample": True, "temperature": 0.7}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

### Message-based generation (recommended)

For conversation-style generation with support for system prompts and tool calls, use:

```
http://localhost:2100/messages
```

Example request:

```py
import requests

url = "http://localhost:2100/messages"
payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "do_sample": True,
    "temperature": 0.7
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

Both endpoints support all arguments in the [Transformers text generation API](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

## local web chat

Chat and swarm management interface is available here:

```
http://localhost:2100
```

![Praxis Chat](./static/chat.png)

## mobile app

We're building a mobile app, to control your experts! You can see that code in the [./axis](./axis) directory.

## to register with transformers

```py
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

config = PraxisConfig(
    embed_size=512,
    hidden_size=384,
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

</details>

<details open>

<summary>won't do</summary>

- cryptocurrency

</details>

<details open>

<summary>community</summary>

- [Demo](https://arc.src.eco)
- [Discord](https://discord.gg/bp3SuFae5M)
- [Youtube](https://youtube.com/@The-Arc)
- [The Source](https://src.eco)

</details>

## limitations

- You will quickly run into rate limits with the Huggingface Datasets API. This is because anonymous users tend to be bots, and are subjected to severe restrictions. To alleviate this problem, you can install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli), and authenticate with a real user account. Praxis will use this user automatically.
