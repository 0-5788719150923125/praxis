# thorns

Triangulated Human Observation for Reasoning in the Natural Sciences

## what this model aims to be

We are trying to build a decentralized, peer-to-peer, always online, always learning LLM - with Hivemind integrated into the model itself. We will do this via a mixture of experts, sparse routing, and by replacing feedforward layers with remote experts.

## install

The model: `pip install .`

Everything: `pip install .[all]`

## ideas

- a GLOBAL swarm, by default
- replace dense layers with remote experts, then leverage [self-modeling](https://arxiv.org/abs/2407.10188) to train the network to model those peers
- make experts deeper; if an expert is comprised of multiple transformer blocks, rather than a single dense layer, then the network could learn to dynamically-route through deeper networks, or it could learn to relay information across multiple peers, or it could learn that "no relay is needed" at all, simply returning simple predictions back to the requestee. For example, let's say that every peer runs a 6-layer transformer, comprised of three experts - with 2 layers each. By routing through multiple peers, one could theoretically turn a 6-layer transformer into 600 layers (if routing between tons of peers). Obviously, 600 layers would be tough to optimize, but if each peer kept optimization local - that's just 6 layers to optimize. Each peer becomes highly specialized in "something."
- treat every peer as an experiment in a massive hyperparameter search; publish results to the DHT, and ensure that better-performing hparams are assigned more often
- ship with an API server; if you run a node, you also get a local API that you can use for inference
- build connectors, allowing people to specialize their nodes on their own data
- [Mixture of a Million Experts](https://arxiv.org/abs/2407.04153)?