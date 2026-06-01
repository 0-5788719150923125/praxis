# World Models: Approximating the Whole Thing Across a Consenting Swarm

> Status: **vision / unscoped** (2026-06-01). The north star: build a full world
> model whose capacity lives not on one machine but across a fleet of consenting
> peers, every one of them gated behind the Praxis web app and the browser
> sandbox. Grounded in three things that already run - the Mono-Forward trainer
> (`praxis/trainers/mono_forward/`), the bias-variance-as-orthogonal-axes framing
> from the paper (`research/main.tex`), and the GUN swarm extension at `../fvn`.
> Companion to [oscillatory_axes.md](oscillatory_axes.md), [harmony.md](harmony.md),
> and the scaling conjecture in the paper.

## The thesis, stated plainly

We are trying to **approximate a whole world model** - not a chatbot, a model of
how things are - and we will do it by *renting the unused compute of people who
consent to help*. The agents we plant are tiny. They run in a browser tab, or
persistently in a browser extension (the `../fvn` pattern: a VSCode/GUN extension
that joins a peer swarm, advertises RPS, and shows online/joining/offline state).
The Praxis web app is the **permission boundary and the console**: every local or
remote peer is forever gated behind it, so user consent and the browser sandbox
are what keep everyone safe. Nobody runs arbitrary native code on a stranger's
machine. They run a sandboxed agent the user opted into, visible in a dashboard,
revocable at any time.

The bet underneath: a world model does not have to be one artifact you download.
It can be a **standing approximation** that the network assembles on demand,
streaming in the pieces it needs, when it needs them. GUN is built for exactly
this - it is a graph database whose native verb is the *streaming update*: peers
subscribe to nodes and receive changes as they happen. Weights that arrive as a
stream of updates are a natural fit for a protocol that was designed to sync
graphs of streaming state.

And GUN runs **entirely in the browser**. There is no Praxis-run server in the
data path: peers find each other through remote STUN/TURN and sync directly,
browser to browser. The Praxis web app's existing websocket (socket.io, already
carrying the live metrics + reload streams) is the *local* bridge - the Python
training process talks to the in-page GUN peer over that same efficient streaming
interface, on localhost. So the only thing we host is the app the user already
chose to open; the mesh itself is serverless from our side. That strengthens the
safety story rather than complicating it: there is no central relay to trust,
compromise, or subpoena. The "server" is remote STUN/TURN we do not own, and the
sandbox boundary is the browser's.

## Why "approximate" is the load-bearing word

The paper's central move is that in a harmonic latent space, **bias and variance
stop being a single dial** and become orthogonal coordinates. A static spectrum
is pure bias - the corpus-average rhythm, one beat imposed on everything. An
input-conditional modulation of that spectrum is structured variance - deviation
that still respects the learned harmonic constraints. Because they live on
separate axes with separate gradients, you can lower both at once; the U-shaped
tradeoff becomes a 2D manifold you have coordinates on.

That reframing is what makes a *streamed, partial, distributed* world model
tractable rather than reckless:

- **The static prior is small and shared.** The corpus-average rhythm - the
  bias term - is a compact object every peer can hold. It is the low-frequency
  band that crystallizes in the spectrum heatmap; it does not need to stream.
- **The variance is what streams.** The input-conditional delta - the part that
  specializes to *this* context - is exactly the part you fetch on demand, from
  whichever experts are best positioned to supply it. You stream variance against
  a stable, resident bias.
- **Partial is principled, not a degradation.** A model missing some of its
  streamed variance is not broken; it has fallen back toward the prior. It
  answers from the mean until the specializing modes arrive. There is a graceful,
  *bounded* degradation guarantee here precisely because the architecture
  separates the always-present bias from the streamed-in variance. "Approximate
  the whole thing" means: hold the prior everywhere, stream the corrections.

This is the performance guarantee the bias/variance framing buys us. A consensus
model that stores everything as a mean has no such fallback - drop part of it and
you drop part of the answer. A harmonic model degrades toward its prior, which is
the best constant guess, and recovers as interference modes stream in. The
scaling conjecture (consensus $O(N)$, interference $O(\log N)$) says the modes
worth streaming are few: $K \sim \log N$ components address exponentially many
configurations, so the variance channel is *thin*. You are streaming logarithmic
corrections onto a linear-cheap prior.

## The substrate we already built

We did not start this from nothing. `MonoForwardTrainer` is the single-host
rehearsal of the whole architecture. It runs one `LayerActor` per `LocalLayer`,
and each actor:

- owns its layer's weights **plus its own projection `M_i`** to the label space
  (paper §3.1 - no shared head, no head sync),
- holds a **local optimizer** and trains a local goodness/cross-entropy loss
  against the next-token objective,
- exchanges **only activation tensors** with its neighbors - gradients never
  cross a layer boundary, so activation memory is O(1) in depth.

A transformer trained with local, layer-wise losses needs **no global backward
pass**. Once that's true, the layers stop being a monolith and become a fleet of
workers that pass activations, not gradients - and the transport between them is
a free variable. Swap "Ray actor on a GPU" for "sandboxed agent in a peer's
browser" and "ObjectRef" for "GUN streaming update," and the topology is
unchanged. The protocol boundary is the only thing that moves.

This is why the vision is engineering, not fantasy: the training math that makes
distribution possible is running today.

## The agents: tiny, sandboxed, persistent

An agent is a single small layer-expert. At `hidden_dim = 16` it is so small that
it can live inside a browser tab without bothering its host, train its local
goodness loss on a data shard, and serve activations on demand. The `../fvn`
extension shows the persistence story: a browser/editor extension that joins a
GUN bootstrap swarm, hosts a local API, survives across sessions, and reports its
own health to a swarm view. An agent planted that way is **always-on but always
consensual** - it is the user's extension, doing the user's opted-in work,
visible and killable.

The agents are *not redundant copies*. Each trained on its own shard with its own
init and its own local loss, so the pool is a set of **diverse bases**, not
replicas. That diversity is the product - it is what lets the buyer compose a
representation richer than any single tiny model, by mixing many idiosyncratic
narrow experts.

## The rank-priced connection

The economic primitive is a **low-rank channel** to an agent. A `hidden_dim = 16`
agent can sell a **rank-4 connection**: it projects its activation down to `r = 4`
before sending, `z = P a` with `P ∈ ℝ^{4×16}`. Cheaper for the agent (less to
send, less it reveals of its shard), cheaper for the buyer (less to mix), and the
rank is a literal, meterable price knob - "buy rank 4 of agent X" is a concrete
transaction. A swarm of hundreds of tiny agents sold at small rank composes into
an arbitrarily wide distributed model: you assemble the width you can afford out
of many narrow, cheap contributions.

The projection `P` is a **learned interface**, not blind compression - it can be
trained so the rank-`r` channel carries the most useful `r` directions, the same
move CALM makes compressing a patch into a latent. Here the latent is the billing
boundary, and - tying back to the streaming story - the rank-`r` stream *is* the
variance channel: a thin, on-demand correction on top of the resident prior.

## Mixing: mean, vote, or explore

Given rank-`r` channels from a pool, the buyer needs a mixing operator. Three
regimes, each already echoed in Praxis:

1. **Distributed mixture-of-experts.** A router weights and sums the agents'
   contributions - classic MoE, but the experts live on other people's machines.
2. **Expert voting (the CALM approach).** Each agent votes over the next-token
   distribution; aggregate. Robust to any single bad or absent agent - which is
   also the graceful-degradation property: a missing voter just lowers the
   resolution of the variance, it does not break the answer.
3. **Stochastic sampling.** Each step, sample a subset of the pool. The expected
   mix is the pool mean, but the **variance is the point** - sampling visits
   representations *off* the mean, excursions into geometric or harmonic space
   that a deterministic average smooths away. The pool converges to its mean;
   sampling is how you orbit the shell around it.

This is the bias-variance manifold again, now at the level of *which experts you
query*. Sit at the mean (vote/average) for robustness; orbit it (sample) for
exploration. Temperature, in the paper's sense, becomes a position dial over the
swarm, not noise added at a point.

## What has to get built

Roughly in dependency order. None of this is scoped; it is a sketch.

### The permission plane (first, because it is the safety story)

The Praxis web app is the gate. Before any compute crosses a machine boundary,
there has to be: a **consent surface** (what am I contributing, to whom, how do I
stop), a **sandbox guarantee** (the agent is browser-sandboxed JS, never native),
and an **identity/auth layer below the app** for joining the mesh and signing
"I sold you rank 4" (GUN's SEA is the substrate). The dashboard is the operator's
console; identity and keys live a layer beneath it. Without this plane there is
no consenting swarm - just an open relay, which we will not build.

### The in-browser GUN peer + websocket bridge (no sidecar)

No Node sidecar. GUN runs in the page itself: the in-browser peer joins the mesh
(remote STUN/TURN for discovery), subscribes to the relevant graph nodes, and
brokers activation/weight **streams** browser-to-browser. The Python training
process reaches it through the web app's **existing websocket** (socket.io,
already running for live metrics + reload) - the local bridge between Python and
the in-page peer, on localhost. This replaces the Ray transport in the
Mono-Forward trainer with a path that has *zero* Praxis-run server in it: we host
only the app the user opened, and the mesh is serverless from our side.

### The in-browser agent

A tiny layer-expert that trains its local loss and serves rank-`r` activations.
**Resolved: pure JS, no framework.** The framework question is settled - a
hand-written reverse-mode autograd + transformer (exact SiLU/RMSNorm/softmax,
verified against finite differences) is small, auditable, and dependency-free,
which is exactly what we want for a ~14-dim layer. It now lives in the app at
`praxis/web/src/js/nanoformer.js` (regression test in `__tests__/gradcheck.js`),
and the Stage-tab swarm agents (`swarm.js`) run it: a heartbeat forward pass
while idle, and a real Mono-Forward `trainLayerWise` step once a transport drives
work. TF.js was not needed. The persistence/health model can borrow from `../fvn`.

### The rank-priced streaming protocol

Advertise capacity, negotiate rank, **stream** activations/weight-deltas as GUN
updates, account for what was delivered, score it (loss delta when an agent is
included vs not) so the buyer stops paying for dead weight. This is where "buy
rank 4 of agent X" becomes a concrete, metered, reputation-bearing handshake.

## Honest unknowns

- **Latency and stragglers.** Mono-Forward pipelines to hide per-layer latency on
  one host; a global mesh adds wide-area RTT and flaky peers. The graceful
  fallback-to-prior helps (you answer from bias while variance is in flight) and
  stochastic sampling helps (you only wait for the subset you sampled), but the
  real fill behavior needs measuring.
- **Incentive integrity.** What stops an agent selling noise? Voting/sampling
  tolerate *some* bad actors; a real market needs the buyer's include-vs-exclude
  loss score as a reputation signal, so dead weight stops getting paid.
- **Privacy of the projection.** Rank-`r` leaks less than full activations, but
  "less" isn't "none." Understand what a buyer can reconstruct about an agent's
  shard from its rank-4 stream before promising privacy.
- **Does diversity actually compose?** The whole bet is that heterogeneous tiny
  bases beat one model of equal total width. That is an empirical claim testable
  *today*, single-host - and it is the hinge the entire vision swings on.

## The cheapest first probe (no network required)

Before any browser code, any consent UI, any mesh: **simulate the swarm
locally.** Train a handful of tiny `LocalLayer` actors (the existing Mono-Forward
path) on disjoint shards, each with its own `M_i`, then add a buyer-side mixing
layer that consumes **rank-`r` projections** of their activations and tries one of
the three mixing operators. Two questions fall out, both answerable on one
machine:

1. *Do diverse low-rank experts compose?* Does a pool of rank-4 tiny experts,
   mixed by a learned router, beat a single equal-width baseline?
2. *Does the bias/variance fallback hold?* Drop a fraction of the experts
   mid-inference and confirm the answer degrades *toward the prior* (bounded,
   graceful) rather than breaking - the streaming guarantee, measured.

If both hold, the network engineering is worth it. If they don't, we learned it
cheaply, with code we already have. The global network is the thing - but the
physics it rests on is testable before a single byte crosses a machine boundary.
