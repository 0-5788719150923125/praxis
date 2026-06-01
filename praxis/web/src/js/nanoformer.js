// nanoformer.js - a complete causal transformer in pure JavaScript, no deps.
//
// The model behind the in-browser swarm agent (see next/world_models.md and
// swarm.js): at hidden_size~16 a full framework (TF.js) is overkill, so this is
// a tiny reverse-mode autograd (matrix-granularity) with a transformer on top.
// The math is exact - SiLU, RMSNorm, softmax attention - not approximated, and
// verified against finite differences (see __tests__/gradcheck.js).
//
// Two training modes:
//   * trainGlobal()    - end-to-end backprop through the last layer's head.
//   * trainLayerWise() - Mono-Forward: each layer owns a projection M_i and a
//                        local optimizer, trains its own next-token loss, and
//                        gradients never cross a layer boundary (the property
//                        that makes the layers distributable across the swarm).

// ----------------------------------------------------------------------------
// Autograd engine. A node `T` is a [rows x cols] matrix with data + grad and a
// closure that pushes its output-gradient back into its parents. backward()
// runs a topo sort from a scalar and accumulates grads via +=.
// ----------------------------------------------------------------------------

class T {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    this.data = data || new Float64Array(rows * cols);
    this.grad = new Float64Array(rows * cols);
    this._backward = () => {};
    this._prev = [];
  }
}

function backward(root) {
  const topo = [];
  const seen = new Set();
  (function build(v) {
    if (seen.has(v)) return;
    seen.add(v);
    for (const p of v._prev) build(p);
    topo.push(v);
  })(root);
  root.grad[0] = 1; // seed d(loss)/d(loss) = 1 (root is a scalar)
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}

// Stop gradient: a fresh leaf carrying the same numbers, no parents.
function detach(A) {
  return new T(A.rows, A.cols, A.data.slice());
}

// C = A @ B
function matmul(A, B) {
  const C = new T(A.rows, B.cols);
  const { rows: m, cols: k } = A;
  const n = B.cols;
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A.data[i * k + p] * B.data[p * n + j];
      C.data[i * n + j] = s;
    }
  C._prev = [A, B];
  C._backward = () => {
    for (let i = 0; i < m; i++)
      for (let p = 0; p < k; p++) {
        let ga = 0;
        for (let j = 0; j < n; j++) ga += C.grad[i * n + j] * B.data[p * n + j];
        A.grad[i * k + p] += ga;
      }
    for (let p = 0; p < k; p++)
      for (let j = 0; j < n; j++) {
        let gb = 0;
        for (let i = 0; i < m; i++) gb += A.data[i * k + p] * C.grad[i * n + j];
        B.grad[p * n + j] += gb;
      }
  };
  return C;
}

// Add a [1 x cols] bias row to every row of A.
function addRow(A, b) {
  const C = new T(A.rows, A.cols, A.data.slice());
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < A.cols; j++) C.data[i * A.cols + j] += b.data[j];
  C._prev = [A, b];
  C._backward = () => {
    for (let i = 0; i < A.rows; i++)
      for (let j = 0; j < A.cols; j++) {
        const g = C.grad[i * A.cols + j];
        A.grad[i * A.cols + j] += g;
        b.grad[j] += g;
      }
  };
  return C;
}

// Elementwise A + B (same shape).
function add(A, B) {
  const C = new T(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) C.data[i] = A.data[i] + B.data[i];
  C._prev = [A, B];
  C._backward = () => {
    for (let i = 0; i < A.data.length; i++) {
      A.grad[i] += C.grad[i];
      B.grad[i] += C.grad[i];
    }
  };
  return C;
}

// Elementwise A * B (same shape).
function mul(A, B) {
  const C = new T(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) C.data[i] = A.data[i] * B.data[i];
  C._prev = [A, B];
  C._backward = () => {
    for (let i = 0; i < A.data.length; i++) {
      A.grad[i] += B.data[i] * C.grad[i];
      B.grad[i] += A.data[i] * C.grad[i];
    }
  };
  return C;
}

function scale(A, s) {
  const C = new T(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) C.data[i] = A.data[i] * s;
  C._prev = [A];
  C._backward = () => {
    for (let i = 0; i < A.data.length; i++) A.grad[i] += s * C.grad[i];
  };
  return C;
}

function transpose(A) {
  const C = new T(A.cols, A.rows);
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < A.cols; j++) C.data[j * A.rows + i] = A.data[i * A.cols + j];
  C._prev = [A];
  C._backward = () => {
    for (let i = 0; i < A.rows; i++)
      for (let j = 0; j < A.cols; j++) A.grad[i * A.cols + j] += C.grad[j * A.rows + i];
  };
  return C;
}

// SiLU / swish: x * sigmoid(x). Exact, no approximation.
function silu(A) {
  const C = new T(A.rows, A.cols);
  const sig = new Float64Array(A.data.length);
  for (let i = 0; i < A.data.length; i++) {
    const x = A.data[i];
    const s = 1 / (1 + Math.exp(-x));
    sig[i] = s;
    C.data[i] = x * s;
  }
  C._prev = [A];
  C._backward = () => {
    for (let i = 0; i < A.data.length; i++) {
      const s = sig[i];
      const x = A.data[i];
      A.grad[i] += (s + x * s * (1 - s)) * C.grad[i]; // d/dx [x*sigmoid(x)]
    }
  };
  return C;
}

// RMSNorm per row with a learned gain g [1 x cols]. Exact backward.
function rmsNorm(A, g, eps = 1e-5) {
  const C = new T(A.rows, A.cols);
  const n = A.cols;
  const r = new Float64Array(A.rows); // per-row 1/rms
  for (let i = 0; i < A.rows; i++) {
    let ms = 0;
    for (let j = 0; j < n; j++) ms += A.data[i * n + j] ** 2;
    ms /= n;
    const ri = 1 / Math.sqrt(ms + eps);
    r[i] = ri;
    for (let j = 0; j < n; j++) C.data[i * n + j] = A.data[i * n + j] * ri * g.data[j];
  }
  C._prev = [A, g];
  C._backward = () => {
    for (let i = 0; i < A.rows; i++) {
      const ri = r[i];
      // dot = sum_j gY_j * g_j * x_j  (drives the shared 1/rms term)
      let dot = 0;
      for (let j = 0; j < n; j++)
        dot += C.grad[i * n + j] * g.data[j] * A.data[i * n + j];
      for (let j = 0; j < n; j++) {
        const x = A.data[i * n + j];
        const gy = C.grad[i * n + j];
        A.grad[i * n + j] += g.data[j] * ri * gy - (ri ** 3 * x / n) * dot;
        g.grad[j] += gy * x * ri;
      }
    }
  };
  return C;
}

// Row-wise causal softmax: row i normalizes over columns j <= i, rest are 0.
function causalSoftmax(S) {
  const C = new T(S.rows, S.cols);
  const n = S.cols;
  for (let i = 0; i < S.rows; i++) {
    let mx = -Infinity;
    for (let j = 0; j <= i; j++) mx = Math.max(mx, S.data[i * n + j]);
    let sum = 0;
    for (let j = 0; j <= i; j++) {
      const e = Math.exp(S.data[i * n + j] - mx);
      C.data[i * n + j] = e;
      sum += e;
    }
    for (let j = 0; j <= i; j++) C.data[i * n + j] /= sum;
  }
  C._prev = [S];
  C._backward = () => {
    for (let i = 0; i < S.rows; i++) {
      let dotgp = 0; // sum_k p_k * g_k over allowed entries
      for (let j = 0; j <= i; j++) dotgp += C.data[i * n + j] * C.grad[i * n + j];
      for (let j = 0; j <= i; j++) {
        const p = C.data[i * n + j];
        S.grad[i * n + j] += p * (C.grad[i * n + j] - dotgp);
      }
    }
  };
  return C;
}

// Mean cross-entropy over rows. logits [Tlen x V], targets int[]. Returns a
// scalar T; backward fills logits.grad with (softmax - onehot)/Tlen.
function crossEntropy(logits, targets) {
  const Tlen = logits.rows;
  const V = logits.cols;
  const out = new T(1, 1);
  const probs = new Float64Array(Tlen * V);
  let loss = 0;
  for (let i = 0; i < Tlen; i++) {
    let mx = -Infinity;
    for (let j = 0; j < V; j++) mx = Math.max(mx, logits.data[i * V + j]);
    let sum = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(logits.data[i * V + j] - mx);
      probs[i * V + j] = e;
      sum += e;
    }
    for (let j = 0; j < V; j++) probs[i * V + j] /= sum;
    loss += -Math.log(probs[i * V + targets[i]] + 1e-12);
  }
  out.data[0] = loss / Tlen;
  out._prev = [logits];
  out._backward = () => {
    const s = out.grad[0] / Tlen;
    for (let i = 0; i < Tlen; i++)
      for (let j = 0; j < V; j++)
        logits.grad[i * V + j] += s * (probs[i * V + j] - (j === targets[i] ? 1 : 0));
  };
  return out;
}

// Gather embedding rows for a list of token ids. table [V x d] -> [len x d].
function embedRows(table, ids) {
  const d = table.cols;
  const C = new T(ids.length, d);
  for (let i = 0; i < ids.length; i++)
    for (let j = 0; j < d; j++) C.data[i * d + j] = table.data[ids[i] * d + j];
  C._prev = [table];
  C._backward = () => {
    for (let i = 0; i < ids.length; i++)
      for (let j = 0; j < d; j++) table.grad[ids[i] * d + j] += C.grad[i * d + j];
  };
  return C;
}

// Slice the first `len` rows (the learned positional rows 0..len-1).
function rowsSlice(table, len) {
  const d = table.cols;
  const C = new T(len, d);
  for (let i = 0; i < len; i++)
    for (let j = 0; j < d; j++) C.data[i * d + j] = table.data[i * d + j];
  C._prev = [table];
  C._backward = () => {
    for (let i = 0; i < len; i++)
      for (let j = 0; j < d; j++) table.grad[i * d + j] += C.grad[i * d + j];
  };
  return C;
}

// ----------------------------------------------------------------------------
// Parameters + Adam.
// ----------------------------------------------------------------------------

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function param(rows, cols, rng, scaleInit) {
  const p = new T(rows, cols);
  const s = scaleInit ?? 1 / Math.sqrt(cols);
  for (let i = 0; i < p.data.length; i++) p.data[i] = (rng() * 2 - 1) * s;
  p.m = new Float64Array(p.data.length);
  p.v = new Float64Array(p.data.length);
  return p;
}

class Adam {
  constructor(lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    Object.assign(this, { lr, b1, b2, eps, t: 0 });
  }
  step(params) {
    this.t++;
    const { lr, b1, b2, eps, t } = this;
    const bc1 = 1 - b1 ** t;
    const bc2 = 1 - b2 ** t;
    for (const p of params)
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];
        p.m[i] = b1 * p.m[i] + (1 - b1) * g;
        p.v[i] = b2 * p.v[i] + (1 - b2) * g * g;
        p.data[i] -= (lr * (p.m[i] / bc1)) / (Math.sqrt(p.v[i] / bc2) + eps);
      }
  }
}

function zeroGrad(params) {
  for (const p of params) p.grad.fill(0);
}

// ----------------------------------------------------------------------------
// Model: learned token + positional embeddings, N pre-norm blocks (single-head
// causal attention + SwiGLU MLP), each block owning a projection M_i to vocab.
// ----------------------------------------------------------------------------

class Block {
  constructor(d, hidden, vocab, rng) {
    this.d = d;
    this.n1 = param(1, d, rng, 0); // RMSNorm gains start at... set to 1 below
    this.n2 = param(1, d, rng, 0);
    this.n1.data.fill(1);
    this.n2.data.fill(1);
    this.Wq = param(d, d, rng);
    this.Wk = param(d, d, rng);
    this.Wv = param(d, d, rng);
    this.Wo = param(d, d, rng);
    this.bq = param(1, d, rng, 0);
    this.bk = param(1, d, rng, 0);
    this.bv = param(1, d, rng, 0);
    this.bo = param(1, d, rng, 0);
    this.W1 = param(d, hidden, rng); // SwiGLU value
    this.W3 = param(d, hidden, rng); // SwiGLU gate
    this.W2 = param(hidden, d, rng); // down
    this.M = param(d, vocab, rng); // local projection to logits (Mono-Forward M_i)
  }

  params() {
    return [
      this.n1, this.n2, this.Wq, this.Wk, this.Wv, this.Wo,
      this.bq, this.bk, this.bv, this.bo, this.W1, this.W3, this.W2, this.M,
    ];
  }

  forward(x) {
    const d = this.d;
    // Pre-norm causal self-attention.
    const h = rmsNorm(x, this.n1);
    const Q = addRow(matmul(h, this.Wq), this.bq);
    const K = addRow(matmul(h, this.Wk), this.bk);
    const V = addRow(matmul(h, this.Wv), this.bv);
    const scores = scale(matmul(Q, transpose(K)), 1 / Math.sqrt(d));
    const attn = matmul(causalSoftmax(scores), V);
    const x1 = add(x, addRow(matmul(attn, this.Wo), this.bo)); // residual
    // Pre-norm SwiGLU MLP.
    const h2 = rmsNorm(x1, this.n2);
    const ffn = matmul(mul(silu(matmul(h2, this.W1)), matmul(h2, this.W3)), this.W2);
    return add(x1, ffn); // residual
  }

  logits(x) {
    return matmul(x, this.M);
  }
}

class Nanoformer {
  constructor({ vocab, d = 16, hidden = 32, layers = 2, maxT = 64, seed = 1 }) {
    const rng = mulberry32(seed);
    this.cfg = { vocab, d, hidden, layers, maxT };
    this.tok = param(vocab, d, rng, 0.1);
    this.pos = param(maxT, d, rng, 0.1);
    this.blocks = Array.from({ length: layers }, () => new Block(d, hidden, vocab, rng));
  }

  inputs(ids) {
    return add(embedRows(this.tok, ids), rowsSlice(this.pos, ids.length));
  }

  // End-to-end backprop through every block; logits come from the last block.
  trainGlobal(ids, targets, opt) {
    const params = [this.tok, this.pos, ...this.blocks.flatMap((b) => b.params())];
    zeroGrad(params);
    let x = this.inputs(ids);
    for (const b of this.blocks) x = b.forward(x);
    const loss = crossEntropy(this.blocks[this.blocks.length - 1].logits(x), targets);
    backward(loss);
    opt.step(params);
    return loss.data[0];
  }

  // Mono-Forward: each block trains its own local next-token loss; gradients
  // never cross a block boundary (the input to block i>0 is detached). Block 0
  // also trains the token + positional embeddings. Each block carries its own
  // optimizer, mirroring the per-actor optimizer in praxis/trainers/mono_forward.
  trainLayerWise(ids, targets, opts) {
    let carry = this.inputs(ids); // tracked; block 0 trains the embeddings
    let total = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      const local = i === 0 ? [this.tok, this.pos, ...b.params()] : b.params();
      const inp = i === 0 ? carry : detach(carry);
      zeroGrad(local);
      const out = b.forward(inp);
      const loss = crossEntropy(b.logits(out), targets);
      backward(loss);
      opts[i].step(local);
      total += loss.data[0];
      carry = out; // detached at the top of the next iteration
    }
    return total / this.blocks.length;
  }

  // Greedy continuation from a prompt, using the last block's projection.
  generate(prompt, steps) {
    const ids = prompt.slice();
    for (let s = 0; s < steps; s++) {
      let x = this.inputs(ids);
      for (const b of this.blocks) x = b.forward(x);
      const lg = this.blocks[this.blocks.length - 1].logits(x);
      const V = lg.cols;
      const last = ids.length - 1;
      let best = 0;
      for (let j = 1; j < V; j++)
        if (lg.data[last * V + j] > lg.data[last * V + best]) best = j;
      ids.push(best);
    }
    return ids;
  }
}

export { T, Nanoformer, Block, Adam, backward, crossEntropy, detach, mulberry32 };
