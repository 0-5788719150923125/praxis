// sidecar.js - the backend's own expert host.
//
// A single-process Node app that runs the SAME transformer math as the browser
// agents (nanoformer.js), so the remote-expert pool has peers the moment the
// app comes online - before any browser connects. It hosts a pool of tiny
// experts and serves them over HTTP; the Python ExpertPool talks to it, and the
// frontend joins the same pool by POSTing new experts.
//
// Each expert is a full tiny transformer that maps token ids -> next-token
// logits, i.e. it VOTES over the distribution (the CALM expert-vote model the
// Python `vote` mixer expects). Training is local, layer-wise Mono-Forward.
//
// Endpoints (JSON):
//   GET  /capacity            -> pool stats
//   GET  /experts             -> [{uid, kind, rank, passes, steps, last_loss}]
//   POST /experts {count?}    -> spawn N experts (default 1); returns their uids
//   DELETE /experts/:uid      -> remove one
//   POST /expert/:uid/forward {ids}          -> {logits}  (last-position vote)
//   POST /expert/:uid/train   {ids, targets} -> {loss}
//
// Run: `node sidecar.js [--port 7777] [--experts 4] [--dim 14] [--vocab 16]`

import http from "node:http";
import { Nanoformer, Adam } from "./nanoformer.js";

const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, arr) =>
    a.startsWith("--") ? [a.slice(2), arr[i + 1]] : [null, null]
  ).filter(([k]) => k)
);
const PORT = parseInt(args.port || process.env.PRAXIS_SIDECAR_PORT || "7777", 10);
const DIM = parseInt(args.dim || "14", 10);
const VOCAB = parseInt(args.vocab || "16", 10);
const FFN = DIM * 2;
const INIT_EXPERTS = parseInt(args.experts || "4", 10);

let _seq = 0;

// One sidecar-hosted expert: a tiny Nanoformer + per-layer Adam optimizers.
class Expert {
  constructor() {
    const n = _seq++;
    this.uid = `sidecar-${n.toString(36)}`;
    this.kind = "sidecar";
    this.dim = DIM;
    this.passes = 0;
    this.steps = 0;
    this.lastLoss = null;
    this.model = new Nanoformer({
      vocab: VOCAB, d: DIM, hidden: FFN, layers: 1, maxT: 64,
      seed: (n * 2654435761) >>> 0,
    });
    this.opts = this.model.blocks.map(() => new Adam(0.01));
  }
  // ids -> logits at the final position (the expert's vote over next token).
  forward(ids) {
    this.passes++;
    let x = this.model.inputs(ids);
    for (const b of this.model.blocks) x = b.forward(x);
    const lg = this.model.blocks[this.model.blocks.length - 1].logits(x);
    const last = ids.length - 1;
    return Array.from(lg.data.slice(last * lg.cols, (last + 1) * lg.cols));
  }
  train(ids, targets) {
    this.lastLoss = this.model.trainLayerWise(ids, targets, this.opts);
    this.steps++;
    this.passes++;
    return this.lastLoss;
  }
  info() {
    return { uid: this.uid, kind: this.kind, rank: this.dim,
             passes: this.passes, steps: this.steps, last_loss: this.lastLoss };
  }
}

const experts = new Map();
function spawn(count = 1) {
  const made = [];
  for (let i = 0; i < count; i++) { const e = new Expert(); experts.set(e.uid, e); made.push(e.uid); }
  return made;
}
spawn(INIT_EXPERTS); // backend comes online with a starter pool

function capacity() {
  const all = [...experts.values()];
  return {
    experts_total: all.length,
    experts_alive: all.length,
    total_rank: all.reduce((s, e) => s + e.dim, 0),
    passes: all.reduce((s, e) => s + e.passes, 0),
    steps: all.reduce((s, e) => s + e.steps, 0),
  };
}

// --- tiny JSON HTTP router ---------------------------------------------------

function send(res, code, body) {
  const data = JSON.stringify(body);
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) });
  res.end(data);
}
function readJson(req) {
  return new Promise((resolve) => {
    let b = "";
    req.on("data", (c) => (b += c));
    req.on("end", () => { try { resolve(b ? JSON.parse(b) : {}); } catch { resolve({}); } });
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const path = url.pathname;
  try {
    if (req.method === "GET" && path === "/capacity") return send(res, 200, capacity());
    if (req.method === "GET" && path === "/experts")
      return send(res, 200, [...experts.values()].map((e) => e.info()));
    if (req.method === "POST" && path === "/experts") {
      const body = await readJson(req);
      const uids = spawn(Math.max(1, parseInt(body.count || 1, 10)));
      return send(res, 200, { spawned: uids, capacity: capacity() });
    }
    const mDel = path.match(/^\/experts\/(.+)$/);
    if (req.method === "DELETE" && mDel) {
      const ok = experts.delete(decodeURIComponent(mDel[1]));
      return send(res, ok ? 200 : 404, { removed: ok, capacity: capacity() });
    }
    const mFwd = path.match(/^\/expert\/(.+)\/forward$/);
    if (req.method === "POST" && mFwd) {
      const e = experts.get(decodeURIComponent(mFwd[1]));
      if (!e) return send(res, 404, { error: "no such expert" });
      const { ids } = await readJson(req);
      return send(res, 200, { logits: e.forward(ids) });
    }
    const mTrn = path.match(/^\/expert\/(.+)\/train$/);
    if (req.method === "POST" && mTrn) {
      const e = experts.get(decodeURIComponent(mTrn[1]));
      if (!e) return send(res, 404, { error: "no such expert" });
      const { ids, targets } = await readJson(req);
      return send(res, 200, { loss: e.train(ids, targets) });
    }
    send(res, 404, { error: "not found" });
  } catch (err) {
    send(res, 500, { error: String(err) });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`[sidecar] expert host on http://127.0.0.1:${PORT} (${INIT_EXPERTS} experts, dim ${DIM}, vocab ${VOCAB})`);
});
