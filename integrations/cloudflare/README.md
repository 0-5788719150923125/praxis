# Cloudflare Pages integration

Publishes a static snapshot of the Praxis dashboard to [Cloudflare Pages](https://pages.cloudflare.com/)
so the showcase stays online for free after the training server shuts down.

Running the dashboard as an always-on public server is expensive. This
integration freezes the dashboard into a directory of static files - the built
frontend plus a dump of every read-only endpoint - and pushes it to Pages on a
cadence. The captured page keeps the charts, dynamics, architecture, evolution,
and a browsable docs/code knowledge base. The live and interactive surfaces
(chat, generation, the swarm, the metrics websocket) are disabled behind an
offline banner in the notification bell and greyed-out controls.

## How it works

Two phases, so failures surface fast and content waits for real data:

1. **Bootstrap (immediate, REST-only, fail-loud).** When the run starts,
   `ensure_infrastructure()` provisions the Cloudflare Pages project and - if a
   showcase domain is configured - attaches the custom domain, creates the
   proxied DNS `CNAME`, and re-validates it. This is a handful of REST calls
   (seconds), so a bad token, account, or zone errors at startup instead of
   after a 10-minute `torch.compile`.
2. **Content upload (deferred).** The static site is exported and pushed via
   `wrangler` only once training has actually stepped (past compile), so the
   debut snapshot carries real curves rather than an empty pre-training
   dashboard, then re-published on the interval.

The export runs **in-process** against the live Flask app's test client, so the
model-dependent endpoints (activation curves, head snapshots, dynamics) dump
with real data straight from the already-warm snapshot store. No route logic is
re-implemented; the only new client-side code is `static_mode.js`, injected into
the exported copy, which redirects `/api/*` fetches onto the dumped files and
stubs the websocket.

Two Cloudflare gotchas the integration handles for you (learned the hard way):

- **The project's `pages.dev` subdomain is not `<project>.pages.dev`.** Those
  names are globally unique, so a taken name (e.g. `praxis`) gets a random
  suffix - `praxis-6gm.pages.dev`. The integration fetches the *canonical*
  subdomain from the API rather than assuming one.
- **Cloudflare does not reliably auto-create the DNS record** for a Pages custom
  domain. The integration creates the proxied `CNAME` explicitly (pointing at
  the canonical subdomain) and re-validates the domain, since the first
  validation fails if it runs before the record exists.

Scope (v1, deliberately small):

- **Active run only.** Multi-run compare is dropped; any `runs=` request
  resolves to the single current-run dump.
- **KB = docs + code.** Crawled pages and external links are excluded; search
  runs client-side over the dumped feed.
- **Static business-card seed** (42), pre-rendered front/back x light/dark.

## Setup

The Cloudflare CLI (`wrangler`) and Node.js are baked into the Docker image (see
the `RUN ... npm install -g wrangler` block in the `Dockerfile`), because the
training container runs as a non-root user and can't install them at runtime.
After pulling these changes, **rebuild the image** - just re-run `./launch ...`,
which detects the Dockerfile change and rebuilds. Outside Docker, if `wrangler`
is missing the integration tries `npm install -g wrangler` (and, as root,
`apt-get install nodejs npm`) as a fallback. All you have to provide are the
credentials:

1. Create an API token at <https://dash.cloudflare.com/profile/api-tokens> with
   the **Cloudflare Pages: Edit** permission, and find your account ID on the
   dashboard home page.

2. Store the credentials where the integration can read them - either exported
   in the environment or in the repo-root `.env` file (loaded via python-dotenv):

   ```
   CLOUDFLARE_API_TOKEN=your-token
   CLOUDFLARE_ACCOUNT_ID=your-account-id
   ```

### Custom showcase domain

To serve the snapshot from a domain you already have on the Cloudflare account
(e.g. `arc.src.eco`), add one line to `.env`:

```
CLOUDFLARE_PAGES_DOMAIN=arc.src.eco
```

When this is set, the active run is deployed as the project's **production**
deployment. At bootstrap the integration attaches the domain, creates the proxied
`CNAME` pointing at the project's canonical subdomain, and re-validates - all via
the Cloudflare API, idempotently. The zone must be on the same Cloudflare account
(so the token can write the DNS record); the TLS cert provisions automatically
and the first publish may take a minute to go live. Re-publishing overwrites the
same URL, so the showcase always shows the latest run.

The production branch the domain serves defaults to `main` and is fixed by the
project's first deployment; override it only if your project uses a different
production branch:

```
# CLOUDFLARE_PAGES_PRODUCTION_BRANCH=main
```

Without `CLOUDFLARE_PAGES_DOMAIN`, each run instead publishes to its own
`<run-hash>.<canonical-subdomain>` preview alias (see below).

## Usage

Add the flag to a normal training launch:

```
python main.py ... --publish-snapshot --publish-project praxis --publish-interval 30
```

- `--publish-snapshot` - enable periodic publishing.
- `--publish-project` - Cloudflare Pages project name (default: `praxis`).
- `--publish-interval` - minutes between publishes (default: 30).

The project/domain/DNS are provisioned at startup; the first *content* upload
waits until training has stepped past `torch.compile` (capped at 20 minutes so a
stalled run still publishes), then re-publishes every interval. The static site is
assembled at `build/snapshot/` and uploaded with `wrangler pages deploy`. If the
credentials or `wrangler` are missing, the export still completes and the
directory is left in place for a manual upload.

## One project, one URL per run

Everything deploys to a **single** Cloudflare Pages project (`--publish-project`,
default `praxis`), so pivoting across runs never proliferates or orphans Pages
projects. Each run lands on its own deterministic **branch alias** derived from
its run hash, under the project's canonical subdomain:

```
<run-hash>.praxis-6gm.pages.dev
```

Because the alias is a pure function of the run hash, re-publishing the same run
overwrites its own URL in place instead of creating a new site, and old runs'
aliases are just branch deployments under the one project (free, and removable
with `wrangler pages deployment` if you want to prune them). The project's
production URL (the canonical `praxis-6gm.pages.dev`, or your custom domain)
tracks whichever deployment Cloudflare has marked production.

## Run identity

The `--publish-*` flags are runtime/infra switches, not model architecture, so
they must not change a run's hash (which would fork a new run directory just for
toggling publishing). The integration declares them via
`BaseIntegration.hash_exclusions()`, and the CLI merges that into the args-hash
exclusion list whenever the integration is loaded - so `python main.py ...` and
`python main.py ... --publish-snapshot` resolve to the same run.

## What can't come along

Endpoints with no static equivalent are gated: chat / Evaluate / Print / Loop,
swarm join, `/api/agents` live scans, git smart-HTTP, and the metrics
websocket. Point the git clone URL at the GitHub mirror instead.
