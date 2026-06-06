# KB Provenance & Spider

Two upgrades to the knowledge base: first-class provenance on every search
entry, and a slow, bounded web spider that grounds the KB in external sites.

## Part 1 - Provenance in the registration schema

Supporting details are currently ad-hoc: notes stash `meta["document"]`, runs
stash `meta["summary"]`, links stash `meta["found_in"]`, and cards/agents carry
nothing. Three sources invented three names for "where did this come from",
and nothing guarantees a new source provides any of it.

`KBItem` gains three first-class fields:

- `source` - registry name of the producer ("notes", "links", "pages").
  Stamped by the bus during indexing, not self-reported (same spirit as the
  Dynamics caller labels).
- `origin` - the document or event that yielded the item: "next/roadmap.md",
  a run's config, the watched site a page came from.
- `summary` - a one-line supporting detail for feeds and result rows. When a
  source leaves it empty, the bus derives one from the first non-heading line
  of the body, so even a bare link gets something legible.

`kb.db` is a rebuildable cache, so no migration pain: recreate the FTS table
with the new columns and rebuild. `origin` and `summary` are searchable;
`source` is unindexed. Result rows render origin as a secondary chip.

## Part 2 - The spider (praxis.spider)

Slow, inobtrusive, eventually consistent. It does not try to crawl anything
all at once: it grounds itself in a small set of watched sites, then walks
each one top-to-bottom in pieces, one fetch per tick, over days.

### Storage split

`kb.db` is deleted on every reindex, so crawled content cannot live there.
The spider owns `build/spider.db` (plain SQLite, the store of record):

- `sites(url, added, last_fetch, error_streak, enabled)` - the watchlist
- `frontier(site, url, depth, discovered, next_due)` - per-site queue
- `pages(url, site, title, text, summary, fetched, etag, last_modified)`

A `PagesSource` reads spider.db into the index like any other source. The
spider thread only ever writes spider.db, never the live index - pages flow
into kb.db through the normal reindex path.

### Constraints

Fixed constants in `SPIDER_REGISTRY` profiles - no per-run tuning:

| Knob | gentle | Why |
|---|---|---|
| `max_sites` | 16 | watchlist cap; adding past it evicts the stalest |
| `max_pages_per_site` | 50 | frontier + stored pages cap per site |
| `max_kb_pages` | 200 | cap on "page" items surfaced into the index |
| `tick_seconds` | 300 | one HTTP request per tick, globally |
| `domain_seconds` | 1800 | minimum spacing between hits to one host |
| `max_page_bytes` | 2 MB | size cap, text/html only (the multi_dir lesson) |
| `revisit_days` | 7 | re-fetch the stalest page when the frontier is dry |

### Behavior per tick

1. Pick the most-overdue frontier entry whose domain is not rate-limited.
2. Check robots.txt (cached); send a conditional GET (ETag/Last-Modified) -
   a 304 costs the site almost nothing and makes "watching" cheap.
3. Extract title, readable text, and same-site links with stdlib
   `html.parser`. No new dependencies for v1.
4. Append same-site links to the frontier breadth-first (top-to-bottom),
   respecting `max_pages_per_site`.
5. On errors: exponential backoff per domain; a long error streak disables
   the site rather than retrying forever.

### Seeding

The watchlist seeds from what `LinksSource` already finds - URLs cited in
docs/ and next/ are by definition the interesting sites to ground in. Manual
add/remove via `tools/watch_url.py`, later a dashboard affordance.

### Enabling

A single key/valued CLI flag, off by default:

    --spider                      # gentle profile, all defaults
    --spider profile=gentle tick_seconds=600 max_sites=8

`profile` picks a base from `SPIDER_REGISTRY`; remaining KEY=VALUE entries
override individual fields. The worker runs as a daemon thread alongside the
API server (rank 0 only), same lifecycle as the dashboard.

### Link graph & telemetry (built)

The spider tracks its steps: a `refs` table records every citation edge
(page -> link) as it crawls. Citation count ranks the frontier - URLs many
pages point at are fetched before one-off (likely bad) links. Links to
unwatched sites accrue in `external_refs`; a site cited by 3+ distinct pages
is promoted into a free watchlist slot (promotion never evicts - capacity
stays the limit). This is how the spider finds something it wants to watch.
Safety holds throughout: fetch and parse only, no script execution, robots
honored, size/type caps enforced.

Telemetry flows two ways:
- Scalars - `SpiderCallback` mirrors spider.db counts into the metrics
  stream at logging intervals: pages held, cumulative discoveries, revisits,
  frontier size, watchlist size. Research-tab cards, only emitted when
  `--spider` is on.
- Link graph - the "Spider Citations" card (standalone `/api/spider`) ranks
  top cited URLs as bars, with top referrers in the tooltip.

### Phases

1. **Schema** - provenance fields, bus stamping, frontend chips. Done first;
   standalone value even without the spider.
2. **Spider core** - spider.db, fetch/extract/frontier loop, seeding.
3. **Surfacing** - `PagesSource`, `--spider` flag, worker startup, web
   rendering ("page" reads inline, with an open-externally affordance).
4. **Management** - watchlist CRUD on the dashboard, per-site status.
