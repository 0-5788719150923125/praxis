# staging/

Welcome to the junkyard! This is where we dump experimental code that doesn't belong in core Praxis.

## What goes here?

**Experimental and temporary code only:**

- **Experimental features** that might break things
- **Work-in-progress implementations** not ready for prime time
- **Test scripts and prototypes** for trying out ideas
- **Random, orphaned code** that needs a temporary home
- **Old implementations** kept for reference but not actively used

## What DOESN'T go here anymore?

**Integrations have moved!** External integrations (wandb, ngrok, gun, quantum, etc.) now live in `/integrations/` at the project root. They're no longer experimental - they're first-class features with proper discovery and loading.

## The deal

This is truly a staging area now - code here is either:
1. On its way IN (being developed for eventual inclusion in core)
2. On its way OUT (deprecated but kept temporarily for reference)
3. Perpetually experimental (useful for specific cases but too niche for core)

Keep your experiments contained. Clean up after yourself. And when something becomes stable and useful, promote it to either:
- Core Praxis (if it's essential)
- `/integrations/` (if it's an optional extension)

## Note on old references

If you're looking for the integration system that used to be here, check `/integrations/` instead. The module loading system now uses `spec.yaml` files and the `IntegrationLoader` class.
