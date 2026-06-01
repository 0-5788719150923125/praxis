# web/src/js/__tests__

Node-run checks for the pure-JS swarm model. Not served to the browser - the
build globs `src/js/*.js` non-recursively, so this subdirectory is excluded.

## gradcheck.js

Finite-difference verification that `nanoformer.js`'s autograd is exact: perturbs
a sample of parameters by ±eps and compares the numeric gradient to the analytic
one from `backward()` across attention, RMSNorm, SwiGLU, and cross-entropy.

```
node praxis/web/src/js/__tests__/gradcheck.js
```

Expected: `worst relative error: ~1e-7  PASS`. Run it after touching any op in
`nanoformer.js`.
