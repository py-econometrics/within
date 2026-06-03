| Dataset | Rows | DOFs | Comp. | LCC edge share | $\lambda_2$ | $\phi(S)$ | Approx resid. | Rel. y gap | One-shot time | Corrected time | Solver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `credit` | 516,810 | 14,086 | 1 | 1.0000 | 0.2258 | 0.2348 | 0.0063 | 0.0010 | 0.042s | 0.231s | MAP or accelerated MAP |
| `credit2` | 19,094 | 741 | 1 | 1.0000 | 0.1808 | 0.2163 | 0.0410 | 0.0113 | 0.007s | 0.026s | MAP or accelerated MAP |
| `directors` | 525,012 | 796,775 | 281,627 | 0.0474 | 1.10e-04 | 0.0058 | nan | nan | nan | nan | component-parallel exact |
| `enron` | 367,662 | 73,384 | 1,887 | 0.9836 | 0.0035 | 0.0045 | 0.8274 | 3.0306 | 0.147s | 19.440s | Schwarz-preconditioned exact |
| `github` | 548,843 | 52,328 | 13,232 | 0.3205 | 3.15e-04 | 0.0023 | 0.1006 | 0.1621 | 0.044s | 1.160s | component-parallel exact |
| `patents` | 500,008 | 464,832 | 25,229 | 0.7602 | 1.46e-04 | 6.84e-04 | 0.2247 | 0.8400 | 0.940s | 39.844s | component-parallel exact |
| `schools` | 413,444 | 218,682 | 15 | 0.9989 | 0.0011 | 0.0055 | 0.0279 | 0.0230 | 0.074s | 0.536s | Schwarz-preconditioned exact |
| `soccer` | 73,487 | 1,049 | 1 | 1.0000 | 0.6663 | 0.3982 | 0.0314 | 0.0097 | 0.010s | 0.075s | MAP or accelerated MAP |
| `synthetic-assortative` | 499,155 | 221,034 | 33,042 | 0.6898 | 6.34e-04 | 0.0047 | 0.0531 | 1.7744 | 0.433s | 13.215s | component-parallel exact |
| `synthetic-complete` | 500,000 | 1,500 | 1 | 1.0000 | 1.0000 | 0.5177 | 0.0097 | 0.0019 | 0.047s | 0.291s | MAP or accelerated MAP |
| `synthetic-uniform-easy` | 500,000 | 216,027 | 1 | 1.0000 | 0.3385 | 0.2927 | 2.30e-14 | 2.99e-15 | 0.068s | 0.131s | one-shot parallel |
| `synthetic-uniform-hard` | 500,000 | 245,357 | 194 | 0.9996 | 0.0449 | 0.0769 | 0.4040 | 0.5455 | 0.273s | 10.647s | Schwarz-preconditioned exact |
| `synthetic-uniform-harder` | 500,000 | 432,053 | 13,213 | 0.9588 | 0.0047 | 0.0238 | 0.5476 | 1.7951 | 1.054s | 53.708s | Schwarz-preconditioned exact |
| `synthetic-zigzag` | 10,002 | 10,001 | 1 | 1.0000 | 2.31e-05 | 0.0022 | 3.70e-15 | 2.03e-12 | 0.006s | 0.016s | one-shot parallel |
| `workers` | 504,315 | 247,254 | 19,994 | 0.6132 | 1.40e-04 | 0.0015 | 0.0013 | 0.0199 | 0.224s | 2.931s | component-parallel exact |
