# micro-CASC

A population of small CASC cascades that communicate through a simplex with an EOF vertex. This is a
research prototype in this repository, awaiting a training budget.

Every node is a full CASC block (`casc.py`) at a reduced width. A message in R^d enters at `IN`; then at
each step a shared router reads the message at every position and places it on a simplex over the N nodes
plus one EOF vertex:

```
m_0     = IN(tokens)
alpha_t = softmax(kappa · z_t)        z_tj = W_q(rms(m_t)) · K_j / sqrt(d_r)   over the N nodes + EOF
m_{t+1} = sum_j alpha_tj · node_j(m_t)            dense mixing: the message goes to every node
e_{t+1} = sum_j alpha_tj · read_j(m_t)            each node also produces a local lexical read
logits  = ET( sum_t rho_t · alpha_EOF(t) · e_t )  EOF mass leaves and is read; the rest circulates
```

EOF is one more vertex of the same simplex, so "where to send the message" and "when to stop" are the
same geometric decision, learned from the same object and the same loss. The routing is computed per
position, which keeps the machine causal. A gate is an optional per-node channel (off by default; it was
found not to sparsify the mixture).

The node law is imported from `casc.py`, never duplicated: a node body is the `TriplexBlockCascade` used
by the full 175M model. Run the tests with:

```bash
python test_micro_casc.py
```

They check that each node is the published CASC block, that the machine is causal (logits at position i
depend only on the tokens up to i), that the EOF mass summed over steps is 1, and that every parameter
receives a gradient.
