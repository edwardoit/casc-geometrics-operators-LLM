"""micro-CASC tests: the node reuses the CASC law, the machine is causal, EOF mass is
conserved, and every parameter gets a gradient. Run: python test_micro_casc.py"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # micro_casc.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # casc.py
import casc
from micro_casc import DenseMicroCASC

torch.set_num_threads(min(8, os.cpu_count() or 1))
res = []
def ck(cond, msg): res.append(bool(cond)); print(("  OK   " if cond else "  FAIL ") + msg)


def machine(use_gate=False, cap=4, n=8):
    torch.manual_seed(0)
    return DenseMicroCASC(vocab=256, n_nodes=n, cap=cap, use_gate=use_gate, init_seed=1).eval()


print("law reuse")
m = machine()
ck(isinstance(m.in_node.body, casc.TriplexBlockCascade) and all(isinstance(nd.body, casc.TriplexBlockCascade) for nd in m.nodes),
   "every node body is TriplexBlockCascade imported from casc.py")
ck(m.ET.weight.shape == (256, m.d_et) and len([p for p in m.named_parameters() if p[0] == "ET.weight"]) == 1,
   "one lexical head ET, only at the output")

print("\ncausality")
ids = torch.randint(0, 256, (2, 14))
ids2 = ids.clone(); ids2[:, 9:] = (ids2[:, 9:] + 31) % 256
with torch.no_grad(): a, b = m(ids), m(ids2)
ck(torch.allclose(a[:, :9], b[:, :9], atol=1e-5), "logits before an edited token are unchanged (routing is per position)")
ck(not torch.allclose(a[:, 9:], b[:, 9:]), "logits at and after the edit do change")

print("\nsimplex and EOF")
with torch.no_grad():
    cos, sin = casc.rope_cache(14, m.d_head, ids.device, torch.float32)
    msg = m.in_node(ids, cos, sin)
    alpha0, a_eof0, _ = m.field(msg, 0)
    ck(torch.allclose(alpha0.sum(-1), torch.ones_like(a_eof0), atol=1e-5) and float(a_eof0.abs().max()) == 0.0,
       "step 0: alpha is a simplex over the N nodes, no mass leaves before the message enters the body")
    rho = msg.new_ones(msg.shape[:2]); exit_sum = msg.new_zeros(msg.shape[:2])
    for t in range(m.cap + 1):
        alpha, a_eof, g = m.field(msg, t)
        if t >= 1:                                                     # mirror forward(): flush all at the cap
            h = rho * a_eof if t < m.cap else rho
            exit_sum = exit_sum + h; rho = rho * (1.0 - a_eof) if t < m.cap else rho.new_zeros(rho.shape)
        if t < m.cap:
            msg = torch.einsum("bln,bnld->bld", alpha * g, torch.stack([nd(msg, cos, sin)[0] for nd in m.nodes], 1))
    ck(torch.allclose(exit_sum, torch.ones_like(exit_sum), atol=1e-5) and float(rho.abs().max()) == 0.0,
       "mass is conserved: EOF mass summed over steps = 1 per position, nothing left at the cap")

print("\ngradients")
for gate in (False, True):
    mm = machine(use_gate=gate)
    z = mm(torch.randint(0, 256, (2, 10)))
    torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, 256), torch.randint(0, 256, (18,))).backward()
    miss = sorted(n for n, p in mm.named_parameters() if p.grad is None)
    expect = sorted(n for n, _ in mm.named_parameters() if n.endswith((".gru.beta_r", ".gru.kappa_r", ".gru.rho_log")))
    ck(miss == expect, "gate=%s: only the 3 scalars of the unused GRU mode (gru_norm=0) get none, as in the full model" % gate)

print("\n%d/%d checks passed" % (sum(res), len(res)))
sys.exit(0 if all(res) else 1)
