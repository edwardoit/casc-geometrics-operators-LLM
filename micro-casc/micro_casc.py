"""micro-CASC: dense geometric communication with a simplex-and-EOF router.

A set of N nodes, each a full CASC block (casc.py) at a reduced width. A message
m in R^d enters at IN; then at every step a shared router places the message on a
simplex over the N nodes plus one EOF vertex. The message is sent to every node at
once and the node outputs are summed, weighted by the simplex: this is the dense
mixing, there is no discrete choice. EOF is one more vertex of that same simplex,
not a threshold and not a second mechanism: the fraction of mass on EOF leaves the
machine at that step and is read into lexical space by the single ET head, while the
rest keeps circulating. The next-token distribution is the readout of the mass
collected across steps.

The law of a node is imported from casc.py, never duplicated: a node body is the
TriplexBlockCascade used by the full 175M model, at small width. The routing is
computed per position from the message, so the machine stays causal (see the test).

This is a research prototype awaiting a training budget; see the repository README.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from casc import rms, rope_cache, TriplexBlockCascade


class Node(nn.Module):
    """One micro-CASC node: CASC block -> message out, plus a local lexical read."""

    def __init__(self, d, d_g, n_head, d_head, D, rank, d_et, cat="FBMMVMS", n=5, init_seed=None, idx=0):
        super().__init__()
        self.body = TriplexBlockCascade(d_x=d, d_g=d_g, n_head=n_head, d_head=d_head, D=D,
                                        n=n, rank=rank, cat=cat, init_seed=init_seed, blk=idx)
        self.P_msg = nn.Linear(d, d, bias=True)                        # generate the message for the next step
        self.gamma_msg = nn.Parameter(torch.ones(d))                   # message out = gamma (.) rms(P_msg(x))
        self.P_exit = nn.Linear(d, d_et, bias=True)                    # local read, mixed into the EOF readout
        with torch.no_grad():
            self.P_msg.weight.normal_(0.0, 0.02); self.P_msg.bias.zero_()
            self.P_exit.weight.normal_(0.0, 0.02); self.P_exit.bias.zero_()

    def forward(self, m, cos, sin):
        x = self.body(m, cos, sin)[0]
        m_out = self.gamma_msg * rms(self.P_msg(x))
        return m_out, self.P_exit(m_out)


class InNode(nn.Module):
    """The single entry: token embedding -> CASC block -> message m_0."""

    def __init__(self, vocab, d, d_g, n_head, d_head, D, rank, cat="FBMMVMS", n=5, init_seed=None):
        super().__init__()
        self.tok = nn.Embedding(int(vocab), d)
        self.body = TriplexBlockCascade(d_x=d, d_g=d_g, n_head=n_head, d_head=d_head, D=D,
                                        n=n, rank=rank, cat=cat, init_seed=init_seed, blk=0)
        self.P_msg = nn.Linear(d, d, bias=True)
        self.gamma_msg = nn.Parameter(torch.ones(d))
        with torch.no_grad():
            self.tok.weight.normal_(0.0, 0.02)
            self.P_msg.weight.normal_(0.0, 0.02); self.P_msg.bias.zero_()

    def forward(self, ids, cos, sin):
        x = self.body(self.tok(ids), cos, sin)[0]
        return self.gamma_msg * rms(self.P_msg(x))


class DenseMicroCASC(nn.Module):
    """Dense communication over N nodes with a simplex-and-EOF router.

    forward(ids) -> next-token logits. At each of cap+1 steps the router reads the
    message at every position, forms a simplex over the N nodes + EOF, mixes every
    node's output by it, and lets the EOF fraction of the mass leave into the ET
    readout. The gate is an optional per-node channel; it is off by default (it was
    found not to sparsify the mixture).
    """

    def __init__(self, vocab=64000, n_nodes=64, d=16, d_g=10, n_head=5, d_head=2, D=26,
                 rank=2, d_et=16, cat="FBMMVMS", n=5, cap=6, use_gate=False, lam=4.0, init_seed=0):
        super().__init__()
        self.N, self.cap, self.d_et = int(n_nodes), int(cap), int(d_et)
        self.d_head, self.use_gate, self.lam = int(d_head), bool(use_gate), float(lam)
        self.in_node = InNode(vocab, d, d_g, n_head, d_head, D, rank, cat, n, init_seed)
        self.nodes = nn.ModuleList([Node(d, d_g, n_head, d_head, D, rank, d_et, cat, n, init_seed, i + 1)
                                    for i in range(self.N)])
        self.ET = nn.Linear(self.d_et, int(vocab), bias=False)         # the single lexical head
        with torch.no_grad(): self.ET.weight.normal_(0.0, 0.02)
        d_r = int(d_head * n_head)                                     # routing-space width
        g = torch.Generator().manual_seed(int(init_seed) * 7919 + 777)
        self.W_q = nn.Linear(d, d_r, bias=False)
        self.K = nn.Parameter(torch.randn(self.N + 1, d_r, generator=g) * 0.3)   # last row = EOF vertex
        self.kappa = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad(): self.W_q.weight.copy_(torch.randn(d_r, d, generator=g) * d ** -0.5)
        if use_gate:
            self.W_g = nn.Linear(d, d_r, bias=False)
            self.Kg = nn.Parameter(torch.randn(self.N, d_r, generator=g) * 0.3)
            self.bg = nn.Parameter(torch.zeros(self.N))
            with torch.no_grad(): self.W_g.weight.copy_(torch.randn(d_r, d, generator=g) * d ** -0.5)

    def field(self, m, t):
        """From the message at every position: alpha over nodes (B,L,N), EOF mass (B,L), gate (B,L,N)."""
        h = rms(m)
        z = (self.W_q(h) @ self.K.T) / (self.K.shape[1] ** 0.5)        # (B, L, N+1)
        if t == 0:                                                     # IN: EOF excluded, the message enters the body
            alpha = torch.softmax(self.kappa * z[..., :self.N], -1)
            a_eof = m.new_zeros(m.shape[:2])
        else:
            a = torch.softmax(self.kappa * z, -1)
            alpha, a_eof = a[..., :self.N], a[..., self.N]
        if self.use_gate:
            g = torch.sigmoid(self.lam * ((self.W_g(h) @ self.Kg.T) / (self.Kg.shape[1] ** 0.5) + self.bg))
        else:
            g = torch.ones_like(alpha)
        return alpha, a_eof, g

    def forward(self, ids):
        cos, sin = rope_cache(ids.shape[1], self.d_head, ids.device, torch.float32)
        m = self.in_node(ids, cos, sin)                               # (B, L, d)
        B, L, _ = m.shape
        rho = m.new_ones(B, L)                                        # mass still circulating, per position
        e_tot = m.new_zeros(B, L, self.d_et)
        e_cur = None
        for t in range(self.cap + 1):
            alpha, a_eof, g = self.field(m, t)
            beta = alpha * g
            if t >= 1:                                                # EOF leaves: read the collected mass
                h = rho * a_eof if t < self.cap else rho              # at the cap all remaining mass exits
                e_tot = e_tot + h[..., None] * e_cur
                rho = rho * (1.0 - a_eof)
            if t == self.cap:
                break
            outs = torch.stack([nd(m, cos, sin)[0] for nd in self.nodes], 1)   # (B, N, L, d)
            exs = torch.stack([nd(m, cos, sin)[1] for nd in self.nodes], 1)    # (B, N, L, d_et)
            m = torch.einsum("bln,bnld->bld", beta, outs)             # dense mixing of node outputs
            e_cur = torch.einsum("bln,bnld->bld", beta, exs)
        return self.ET(e_tot)
