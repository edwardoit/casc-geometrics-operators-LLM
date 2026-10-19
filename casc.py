"""CASC: the complete mathematical machine of CASC-NTP A (configuration F-B-M-M-V-M-S).

    x^(0) = E[y]                                             tied token embedding, E in R^{V x d_x}
    for each of the S blocks:
        G      = signed-simplex GRU memory over RMSNorm(x)   causal carrier, one step per token
        x+     = x + A(G) W_O                                communication attention A (RoPE, causal)
        for each of the K stages k (sequential, family c_k in F, B, M, V, S):
            h_k  = RMSNorm_u(x^(k-1))                         the stage re-reads the updated hidden state
            pi_k = softmax(W_k C(h_k) + b_k)                  control attention C, six-slot simplex
            u_k  = SiLU(h_k P_g) * (h_k P_up)                 lift into R^D (projectors shared per block)
            d_k  = sum_{j=1..5} pi_kj op_kj(u_k)              slot 0 is the identity and contributes d = 0
            x^(k)= x^(k-1) + psi_k(d_k) P_dn                  psi = identity for S, SiLU for F/B/M/V
    logits = RMSNorm_f(x) E^T                                readout tied to the same Parameter as E

Training is ordinary next-token cross-entropy with standard backpropagation and AdamW (see train.py).
Every class and function below is the code that trained the published checkpoint, unchanged.
The legacy class name TriplexCascadeNTP is kept so that the checkpoint loads with strict=True and
without key translation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6, learnable=True):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d)) if learnable else None

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.g if self.g is not None else y

def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def rope_cache(L, dh, device, dtype, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh))
    t = torch.arange(L, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return torch.cos(f).to(dtype), torch.sin(f).to(dtype)

def rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)

def entropy(p, dim=-1, eps=1e-12):
    return -(p * torch.log(p.clamp_min(eps))).sum(dim)


class SignedSimplexGRUCell(nn.Module):

    def __init__(self, d_g=512, beta_init=1.0, kappa_init=1.0, eps=1e-6, gru_norm=0,
                 gamma_carrier=False):
        super().__init__()
        self.d_g, self.eps = d_g, eps
        self.gru_norm = int(gru_norm)
        self.beta_r = nn.Parameter(torch.tensor(float(beta_init)))
        self.kappa_r = nn.Parameter(torch.tensor(float(kappa_init)))
        self.rho_log = nn.Parameter(torch.zeros(()))
        self.gamma = nn.Parameter(torch.ones(d_g))
        self.gamma_carrier = bool(gamma_carrier)
        k = 1.0 / math.sqrt(d_g)
        def P(*s): return nn.Parameter(torch.empty(*s).uniform_(-k, k))
        self.Wr, self.Ur, self.br = P(d_g, d_g), P(d_g, d_g), nn.Parameter(torch.zeros(d_g))
        self.Wz, self.Uz, self.bz = P(d_g, d_g), P(d_g, d_g), nn.Parameter(torch.zeros(d_g))
        self.Wh, self.Uh, self.bh = P(d_g, d_g), P(d_g, d_g), nn.Parameter(torch.zeros(d_g))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))

    def proj(self, e):
        return e @ self.Wr + self.br, e @ self.Wz + self.bz, e @ self.Wh + self.bh

    def forward(self, e, gbar_prev):
        er, ez, eh = self.proj(e)
        return self.step(er, ez, eh, gbar_prev)

    def step(self, er, ez, eh, gbar_prev, Urz=None):
        if Urz is None: Urz = torch.cat([self.Ur, self.Uz], 1)
        gu = gbar_prev @ Urz
        r = F.softplus(er + gu[:, : self.d_g])
        z = torch.sigmoid(ez + gu[:, self.d_g :])
        v = r * gbar_prev
        if self.gru_norm == 2:
            s_r = torch.tanh(self.beta_r * v)
            w_r = F.softplus(self.kappa_r * torch.sqrt(v * v + self.eps))
            a_r = w_r / w_r.sum(-1, keepdim=True)
            mmem = (self.d_g * torch.exp(self.rho_log)) * s_r * a_r
        elif self.gru_norm == 1:
            mmem = rms(v, self.eps)
        else:
            mmem = v
        h = torch.tanh(eh + mmem @ self.Uh)
        g_raw = (1.0 - z) * gbar_prev + z * h

        s = torch.tanh(self.beta * g_raw)
        m = torch.sqrt(g_raw * g_raw + self.eps)
        w = F.softplus(self.kappa * m)
        alpha = w / w.sum(-1, keepdim=True)
        g = s * alpha
        gbar = rms(g, self.eps)
        if self.gamma_carrier:
            gbar = gbar * self.gamma
        return g, gbar, dict(alpha=alpha, s=s, z=z, r=r, g_raw=g_raw)


class CausalMHA(nn.Module):

    def __init__(self, d_model=512, n_head=8, d_head=64, tau_init=1.0):
        super().__init__()
        assert n_head * d_head == d_model
        self.H, self.dh = n_head, d_head
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.tau = nn.Parameter(torch.full((n_head,), float(tau_init)))

    def forward(self, g, cos, sin, stats=False):
        B, L, _ = g.shape
        q = self.Wq(g).view(B, L, self.H, self.dh).transpose(1, 2)
        k = self.Wk(g).view(B, L, self.H, self.dh).transpose(1, 2)
        v = self.Wv(g).view(B, L, self.H, self.dh).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)

        sc = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        sc = sc * self.tau.view(1, -1, 1, 1)
        mask = torch.ones(L, L, dtype=torch.bool, device=g.device).tril()
        sc = sc.masked_fill(~mask, float("-inf"))
        beta = torch.softmax(sc, dim=-1)
        c = (beta @ v).transpose(1, 2).reshape(B, L, self.H * self.dh)

        st = None
        if stats:
            pos = torch.arange(L, device=g.device, dtype=beta.dtype)
            dist = (pos[:, None] - pos[None, :]).clamp_min(0)
            st = dict(
                ent=entropy(beta).mean().item(),
                maxw=beta.max(-1).values.mean().item(),
                dist=(beta * dist[None, None]).sum(-1).mean().item(),
                headdiv=(1.0 - torch.einsum("bhij,bgij->bhg", beta, beta).mean()).item(),
            )
        return c, st


class CausalMHAIn(CausalMHA):
    def __init__(self, d_in=896, d_model=512, n_head=8, d_head=64, tau_init=1.0):
        super().__init__(d_model, n_head, d_head, tau_init)
        self.Wq = nn.Linear(d_in, d_model, bias=False)
        self.Wk = nn.Linear(d_in, d_model, bias=False)
        self.Wv = nn.Linear(d_in, d_model, bias=False)


FAMILIES = "FBSMV"
F_INIT_GAIN = 6.0


def _spec(X, k=2048):
    if X.shape[0] > k: X = X[:: max(1, X.shape[0] // k)]
    X = (X - X.mean(0, keepdim=True)).float()
    s = torch.linalg.svdvals(X); e = s * s; c = torch.cumsum(e, 0) / e.sum().clamp_min(1e-12)
    return int((c < 0.90).sum()) + 1, int((c < 0.99).sum()) + 1


class CascadeBranch(nn.Module):

    def __init__(self, fam, D=2048, n=5, rank=32, eps0=0.02, seed=None):
        super().__init__()
        self.fam, self.D, self.n, self.rank = fam, D, n, rank
        sd = 1.0 / (D ** 0.5)
        _g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF) if seed is not None else None
        rn = lambda *sh: torch.randn(*sh, generator=_g)
        if fam == "F":
            self.A = nn.Parameter(rn(n, D, rank) * sd)
            self.Bm = nn.Parameter(rn(n, D, rank) * sd * eps0 * F_INIT_GAIN)
            self.U = nn.Parameter(rn(n, D, rank) * sd)
        elif fam == "B":
            self.T = nn.Parameter(rn(n, D, rank) * sd)
            self.N = nn.Parameter(rn(n, D, rank) * sd * eps0)
        elif fam == "S":
            self.U = nn.Parameter(rn(n, D, rank) * sd)
            self.V = nn.Parameter(rn(n, D, rank) * sd * eps0)
        else:
            self.U = nn.Parameter(rn(n, D, rank) * sd)
            self.A = nn.Parameter(rn(n, D, rank) * sd * 0.1)
            if fam == "M":
                self.beta_gl = nn.Parameter(torch.ones(rank))
                self.laws = ["tanh", "softsign", "asinh", "gl", "silu"][:n]
            else:
                self.omega = nn.Parameter(1.0 + 0.1 * rn(n, rank))
                self.nu = nn.Parameter(1.0 + 0.1 * rn(n, rank))
                self.c_sin = nn.Parameter(torch.ones(n, rank))
                self.c_cos = nn.Parameter(torch.ones(n, rank))
                self.laws = ["tanh", "asinh", "sin", "cos1", "vib"][:n]

    def _phi(self, j, q):
        law = self.laws[j]
        if law == "tanh": return torch.tanh(q)
        if law == "softsign": return q / (1.0 + q.abs())
        if law == "asinh": return torch.asinh(q)
        if law == "silu": return F.silu(q)
        if law == "gl": return q * torch.sigmoid(self.beta_gl * q.abs())
        if law == "sin": return torch.sin(self.omega[j] * q)
        if law == "cos1": return 1.0 - torch.cos(self.nu[j] * q)
        if law == "vib": return self.c_sin[j] * torch.sin(self.omega[j] * q) + self.c_cos[j] * (1.0 - torch.cos(self.nu[j] * q))
        raise ValueError(law)

    def deform(self, u, pi):
        w = pi[:, 1:1 + self.n].T.unsqueeze(-1)
        if self.fam == "F":
            a = torch.einsum("nd,jdr->jnr", u, self.A); b = torch.einsum("nd,jdr->jnr", u, self.Bm)
            return torch.einsum("jnr,jdr->nd", w * (F.silu(a) * b), self.U)
        if self.fam == "B":
            Q, _ = torch.linalg.qr(self.T)
            Np = self.N - Q @ (Q.transpose(1, 2) @ self.N)
            return torch.einsum("jnr,jdr->nd", w * F.silu(torch.einsum("nd,jdr->jnr", u, self.T)), Np)
        if self.fam == "S":
            return torch.einsum("jnr,jdr->nd", w * torch.einsum("nd,jdr->jnr", u, self.V), self.U)
        a = torch.einsum("nd,jdr->jnr", u, self.U); q = torch.einsum("nd,jdr->jnr", u, self.A)
        core = torch.stack([a[j] * self._phi(j, q[j]) for j in range(self.n)])
        return torch.einsum("jnr,jdr->nd", w * core, self.U)


class TriplexBlockCascade(nn.Module):

    def __init__(self, d_x=896, d_g=512, n_head=8, d_head=64, D=2048, n=5, rank=32, cat="BMMVS", d_mem=None,
                 init_seed=None, blk=0, shared_proj=True, proj_init="indep"):
        super().__init__()
        self.shared_proj = bool(shared_proj)
        bad = [c for c in cat if c not in FAMILIES]
        if bad: raise ValueError("famiglie sconosciute %s (ammesse %s)" % (bad, FAMILIES))
        if init_seed is not None:
            torch.manual_seed(int(init_seed) * 7919 + blk)
        self.init_seed, self.blk = init_seed, blk
        self.cat = cat
        self.d_mem = int(d_mem) if d_mem else d_g
        self.norm_g = RMSNorm(d_x)
        self.P_G = nn.Linear(d_x, self.d_mem, bias=False)
        self.gru = SignedSimplexGRUCell(self.d_mem)
        self.gru.gamma_carrier = True
        if self.d_mem != d_g: self.P_mem = nn.Linear(self.d_mem, d_g, bias=False)
        self.att = CausalMHA(d_g, n_head, d_head)
        self.W_O = nn.Linear(d_g, d_x, bias=False)
        self.ctl = CausalMHAIn(d_x, d_g, n_head, d_head)
        with torch.no_grad(): self.W_O.weight.normal_(0.0, 0.02)
        self.norm_u = RMSNorm(d_x)
        self.P_g = nn.Linear(d_x, D, bias=False)
        self.P_up = nn.Linear(d_x, D, bias=False)
        self.P_dn = nn.Linear(D, d_x, bias=False)
        with torch.no_grad(): self.P_dn.weight.normal_(0.0, 0.02)
        if not self.shared_proj:
            _st = torch.random.get_rng_state()
            g0, u0, d0 = self.P_g, self.P_up, self.P_dn
            gs, us, ds = [g0], [u0], [d0]
            for _ in cat[1:]:
                g_, u_, d_ = nn.Linear(d_x, D, bias=False), nn.Linear(d_x, D, bias=False), nn.Linear(D, d_x, bias=False)
                with torch.no_grad():
                    d_.weight.normal_(0.0, 0.02)
                    if proj_init == "copy": g_.weight.copy_(g0.weight); u_.weight.copy_(u0.weight); d_.weight.copy_(d0.weight)
                gs.append(g_); us.append(u_); ds.append(d_)
            del self.P_g, self.P_up, self.P_dn
            self.P_g, self.P_up, self.P_dn = nn.ModuleList(gs), nn.ModuleList(us), nn.ModuleList(ds)
            torch.random.set_rng_state(_st)
        occ = {}
        brs = []
        for f in cat:
            occ[f] = occ.get(f, 0) + 1
            sd_f = None if init_seed is None else int(init_seed) * 1_000_003 + blk * 9973 + ord(f) * 97 + occ[f]
            brs.append(CascadeBranch(f, D, n, rank, seed=sd_f))
        self.branches = nn.ModuleList(brs)
        self.heads = nn.ModuleList([nn.Linear(d_g, 1 + n) for _ in cat])
        self.register_buffer("gamma", torch.tensor(1.0)); self.register_buffer("eta", torch.tensor(1.0))

    def forward(self, x, cos, sin, stats=False):
        B, L, _ = x.shape
        e = self.P_G(self.norm_g(x))
        er, ez, eh = self.gru.proj(e)
        Urz = torch.cat([self.gru.Ur, self.gru.Uz], 1)
        gb = x.new_zeros(B, self.gru.d_g); gbs, alphas = [], []
        for i in range(L):
            g, gb, aux = self.gru.step(er[:, i], ez[:, i], eh[:, i], gb, Urz)
            gbs.append(gb)
            if stats: alphas.append(aux["alpha"])
        Gb = torch.stack(gbs, 1)
        if hasattr(self, "P_mem"): Gb = self.P_mem(Gb)
        cA, stA = self.att(Gb, cos, sin, stats)
        x = x + self.W_O(cA)
        xp_n = x.reshape(B * L, -1).norm(dim=-1).clamp_min(1e-6)
        fam, stC = [], None
        u_prev, dx_cum = None, None
        for k, br in enumerate(self.branches):
            h = self.norm_u(x)
            cC, stC = self.ctl(h, cos, sin, stats)
            pi = torch.softmax(self.heads[k](cC), dim=-1).reshape(B * L, -1)
            hf = h.reshape(B * L, -1)
            Pg, Pu, Pd = (self.P_g, self.P_up, self.P_dn) if self.shared_proj else (self.P_g[k], self.P_up[k], self.P_dn[k])
            u = F.silu(Pg(hf)) * Pu(hf)
            xprev = x
            d = br.deform(u, pi)
            psi = d if br.fam == "S" else F.silu(d)
            dxk = Pd(psi)
            x = x + dxk.reshape(B, L, -1)
            if stats:
                with torch.no_grad():
                    xprev_n = xprev.reshape(B * L, -1).norm(dim=-1).clamp_min(1e-6)
                    r90u, r99u = _spec(d); r90x, r99x = _spec(dxk)
                    f = dict(fam=br.fam, id=float(pi[:, 0].mean()), eH=float(entropy(pi).exp().mean()),
                             arg=(torch.bincount(pi.argmax(-1), minlength=pi.shape[1]).float() / pi.shape[0]).tolist(),
                             u0=float(u.norm(dim=-1).mean()),
                             du_rel=float((d.norm(dim=-1) / u.norm(dim=-1).clamp_min(1e-6)).mean()),
                             dx_rel=float((dxk.norm(dim=-1) / xp_n).mean()),
                             r90u=r90u, r99u=r99u, r90x=r90x, r99x=r99x,
                             dx_rel_prev=float((dxk.norm(dim=-1) / xprev_n).mean()),
                             nPg=float(Pg.weight.norm()), nPup=float(Pu.weight.norm()), nPdn=float(Pd.weight.norm()),
                             dxk=dxk)
                    if u_prev is not None:
                        f.update(du_prev=float(((u - u_prev).norm(dim=-1) / u_prev.norm(dim=-1).clamp_min(1e-6)).mean()),
                                 cos_u=float(F.cosine_similarity(u, u_prev, dim=-1).mean()))
                    if dx_cum is not None:
                        f.update(cos_cum=float(F.cosine_similarity(dxk, dx_cum, dim=-1).mean()))
                    dx_cum = dxk if dx_cum is None else dx_cum + dxk
                    f.update(cum_rel=float((dx_cum.norm(dim=-1) / xp_n).mean()))
                    if br.fam == "F":
                        qa = u @ br.A[0]; qb = u @ br.Bm[0]
                        f.update(rmsA=float(qa.pow(2).mean().sqrt()), rmsB=float(qb.pow(2).mean().sqrt()),
                                 rmsAB=float((F.silu(qa) * qb).pow(2).mean().sqrt()))
                    if br.fam == "V":
                        q = u @ br.A[2]
                        f.update(om=(float(br.omega.mean()), float(br.omega.std())), nu=(float(br.nu.mean()), float(br.nu.std())),
                                 msin=float(torch.sin(br.omega[2] * q).abs().mean()),
                                 mcos=float((1 - torch.cos(br.nu[3] * (u @ br.A[3]))).abs().mean()))
                    fam.append(f)
            u_prev = u.detach()
        st = None
        if stats:
            with torch.no_grad():
                A = torch.stack(alphas, 1); dxs = [f.pop("dxk") for f in fam]
                nb = len(dxs)
                cosm = [[float(F.cosine_similarity(dxs[i], dxs[j], dim=-1).mean()) for j in range(nb)] for i in range(nb)]
                st = dict(alpha_ent=entropy(A).mean().item(), alpha_neff=entropy(A).exp().mean().item(),
                          pi_ent=0.0, pi_top1=0.0, pi_use=[], pi_id=[f["id"] for f in fam],
                          gamma=1.0, eta=1.0, kappa=self.gru.kappa.item(), bsharp=self.gru.beta.item(),
                          gcar=(self.gru.gamma.mean().item(), self.gru.gamma.std().item()),
                          dx=float(torch.stack(dxs).sum(0).norm(dim=-1).mean()), beta=stA, delta=stC,
                          fam=fam, cos=cosm,
                          ops=dict(dR=fam[0]["du_rel"], dD=fam[1]["du_rel"], dS=fam[-1]["du_rel"],
                                   absR=0.0, absD=0.0, ang=0.0, lam=0.0, shr=0.0, rot_norm=1.0, gain=1.0))
        return x, st


class TriplexCascadeNTP(nn.Module):

    def __init__(self, vocab=64000, d_x=1024, S=4, d_g=512, n_head=8, d_head=64, D=4096, n=5,
                 rank=32, cat="FBMMVMS", d_mem=None, init_seed=None, shared_proj=True, proj_init="indep", **_):
        super().__init__()
        if init_seed is not None: torch.manual_seed(int(init_seed))
        self.V, self.d_x, self.cat = int(vocab), int(d_x), cat
        self.tok = nn.Embedding(self.V, self.d_x)
        with torch.no_grad(): self.tok.weight.normal_(0.0, 0.02)
        self.blocks = nn.ModuleList([
            TriplexBlockCascade(self.d_x, d_g, n_head, d_head, D, n, rank, cat, d_mem,
                                init_seed=init_seed, blk=b, shared_proj=shared_proj, proj_init=proj_init) for b in range(S)])
        self.shared_proj = bool(shared_proj)
        self.norm_f = RMSNorm(self.d_x)
        self.d_head = d_head

    def forward(self, ids, stats=False):
        x = self.tok(ids).float()
        cos, sin = rope_cache(ids.shape[1], self.d_head, ids.device, x.dtype)
        sts, xs = [], [x]
        for blk in self.blocks:
            x, st = blk(x, cos, sin, stats)
            xs.append(x)
            if stats: sts.append(st)
        return self.readout(x), sts, xs

    def readout(self, x):
        return F.linear(self.norm_f(x), self.tok.weight)

    def loss(self, ids, mask=None, stats=False):
        z, sts, xs = self(ids, stats)
        ce = F.cross_entropy(z[:, :-1].reshape(-1, self.V), ids[:, 1:].reshape(-1), reduction="none")
        if mask is not None:
            m = mask[:, 1:].reshape(-1).float(); ce = (ce * m).sum() / m.sum().clamp_min(1.0)
        else:
            ce = ce.mean()
        return ce, z, sts, xs

    def param_breakdown(self):
        g = {"embedding": 0, "memory_attention": 0, "projectors": 0, "operators": 0, "heads_norms": 0}
        for nme, prm in self.named_parameters():
            if not prm.requires_grad: continue
            k = prm.numel()
            if nme.startswith("tok."): g["embedding"] += k
            elif ".branches." in nme: g["operators"] += k
            elif any(t in nme for t in (".P_g.", ".P_up.", ".P_dn.")): g["projectors"] += k
            elif any(t in nme for t in (".gru.", ".P_G.", ".att.", ".W_O.", ".ctl.", ".P_mem.")): g["memory_attention"] += k
            else: g["heads_norms"] += k
        g["cascade"] = g["projectors"] + g["operators"] + g["heads_norms"]
        g["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return g
