"""CASC tests: mathematical, regression and integration checks of the five-file machine.

    python test_casc.py              runs every test whose resources are available

Optional resources, passed through environment variables:
    CASC_CKPT=.../NTP_A_4B_final.pt  strict load of the published checkpoint and its reference CE on
                                     fixed token ids
    CASC_TOK=.../tok64k.json         inference: greedy decoding and generate.py (extrap, stream, sample)
    CASC_LEGACY=/path/to/originals   bitwise equivalence with the original triplex*.py sources
    CASC_DDP=1                       two-process data-parallel run of train.py on CPU (gloo backend)
A test whose resource is missing is reported as NOT TESTED, never as passed (pytest reports it as
skipped). The exit code is 1 if any test fails. The reference numbers below were produced by the
original sources listed in REF_REVISION, before they were merged into casc.py.
"""
import hashlib, json, math, os, subprocess, sys, tempfile, time, traceback, unittest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casc
from casc import TriplexCascadeNTP, CascadeBranch, rope_cache, rms

torch.set_num_threads(min(8, os.cpu_count() or 1))

SMALL = dict(vocab=256, d_x=64, d_g=32, n_head=4, d_head=8, D=128, rank=4, S=2, n=5, cat="FBMMVMS")
FULL = dict(vocab=64000, d_x=1024, d_g=512, n_head=8, d_head=64, D=4096, rank=32, S=4, n=5, cat="FBMMVMS")
REF_REVISION = {"triplex.py": "1a3c2a2511ad95a4834db0d5a20bb950", "triplex_cpost.py": "c022f9e669fd20e8b46bd125ff2b521a",
                "triplex_multi.py": "66d1ba1e818df168ba380ff940f80605", "pretrain_ntp.py": "f0b04ba3b9712b9d168326f2f67bf822",
                "gen_ntp.py": "8a7d4ecc9aff21b07124ff3b68d5b72d", "build_tokenizer_64k.py": "efe959f42a5a0ec7a8bcf41075d27c43",
                "pack_ntp_corpus.py": "1425efbb78eb95bdb1ae0147cf39d307", "check_corpus.py": "6308d093c4213887002d02c32ecc3d1a"}
REF = {
    "shared": dict(ce=5.5580735206604, zsum=2.79768761187961, zabs=1737.0475582144122, gnorm=2.842949697563491,
                   after=580.0138056502838, stats_sum=1051.9468098181824, nparam=184806, nkeys=138),
    "unshared": dict(ce=5.55804443359375, zsum=2.9818687165629854, zabs=1737.2063744967695, gnorm=2.8396653899293063,
                     after=626.7005090462862, stats_sum=1025.3949685764237, nparam=479718, nkeys=174),
    "full": dict(nparam=175218812, nkeys=274, keys_md5="5ec61624c0c9a311a21ee36290811c19",
                 breakdown=dict(embedding=65536000, memory_attention=19931220, projectors=50331648, operators=39324544,
                                heads_norms=95400, cascade=89751592, total_trainable=175218812)),
    "ckpt_final": dict(md5="864aa26235b68b584c8be63add0b06cf", ce_random_ids=13.249781608581543),
}


class NotTested(unittest.SkipTest):
    pass


TESTS = []
def test(fn):
    TESTS.append(fn); return fn


def close(a, b, rtol=1e-5, atol=0.0):
    return abs(a - b) <= atol + rtol * abs(b)


def flat(v, p="st", out=None):
    out = {} if out is None else out
    if isinstance(v, dict): [flat(x, p + "." + k, out) for k, x in v.items()]
    elif isinstance(v, (list, tuple)): [flat(x, p + "[%d]" % i, out) for i, x in enumerate(v)]
    elif isinstance(v, (int, float)) and not isinstance(v, bool): out[p] = float(v)
    return out


def small(shared=True):
    torch.manual_seed(0)
    return TriplexCascadeNTP(**SMALL, init_seed=1234, shared_proj=shared)


def small_ids():
    return torch.randint(0, 256, (3, 17), generator=torch.Generator().manual_seed(99))


def md5file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def run(args, cwd, env=None, timeout=900):
    e = dict(os.environ if env is None else env)
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        if env is None or k not in env: e.pop(k, None)
    e["PYTHONPATH"] = HERE + os.pathsep + e.get("PYTHONPATH", "")
    return subprocess.run(args, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)


def load_full(path):
    m = TriplexCascadeNTP(**FULL)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.eval()


@test
def test_regression_small():
    out = []
    for tag, shared in (("shared", True), ("unshared", False)):
        r, m, ids = REF[tag], small(shared), small_ids()
        z, _, _ = m(ids)
        ce = F.cross_entropy(z[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)); ce.backward()
        gnorm = math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in m.parameters() if p.grad is not None))
        _, sts, _ = m(ids, stats=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        zabs = float(z.double().abs().sum())
        got = dict(ce=(float(ce), 1e-5, 0.0), zsum=(float(z.double().sum()), 0.0, 1e-6 * zabs), zabs=(zabs, 1e-5, 0.0),
                   gnorm=(gnorm, 1e-5, 0.0), after=(sum(float(p.detach().double().sum()) for p in m.parameters()), 1e-6, 1e-4),
                   stats_sum=(sum(flat(sts).values()), 1e-5, 1e-4))
        for k, (v, rt, at) in got.items():
            assert close(v, r[k], rt, at), "%s %s: %r, reference %r" % (tag, k, v, r[k])
        assert sum(p.numel() for p in m.parameters()) == r["nparam"] and len(m.state_dict()) == r["nkeys"]
        out.append("%s CE %.8f" % (tag, float(ce)))
    return "; ".join(out) + " (projectors shared per block / per stage)"


@test
def test_full_size_names_shapes_count():
    m = TriplexCascadeNTP(**FULL)
    n = sum(p.numel() for p in m.parameters())
    keys = "\n".join("%s:%s" % (k, ",".join(map(str, t.shape))) for k, t in m.state_dict().items())
    assert n == REF["full"]["nparam"], n
    assert len(m.state_dict()) == REF["full"]["nkeys"] and hashlib.md5(keys.encode()).hexdigest() == REF["full"]["keys_md5"], \
        "state-dict names or shapes differ from the published checkpoint"
    assert m.param_breakdown() == REF["full"]["breakdown"], m.param_breakdown()
    return "%s parameters, %d state-dict entries" % (format(n, ","), len(m.state_dict()))


@test
def test_tied_embedding_and_readout():
    m = small(); E = m.tok.weight
    big = [p for p in m.parameters() if p.shape == E.shape]
    assert len(big) == 1 and big[0] is E
    ids = small_ids()
    assert torch.equal(m.tok(ids), E[ids])
    x = torch.randn(2, 5, SMALL["d_x"])
    assert torch.allclose(m.readout(x), m.norm_f(x) @ E.T, atol=1e-5)
    return "one (V, d_x) parameter serves input and output"


@test
def test_gru_signed_simplex():
    torch.manual_seed(1); g = casc.SignedSimplexGRUCell(32); g.gamma_carrier = True
    with torch.no_grad(): g.gamma.uniform_(0.5, 1.5)
    e, gp = torch.randn(4, 32), torch.randn(4, 32)
    with torch.no_grad():
        go, gbar, aux = g(e, gp)
        r = F.softplus(e @ g.Wr + g.br + gp @ g.Ur)
        z = torch.sigmoid(e @ g.Wz + g.bz + gp @ g.Uz)
        h = torch.tanh(e @ g.Wh + g.bh + (r * gp) @ g.Uh)
        graw = (1 - z) * gp + z * h
        s = torch.tanh(g.beta * graw)
        q = F.softplus(g.kappa * torch.sqrt(graw * graw + g.eps)); al = q / q.sum(-1, keepdim=True)
        r_big = F.softplus((3 * torch.randn(64, 32)) @ g.Wr + (3 * torch.randn(64, 32)) @ g.Ur + g.br)
    assert torch.allclose(r, aux["r"], atol=1e-5) and torch.allclose(z, aux["z"], atol=1e-5)
    assert float(r_big.max()) > 1.0 and float(r.min()) > 0.0, "the reset gate is not confined to (0, 1)"
    assert torch.allclose(graw, aux["g_raw"], atol=1e-5)
    assert torch.allclose(al, aux["alpha"], atol=1e-5) and bool((al > 0).all()) and torch.allclose(al.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(go, s * al, atol=1e-5) and float(go.abs().sum(-1).max()) < 1.0
    assert torch.allclose(gbar, g.gamma * rms(go, g.eps), atol=1e-5)
    assert g.gru_norm == 0
    return "g = tanh(beta g_raw) * alpha, alpha on the simplex, ||g||_1 < 1, carrier gamma * rms(g)"


@test
def test_attention_causality():
    att = casc.CausalMHA(32, 4, 8); ctl = casc.CausalMHAIn(64, 32, 4, 8); cos, sin = rope_cache(10, 8, "cpu", torch.float32)
    for mod, d in ((att, 32), (ctl, 64)):
        a = torch.randn(2, 10, d); b = a.clone(); b[:, 6:] = torch.randn(2, 4, d)
        with torch.no_grad(): ya, yb = mod(a, cos, sin)[0], mod(b, cos, sin)[0]
        assert torch.equal(ya[:, :6], yb[:, :6]) and not torch.allclose(ya[:, 6:], yb[:, 6:])
    return "outputs before an edited position are bit-identical"


@test
def test_model_causality():
    m = small().eval(); ids = small_ids(); ids2 = ids.clone(); ids2[:, 9:] = (ids2[:, 9:] + 17) % 256
    with torch.no_grad(): a, b = m(ids)[0], m(ids2)[0]
    assert torch.equal(a[:, :9], b[:, :9]) and not torch.allclose(a[:, 9:], b[:, 9:])
    return "positions 0-8 bit-identical after editing tokens 9-16"


@test
def test_identity_slot_and_convex_mixture():
    pi0 = torch.zeros(8, 6); pi0[:, 0] = 1.0
    for f in "FBMVS":
        br = CascadeBranch(f, D=64, n=5, rank=4, seed=1); u = torch.randn(8, 64)
        with torch.no_grad():
            assert float(br.deform(u, pi0).abs().max()) == 0.0, f
            pr = torch.softmax(torch.randn(8, 6), -1)
            man = sum(pr[:, 1 + j:2 + j] * br.deform(u, F.one_hot(torch.full((8,), 1 + j), 6).float()) for j in range(5))
            assert torch.allclose(br.deform(u, pr), man, atol=1e-5), f
    return "all five families"


@test
def test_operator_families():
    D, r, j = 64, 4, 1; torch.manual_seed(2); u = torch.randn(16, D) * 0.5
    oh = lambda k: F.one_hot(torch.full((16,), 1 + k), 6).float()
    with torch.no_grad():
        b = CascadeBranch("F", D=D, n=5, rank=r, seed=2)
        assert torch.allclose(b.deform(u, oh(j)), (F.silu(u @ b.A[j]) * (u @ b.Bm[j])) @ b.U[j].T, atol=1e-5)
        b = CascadeBranch("B", D=D, n=5, rank=r, seed=3); Q, _ = torch.linalg.qr(b.T); Np = b.N - Q @ (Q.transpose(1, 2) @ b.N)
        assert torch.allclose(b.deform(u, oh(j)), F.silu(u @ b.T[j]) @ Np[j].T, atol=1e-5)
        assert float((b.T[j].T @ Np[j]).abs().max()) < 1e-5, "N_perp is orthogonal to span(T_j)"
        b = CascadeBranch("S", D=D, n=5, rank=r, seed=4)
        assert torch.allclose(b.deform(u, oh(j)), (u @ b.V[j]) @ b.U[j].T, atol=1e-5)
        b = CascadeBranch("M", D=D, n=5, rank=r, seed=5)
        lawM = [torch.tanh, lambda q: q / (1 + q.abs()), torch.asinh, lambda q: q * torch.sigmoid(b.beta_gl * q.abs()), F.silu]
        for k in range(5):
            assert torch.allclose(b.deform(u, oh(k)), ((u @ b.U[k]) * lawM[k](u @ b.A[k])) @ b.U[k].T, atol=1e-5), "M %d" % k
        v = CascadeBranch("V", D=D, n=5, rank=r, seed=6)
        lawV = [torch.tanh, torch.asinh, lambda q: torch.sin(v.omega[2] * q), lambda q: 1 - torch.cos(v.nu[3] * q),
                lambda q: v.c_sin[4] * torch.sin(v.omega[4] * q) + v.c_cos[4] * (1 - torch.cos(v.nu[4] * q))]
        for k in range(5):
            assert torch.allclose(v.deform(u, oh(k)), ((u @ v.U[k]) * lawV[k](u @ v.A[k])) @ v.U[k].T, atol=1e-5), "V %d" % k
    assert casc.FAMILIES == "FBSMV" and casc.F_INIT_GAIN == 6.0
    return "F, B (with N_perp), S, five M laws, five V laws"


@test
def test_block_composition():
    for shared in (True, False):
        blk = casc.TriplexBlockCascade(d_x=64, d_g=32, n_head=4, d_head=8, D=128, n=5, rank=4, cat="FBMMVMS",
                                       init_seed=5, blk=0, shared_proj=shared)
        with torch.no_grad():
            for q in blk.branches.parameters(): q.mul_(3.0)
            for P in (blk.P_dn if isinstance(blk.P_dn, torch.nn.ModuleList) else [blk.P_dn]): P.weight.normal_(0.0, 0.3)
        x = torch.randn(2, 9, 64) * 0.5; cos, sin = rope_cache(9, 8, "cpu", torch.float32)

        def ref(psi_s_silu=False, frozen_control=False):
            e = blk.P_G(blk.norm_g(x)); gb = x.new_zeros(2, 32); G = []
            for t in range(9):
                _, gb, _ = blk.gru(e[:, t], gb); G.append(gb)
            xr = x + blk.W_O(blk.att(torch.stack(G, 1), cos, sin)[0]); h0 = blk.norm_u(xr)
            for k, br in enumerate(blk.branches):
                h = blk.norm_u(xr)
                pi = torch.softmax(blk.heads[k](blk.ctl(h0 if frozen_control else h, cos, sin)[0]), -1).reshape(18, -1)
                Pg, Pu, Pd = (blk.P_g, blk.P_up, blk.P_dn) if shared else (blk.P_g[k], blk.P_up[k], blk.P_dn[k])
                hf = h.reshape(18, -1); d = br.deform(F.silu(Pg(hf)) * Pu(hf), pi)
                xr = xr + Pd(d if (br.fam == "S" and not psi_s_silu) else F.silu(d)).reshape(2, 9, -1)
            return xr
        with torch.no_grad():
            y = blk(x, cos, sin)[0]; good = ref()
            err = float((y - good).abs().max()); assert torch.allclose(y, good, rtol=1e-5, atol=1e-5), err
            d_psi = float((y - ref(psi_s_silu=True)).abs().max()); d_ctl = float((y - ref(frozen_control=True)).abs().max())
            assert d_psi > 1e3 * max(err, 1e-7) and d_ctl > 1e3 * max(err, 1e-7), (err, d_psi, d_ctl)
    return "max error %.1e; SiLU on S would move the output by %.1e, a control frozen at stage 0 by %.1e" % (err, d_psi, d_ctl)


@test
def test_sharing_and_constants():
    m = small(True)
    for b in m.blocks:
        assert isinstance(b.P_g, torch.nn.Linear) and not any(hasattr(br, a) for br in b.branches for a in ("P_g", "P_up", "P_dn"))
        assert b.ctl is not b.att and tuple(b.ctl.Wq.weight.shape) == (32, 64) and len(b.heads) == 7 == len(b.branches)
        assert [br.fam for br in b.branches] == list("FBMMVMS")
        bufs = dict(b.named_buffers())
        assert float(bufs["gamma"]) == 1.0 and float(bufs["eta"]) == 1.0 and "gamma" not in dict(b.named_parameters())
    for b in small(False).blocks:
        assert isinstance(b.P_g, torch.nn.ModuleList) and len(b.P_g) == len(b.P_up) == len(b.P_dn) == 7
    return "shared_proj=True: one triplet per block; shared_proj=False: one per stage"


@test
def test_gradients():
    m = small(); ce, _, _, _ = m.loss(small_ids()); ce.backward()
    none = sorted(n for n, p in m.named_parameters() if p.grad is None)
    expect = sorted("blocks.%d.gru.%s" % (b, k) for b in range(SMALL["S"]) for k in ("beta_r", "kappa_r", "rho_log"))
    assert none == expect, none
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0, n
    return "%d tensors; the 3 scalars of the unused GRU mode (gru_norm=0) get none and stay registered" % len(list(m.parameters()))


@test
def test_legacy_equivalence():
    path = os.environ.get("CASC_LEGACY")
    if not path: raise NotTested("set CASC_LEGACY to the directory holding the original triplex*.py")
    sys.path.insert(0, path)
    import importlib; tm = importlib.import_module("triplex_multi")
    rev = {f: hashlib.md5(open(os.path.join(path, f), "rb").read()).hexdigest() for f in ("triplex.py", "triplex_cpost.py", "triplex_multi.py")}
    assert all(rev[f] == REF_REVISION[f] for f in rev), "CASC_LEGACY is not the reference revision: %s" % rev
    for shared in (True, False):
        torch.manual_seed(0); a = tm.TriplexCascadeNTP(**SMALL, init_seed=1234, shared_proj=shared)
        b = small(shared); sa, sb = a.state_dict(), b.state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), "initialization differs"
        ids = small_ids()
        for stats in (False, True):
            za, sta, xa = a(ids, stats=stats); zb, stb, xb = b(ids, stats=stats)
            assert torch.equal(za, zb) and all(torch.equal(p, q) for p, q in zip(xa, xb)) and flat(sta) == flat(stb)
        for mm in (a, b):
            z = mm(ids)[0]; F.cross_entropy(z[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)).backward()
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            assert (p.grad is None) == (q.grad is None) and (p.grad is None or torch.equal(p.grad, q.grad)), n
        for mm in (a, b):
            opt = torch.optim.AdamW(mm.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
            torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0); opt.step()
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    hashes = []
    for cls in (tm.TriplexCascadeNTP, TriplexCascadeNTP):
        mm = cls(**FULL, init_seed=20260916)
        hashes.append([hashlib.md5(t.numpy().tobytes()).hexdigest() for t in mm.state_dict().values()]); del mm
    assert hashes[0] == hashes[1], "full-size initialization differs"
    return "small: init/forward/stats/grad/AdamW bit-identical (both projector modes); full size: init bit-identical"


@test
def test_checkpoint_strict_load():
    ck = os.environ.get("CASC_CKPT")
    if not ck: raise NotTested("UNVERIFIED: checkpoint not supplied (set CASC_CKPT)")
    files = [ck]
    m = TriplexCascadeNTP(**FULL); names = []
    for p in files:
        h0 = md5file(p); sd = torch.load(p, map_location="cpu", weights_only=True); sd = sd.get("model", sd)
        r = m.load_state_dict(sd, strict=True)
        assert not r.missing_keys and not r.unexpected_keys
        msd = m.state_dict(); assert set(msd) == set(sd) and all(torch.equal(msd[k], sd[k]) for k in sd)
        assert md5file(p) == h0
        names.append(os.path.basename(p)); del sd
    note = ""
    if md5file(ck) == REF["ckpt_final"]["md5"]:
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True); m.eval()
        ids = torch.randint(0, 64000, (2, 96), generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            z = m(ids)[0]; ce = float(F.cross_entropy(z[:, :-1].reshape(-1, 64000), ids[:, 1:].reshape(-1)))
        assert close(ce, REF["ckpt_final"]["ce_random_ids"], 1e-4), ce
        note = "; published NTP_A_4B_final.pt: CE %.6f on fixed ids = reference" % ce
    return ", ".join(names) + note


def _prompt_ids(tok):
    return tok.encode("The capital of France is").ids


@test
def test_inference_greedy():
    ck, tk = os.environ.get("CASC_CKPT"), os.environ.get("CASC_TOK")
    if not (ck and tk): raise NotTested("set CASC_CKPT and CASC_TOK")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tk); m = load_full(ck); ids = _prompt_ids(tok)
    with torch.no_grad():
        for _ in range(12):
            z = m(torch.tensor([ids]))[0][0, -1]; assert bool(torch.isfinite(z).all()); ids.append(int(z.argmax()))
    return repr(tok.decode(ids))


@test
def test_generate_cli():
    ck, tk = os.environ.get("CASC_CKPT"), os.environ.get("CASC_TOK")
    if not (ck and tk): raise NotTested("set CASC_CKPT and CASC_TOK")
    para = ("The river runs through the old town and under seven stone bridges. Each bridge was built in a "
            "different century, and the people who cross them every morning rarely notice the difference. ")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "heldout", "wiki_validation"))
        with open(os.path.join(d, "heldout", "wiki_validation", "a.jsonl"), "w") as f:
            for i in range(2):
                f.write(json.dumps({"text": " ".join("%d. %s" % (k, para) for k in range(6 + i)), "domain": "dolma_wiki"}) + "\n")
        base = dict(os.environ, CK=ck, TOK=tk, TXT=os.path.join(d, "heldout"), LW="128", NPD="2", BS="8")
        out = []
        for mode, extra, marker in (("extrap", {}, "CE per bin"), ("stream", {}, "streaming GRU continua"),
                                    ("sample", dict(NEW="12", PROMPT_TOK="16", NOBAN="1", REP="1.0"), "DOMINIO dolma_wiki")):
            r = run([sys.executable, os.path.join(HERE, "generate.py")], cwd=d, env=dict(base, MODE=mode, **extra))
            assert r.returncode == 0 and marker in r.stdout, (mode, r.stdout[-400:], r.stderr[-800:])
            out.append(mode)
    return "exit 0 for " + ", ".join(out)


@test
def test_import_safety_and_help():
    with tempfile.TemporaryDirectory() as d:
        r = run([sys.executable, "-c", "import casc, train, data, generate"], cwd=d)
        assert r.returncode == 0 and r.stdout.strip() == "" and os.listdir(d) == [], (r.stdout, r.stderr[-400:])
        for args in (["train.py", "--help"], ["generate.py", "--help"], ["data.py", "--help"], ["data.py", "tokenizer", "--help"],
                     ["data.py", "pack", "--help"], ["data.py", "check", "--help"]):
            r = run([sys.executable, os.path.join(HERE, args[0])] + args[1:], cwd=d)
            assert r.returncode == 0 and r.stdout.strip(), (args, r.stderr[-400:])
        assert os.listdir(d) == []
    return "casc, train, data, generate; 6 help screens"


def tiny_corpus(d):
    import random
    rnd = random.Random(0); words = ["alpha", "beta", "gamma", "delta", "river", "stone", "bridge", "light", "north", "seven",
                                     "def", "return", "for", "in", "range", "print", "if", "else", "x", "y", "=", "+", "(", ")", ":"]
    def doc(n): return " ".join(rnd.choice(words) for _ in range(n))
    sh = os.path.join(d, "shards"); os.makedirs(sh)
    docs = [doc(rnd.randint(60, 160)) for _ in range(120)]
    for nm in ("code", "web"):
        with open(os.path.join(sh, nm + ".jsonl"), "w") as f: f.writelines(json.dumps({"text": t}) + "\n" for t in docs)
    for nm in ("vcode", "vweb"):
        with open(os.path.join(sh, nm + ".jsonl"), "w") as f:
            f.writelines(json.dumps({"text": doc(120), "domain": "syn|" + nm}) + "\n" for _ in range(10))
    corpus = os.path.join(d, "corpus"); tokf = os.path.join(corpus, "tok64k.json")
    r = run([sys.executable, os.path.join(HERE, "data.py"), "tokenizer", "--code", sh + "/code.jsonl", "--noncode", sh + "/web.jsonl",
             "--out", tokf, "--vocab", "400"], cwd=d)
    assert r.returncode == 0 and "round-trip byte-esatto" in r.stdout, r.stderr[-600:]
    r = run([sys.executable, os.path.join(HERE, "data.py"), "pack", "--tok", tokf, "--out", corpus, "--code", sh + "/code.jsonl",
             "--noncode", sh + "/web.jsonl", "--val-code", sh + "/vcode.jsonl", "--val-noncode", sh + "/vweb.jsonl",
             "--code-tokens", "1e9", "--noncode-tokens", "1e9", "--val-L", "32", "--val-per-domain", "8"], cwd=d)
    assert r.returncode == 0, (r.stdout[-600:], r.stderr[-600:])
    return corpus


TINY_TRAIN = dict(VOCAB="400", DX="64", D="128", OPRANK="4", N_BLOCKS="1", DG="32", NH="4", L="32", BS="4", WARM="2", VEVERY="10")


@test
def test_data_pipeline():
    with tempfile.TemporaryDirectory() as d:
        corpus = tiny_corpus(d)
        meta = json.load(open(os.path.join(corpus, "meta.json")))
        assert meta["eos_id"] == 1 and meta["pad_id"] == 2 and meta["dtype"] == "uint16" and abs(meta["code_frac"] - 0.5) < 1e-9
        r = run([sys.executable, os.path.join(HERE, "data.py"), "check", corpus], cwd=d)
        assert r.returncode == 1 and "FALLITO vocab 64000" in r.stdout and "OK   EOS=1 PAD=2" in r.stdout \
            and "INFO tokenizer diverso da" in r.stdout and "FALLITO tokenizer identico" not in r.stdout, r.stdout[-800:]
        r = run([sys.executable, os.path.join(HERE, "data.py"), "check", corpus, "--frozen-tokenizer"], cwd=d)
        assert "FALLITO tokenizer identico" in r.stdout, r.stdout[-600:]
        env = dict(os.environ, DATA=corpus, TOKENS="3000", OUT=os.path.join(d, "run"), TAG="pipe", **TINY_TRAIN)
        r = run([sys.executable, os.path.join(HERE, "train.py")], cwd=d, env=env)
        log = open(os.path.join(d, "run", "pipe.log")).read() if os.path.exists(os.path.join(d, "run", "pipe.log")) else ""
        assert r.returncode == 0 and "FINE pipe" in log, (r.stderr[-800:], log[-400:])
    return "BPE round trip, uint16 streams, meta/val_dom, check (reference md5 optional), 24 training steps"


@test
def test_train_smoke_extension_resume():
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, SELFTEST="1", OUT=os.path.join(d, "run"), TAG="smoke", LEXT="128", TEXT="8000")
        r1 = run([sys.executable, os.path.join(HERE, "train.py")], cwd=d, env=env)
        log = open(os.path.join(d, "run", "smoke.log")).read()
        assert r1.returncode == 0 and "FINE smoke" in log, (r1.stderr[-800:], log[-400:])
        assert "FASE DI ESTENSIONE a L=128" in log and "[L=128]" in log
        for f in ("smoke_state.pt", "smoke_best.pt", "smoke_pre_ext.pt", "smoke_final.pt"):
            assert os.path.exists(os.path.join(d, "run", f)), f
        r2 = run([sys.executable, os.path.join(HERE, "train.py")], cwd=d, env=env)
        log2 = open(os.path.join(d, "run", "smoke.log")).read()[len(log):]
        assert r2.returncode == 0 and "RESUME da" in log2 and "FINE smoke" in log2, (r2.stderr[-600:], log2[-400:])
        m = TriplexCascadeNTP(vocab=512, d_x=128, S=2, d_g=512, n_head=8, d_head=64, D=256, n=5, rank=8, cat="FBMMVMS")
        m.load_state_dict(torch.load(os.path.join(d, "run", "smoke_final.pt"), weights_only=True), strict=True)
        best = [l for l in log.split("\n") if l.startswith("FINE")][0]
    return best.split("|")[2].strip() + "; resume and strict reload ok"


@test
def test_ddp_cpu():
    if os.environ.get("CASC_DDP") != "1": raise NotTested("set CASC_DDP=1; the 6-GPU NCCL launcher itself needs GPUs")
    with tempfile.TemporaryDirectory() as d:
        corpus = tiny_corpus(d)
        env = dict(os.environ, DATA=corpus, TOKENS="6000", OUT=os.path.join(d, "run"), TAG="ddp", OMP_NUM_THREADS="2", **TINY_TRAIN)
        r = run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29731",
                 os.path.join(HERE, "train.py")], cwd=d, env=env)
        log = open(os.path.join(d, "run", "ddp.log")).read() if os.path.exists(os.path.join(d, "run", "ddp.log")) else ""
        assert r.returncode == 0 and "FINE ddp" in log and "2 ranghi" in log, (r.stderr[-1500:], log[-400:])
        r2 = run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29732",
                  os.path.join(HERE, "train.py")], cwd=d, env=env)
        log2 = open(os.path.join(d, "run", "ddp.log")).read()[len(log):]
        assert r2.returncode == 0 and "RESUME da" in log2, (r2.stderr[-1500:], log2[-400:])
    return "2 ranks, gloo: training, validation on rank 0, checkpoints, resume"


def main():
    rows, t0 = [], time.time()
    for fn in TESTS:
        t = time.time()
        try:
            info, status = fn() or "", "PASS"
        except unittest.SkipTest as e:
            info, status = str(e), "NOT TESTED"
        except Exception as e:
            info, status = "%s: %s" % (type(e).__name__, str(e)[:300]), "FAIL"; traceback.print_exc()
        rows.append((fn.__name__, status, time.time() - t, info))
        print("%-38s %-10s %6.1fs  %s" % rows[-1], flush=True)
    n = {s: sum(r[1] == s for r in rows) for s in ("PASS", "FAIL", "NOT TESTED")}
    print("\n%d passed, %d failed, %d not tested  (%.0f s, torch %s)" % (n["PASS"], n["FAIL"], n["NOT TESTED"], time.time() - t0, torch.__version__))
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
