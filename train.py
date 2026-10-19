"""CASC next-token trainer: from scratch, plain cross-entropy, standard backpropagation, AdamW.

The task is next-token prediction; the loss is the cross-entropy of the next token, minimized with
ordinary reverse-mode autograd (loss.backward()) and torch.optim.AdamW(betas=(0.9, 0.95),
weight_decay=0.01), global gradient-norm clipping and a warmup / constant / linear-decay (WSD)
learning-rate schedule.

Corpus expected in $DATA (written by `python data.py pack`), uint16 tokens:
    train_code.bin  train_noncode.bin     separate streams, every document closed by EOS
    val_code.bin    val_noncode.bin       held-out, disjoint from train by document hash
    meta.json       {vocab, eos_id, pad_id, tokens, ...}
    val_dom.json    optional sub-domain label for every validation window (per-domain MACRO)

The code/non-code mixer is deterministic (Bresenham counter on CODE_FRAC): with the same SEED two runs
see exactly the same tokens in the same order, also across a resume. Configuration is read from
environment variables (defaults at the top of the main block):
    FAMILY_SEQ VOCAB DX D OPRANK N_BLOCKS DG NH NOPS    model shape; OPRANK is the operator rank
                                                        (RANK is reserved for the torchrun process rank)
    L BS ACC LR WARM DECAY CLIP TOKENS CODE_FRAC SEED   protocol; TOKENS counts predicted tokens
    VTOK VEVERY RESUME DATA OUT TAG COMPILE             validation cadence, checkpoints, logging
    LEXT TEXT BSEXT                                     final context-extension phase (e.g. 96 -> 384)
    SHARED_PROJ PROJ_INIT                               projectors shared per block (default) or per stage
Checkpoints in $OUT: TAG_state.pt (full resume state), TAG_best.pt (best validation MACRO),
TAG_pre_ext.pt (before the extension phase), TAG_final.pt (last step).

Usage:  SELFTEST=1 python train.py                   tiny model on synthetic data, no corpus needed
        DATA=corpus TAG=run TOKENS=4e9 python train.py
        torchrun --nproc_per_node=6 train.py          data parallel; README lists the NTP_A_4B settings
"""
import json, math, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from casc import TriplexCascadeNTP

if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__); sys.exit(0)

    FAMILY_SEQ = os.environ.get("FAMILY_SEQ", "FBMMVMS")
    VOCAB = int(os.environ.get("VOCAB", "64000"))
    DX    = int(os.environ.get("DX", "1024"))
    Dop   = int(os.environ.get("D", "4096"))
    RANK  = int(os.environ.get("OPRANK", os.environ.get("RANK", "32") if "WORLD_SIZE" not in os.environ else "32"))
    S     = int(os.environ.get("N_BLOCKS", "4"))
    DG    = int(os.environ.get("DG", "512"))
    NH    = int(os.environ.get("NH", "8"))
    NOPS  = int(os.environ.get("NOPS", "5"))
    L     = int(os.environ.get("L", "512"))
    BS    = int(os.environ.get("BS", "12"))
    LR    = float(os.environ.get("LR", "3e-4"))
    TOKENS = float(os.environ.get("TOKENS", "4e9"))
    CODE_FRAC = float(os.environ.get("CODE_FRAC", "0.30"))
    WARM  = int(os.environ.get("WARM", "3000"))
    DECAY = float(os.environ.get("DECAY", "0.15"))
    VEVERY = int(os.environ.get("VEVERY", "2000"))
    VTOK   = float(os.environ.get("VTOK", "0"))
    RESUME = int(os.environ.get("RESUME", "1"))
    SEED  = int(os.environ.get("SEED", "20260916"))
    DATA  = os.environ.get("DATA", "ntp_v1")
    OUT   = os.environ.get("OUT", "run_ntp")
    TAG   = os.environ.get("TAG", "ntp_%s" % FAMILY_SEQ)
    SELFTEST = int(os.environ.get("SELFTEST", "0"))
    CLIP  = float(os.environ.get("CLIP", "1.0"))
    SHARED_PROJ = int(os.environ.get("SHARED_PROJ", "1"))
    PROJ_INIT = os.environ.get("PROJ_INIT", "indep")
    ACC   = int(os.environ.get("ACC", "1"))
    LEXT  = int(os.environ.get("LEXT", "0"))
    TEXT  = float(os.environ.get("TEXT", "0"))
    BSEXT = int(os.environ.get("BSEXT", "0"))
    import torch.distributed as dist
    DDP = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    WORLD = int(os.environ.get("WORLD_SIZE", "1")) if DDP else 1
    PRANK = int(os.environ.get("RANK", "0")) if DDP else 0
    LOCAL = int(os.environ.get("LOCAL_RANK", "0")) if DDP else 0
    if DDP:
        dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl" if torch.cuda.is_available() else "gloo"))
        if torch.cuda.is_available(): torch.cuda.set_device(LOCAL)
    dev = ("cuda:%d" % LOCAL) if torch.cuda.is_available() else "cpu"
    COMPILE = int(os.environ.get("COMPILE", "0"))
    os.makedirs(OUT, exist_ok=True)
    LOG = os.path.join(OUT, TAG + ".log")


    def log(msg):
        if PRANK != 0: return
        print(msg, flush=True)
        with open(LOG, "a") as f: f.write(msg + "\n")


    class Stream:

        def __init__(self, path, L, dt=np.uint16):
            self.t = np.memmap(path, dtype=dt, mode="r")
            self.L, self.n = L, len(self.t) // L

        def take(self, idx):
            out = np.empty((len(idx), self.L), dtype=np.int64)
            for k, i in enumerate(idx): out[k] = self.t[i * self.L:(i + 1) * self.L]
            return torch.from_numpy(out)


    def synth(path, n_tok, vocab, seed):
        rng = np.random.default_rng(seed)
        dt = np.uint16 if vocab <= 65535 else np.uint32
        a = np.memmap(path, dtype=dt, mode="w+", shape=(n_tok,))
        a[:] = rng.integers(0, vocab, size=n_tok, dtype=dt); a.flush()


    if SELFTEST:
        DATA = os.path.join(OUT, "_selftest_data"); os.makedirs(DATA, exist_ok=True)
        VOCAB, DX, Dop, RANK, S, L, BS = 512, 128, 256, 8, 2, 64, 4
        TOKENS, WARM, VEVERY = 4e4, 5, 20
        for nm, n in (("train_code", 60000), ("train_noncode", 140000), ("val_code", 8000), ("val_noncode", 8000)):
            p = os.path.join(DATA, nm + ".bin")
            if not os.path.exists(p): synth(p, n, VOCAB, hash(nm) % 10**6)
        json.dump(dict(vocab=VOCAB, eos_id=0, pad_id=1, dtype="uint16"), open(os.path.join(DATA, "meta.json"), "w"))

    meta = json.load(open(os.path.join(DATA, "meta.json")))
    assert VOCAB >= int(meta["vocab"]), "VOCAB %d < vocabolario del corpus %s" % (VOCAB, meta["vocab"])
    DT = np.dtype(meta.get("dtype", "uint16"))
    tr = {k: Stream(os.path.join(DATA, "train_%s.bin" % k), L, DT) for k in ("code", "noncode")}
    if LEXT and not BSEXT: BSEXT = max(1, (BS * (L - 1)) // (LEXT - 1))
    tr_ext = {k: Stream(os.path.join(DATA, "train_%s.bin" % k), LEXT, DT) for k in ("code", "noncode")} if LEXT else None
    VL = int(meta.get("val_L", L))
    va = {k: Stream(os.path.join(DATA, "val_%s.bin" % k), VL, DT) for k in ("code", "noncode")}
    VDOM = None
    _vd = os.path.join(DATA, "val_dom.json")
    if os.path.exists(_vd): VDOM = json.load(open(_vd))

    log("corpus %s | train code %.3fM finestre (%.2fB token) noncode %.3fM (%.2fB) | val %d/%d finestre | L=%d" % (
        DATA, tr["code"].n / 1e6, tr["code"].n * L / 1e9, tr["noncode"].n / 1e6, tr["noncode"].n * L / 1e9,
        va["code"].n, va["noncode"].n, L))

    MIX = {ph: dict(perm={k: None for k in tr}, pos={k: 0 for k in tr}, pss={k: 0 for k in tr}, acc=0.0) for ph in ("main", "ext")}
    PHASE = "main"


    def _next(k, n):
        st = MIX[PHASE]; src = tr if PHASE == "main" else tr_ext; base = SEED if PHASE == "main" else SEED + 99991
        out, got = [], 0
        while got < n:
            if st["perm"][k] is None or st["pos"][k] >= len(st["perm"][k]):
                st["perm"][k] = np.random.default_rng(base + 7717 * st["pss"][k] + (0 if k == "code" else 1)).permutation(src[k].n)[PRANK::WORLD][: src[k].n // WORLD]
                st["pos"][k] = 0; st["pss"][k] += 1
            take = min(n - got, len(st["perm"][k]) - st["pos"][k])
            out.append(st["perm"][k][st["pos"][k]:st["pos"][k] + take]); st["pos"][k] += take; got += take
        return np.concatenate(out)


    def next_batch():
        st = MIX[PHASE]; src = tr if PHASE == "main" else tr_ext; nb = BS if PHASE == "main" else BSEXT
        nc = 0
        for _ in range(nb):
            st["acc"] += CODE_FRAC
            if st["acc"] >= 1.0: nc += 1; st["acc"] -= 1.0
        ids = []
        if nc: ids.append(src["code"].take(_next("code", nc)))
        if nb - nc: ids.append(src["noncode"].take(_next("noncode", nb - nc)))
        return torch.cat(ids, 0).to(dev), nc


    model = TriplexCascadeNTP(vocab=VOCAB, d_x=DX, S=S, d_g=DG, n_head=NH, d_head=DG // NH, D=Dop,
                              n=NOPS, rank=RANK, cat=FAMILY_SEQ, init_seed=SEED, shared_proj=bool(SHARED_PROJ), proj_init=PROJ_INIT).to(dev)
    pb = model.param_breakdown()
    if not SHARED_PROJ: print("PROIETTORI NON CONDIVISI: un terzetto per stadio (init %s)" % PROJ_INIT, flush=True)
    if COMPILE:
        for _b in model.blocks: _b.gru.step = torch.compile(_b.gru.step, dynamic=False)
    net = model
    if DDP:
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL] if torch.cuda.is_available() else None,
                                                        find_unused_parameters=True)
    log("CASC-NTP %s | FAMILY_SEQ=%s (K=%d) | vocab=%d dx=%d D=%d r=%d blocchi=%d dg=%d(x%d teste) L=%d BS=%d seed=%d" % (
        TAG, FAMILY_SEQ, len(FAMILY_SEQ), VOCAB, DX, Dop, RANK, S, DG, NH, L, BS, SEED))
    log("PARAMETRI | " + " ".join("%s %.3fM" % (k, v / 1e6) for k, v in pb.items()) +
        " | token/parametro a %.2fB: %.1f" % (TOKENS / 1e9, TOKENS / pb["total_trainable"]))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    GB = BS * ACC * WORLD
    TOT = int(TOKENS / (GB * (L - 1)))
    if VTOK > 0: VEVERY = max(1, int(round(VTOK / (GB * (L - 1)))))
    _dec0 = int(TOT * (1.0 - DECAY))


    def _lr(s):
        if s < WARM: return (s + 1) / WARM
        if s < _dec0: return 1.0
        return max(0.0, (TOT - s) / max(1, TOT - _dec0))


    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    _wc = TOT * BS * CODE_FRAC; _wn = TOT * BS * (1 - CODE_FRAC)
    if _wc > tr["code"].n * 1.001 or _wn > tr["noncode"].n * 1.001:
        log("ATTENZIONE: il budget richiede %.0f finestre code e %.0f noncode, il corpus ne ha %d e %d: "
            "il flusso in difetto verra' RIPETUTO (piu' di una passata)." % (_wc, _wn, tr["code"].n, tr["noncode"].n))
    log("PROTOCOLLO NTP: %.2fB token di predizione = %d step x (%d ranghi x %d acc x %d) finestre x %d posizioni | batch "
        "globale %d finestre = %d token | WSD warm %d decay %.2f | mix code %.0f%% deterministico | validazione ogni %d step = "
        "%.1fM token | compile=%d" % (TOKENS / 1e9, TOT, WORLD, ACC, BS, L - 1, GB, GB * (L - 1), WARM, DECAY, 100 * CODE_FRAC,
                                      VEVERY, VEVERY * GB * (L - 1) / 1e6, COMPILE))
    if LEXT:
        log("ESTENSIONE FINALE: ultimi %.0fM token di predizione a finestre L=%d, batch %d/rango (%d token/passo/rango, "
            "contro %d in fase principale); validazione invariata a L=%d" % (TEXT / 1e6, LEXT, BSEXT, BSEXT * (LEXT - 1), BS * (L - 1), VL))

    STATE = os.path.join(OUT, TAG + "_state.pt")
    step0, best, seen = 0, 1e9, 0
    if RESUME and os.path.exists(STATE):
        ck = torch.load(STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step0, best, seen = ck["step"], ck["best"], ck["seen"]
        for ph in ("main", "ext"):
            stt = ck["mix"][ph]; MIX[ph]["pos"], MIX[ph]["pss"], MIX[ph]["acc"] = stt["pos"], stt["pss"], stt["acc"]
            src = tr if ph == "main" else tr_ext; base = SEED if ph == "main" else SEED + 99991
            MIX[ph]["perm"] = {k: (np.random.default_rng(base + 7717 * (stt["pss"][k] - 1) + (0 if k == "code" else 1)).permutation(src[k].n)[PRANK::WORLD][: src[k].n // WORLD]
                                   if (stt["pss"][k] > 0 and src is not None) else None) for k in tr}
        PHASE = ck.get("phase", "main")
        log("RESUME da %s: step %d, %.3fB token visti, best %.4f" % (STATE, step0, seen / 1e9, best))


    @torch.no_grad()
    def validate():
        model.eval()
        dom_ce, lang_ce, out = {}, {}, {}
        for k, st in va.items():
            tot = n = 0.0
            for i0 in range(0, st.n, BS):
                idx = range(i0, min(i0 + BS, st.n))
                ids = st.take(idx).to(dev)
                z = model(ids)[0]
                ce = F.cross_entropy(z[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1), reduction="none")
                ce = ce.view(ids.shape[0], -1).sum(-1)
                tot += float(ce.sum()); n += ids.shape[0] * (VL - 1)
                for j, w in enumerate(idx):
                    lab = VDOM[k][w] if VDOM and k in VDOM and w < len(VDOM[k]) else k
                    top = lab.split("|")[0]
                    a = dom_ce.setdefault(top, [0.0, 0.0]); a[0] += float(ce[j]); a[1] += VL - 1
                    if "|" in lab:
                        b = lang_ce.setdefault(lab.split("|")[1], [0.0, 0.0]); b[0] += float(ce[j]); b[1] += VL - 1
            out[k] = tot / max(n, 1)
        nc, nn_ = va["code"].n, va["noncode"].n
        out["all"] = (out["code"] * nc + out["noncode"] * nn_) / max(nc + nn_, 1)
        out["per_dom"] = {d: v[0] / v[1] for d, v in dom_ce.items()}
        out["per_lang"] = {d: v[0] / v[1] for d, v in lang_ce.items()}
        out["macro"] = sum(out["per_dom"].values()) / max(len(out["per_dom"]), 1)
        model.train(); return out


    def save_state(step, best, seen):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        best=best, seen=seen, phase=PHASE,
                        mix={ph: dict(pos=MIX[ph]["pos"], pss=MIX[ph]["pss"], acc=MIX[ph]["acc"]) for ph in MIX}), STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)


    model.train(); t0 = time.time(); run_ce = 0.0; nrun = 0; run_gn = 0.0; n_clip = 0
    for step in range(step0, TOT):
        if LEXT and PHASE == "main" and seen >= TOKENS - TEXT:
            PHASE = "ext"; log("--> FASE DI ESTENSIONE a L=%d da step %d (%.3fB token visti)" % (LEXT, step, seen / 1e9))
            if PRANK == 0: torch.save(model.state_dict(), os.path.join(OUT, TAG + "_pre_ext.pt"))
        acc_ce = 0.0
        for _a in range(ACC):
            ids, nc = next_batch()
            z = net(ids, False)[0]
            ce = F.cross_entropy(z[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
            if int(os.environ.get("FORCE_NAN_AT", "-1")) == step: ce = ce * float("nan")
            if not torch.isfinite(ce):
                bad = torch.tensor([1.0], device=ce.device)
                if DDP: dist.all_reduce(bad)
                if PRANK == 0:
                    open(os.path.join(OUT, TAG + ".STOP"), "w").write("perdita non finita a step %d, seen %d\n" % (step, seen))
                    log("STOP: perdita non finita a step %d (%.3fB token). Stato NON salvato; ultimo stato buono in %s"
                        % (step, seen / 1e9, STATE))
                if DDP: dist.barrier()
                sys.exit(3)
            (ce / ACC).backward()
            acc_ce += float(ce.detach()) / ACC
            seen += ids.shape[0] * (ids.shape[1] - 1) * WORLD
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_ce += acc_ce; nrun += 1; run_gn += gn; n_clip += int(gn > CLIP)

        if (step + 1) % VEVERY == 0 or step + 1 == TOT:
            if DDP: dist.barrier()
            if PRANK != 0: continue
            v = validate()
            with torch.no_grad():
                st = model(ids[:2], stats=True)[1]
            line = ("CASC-NTP %s %.3fBtok (step %d/%d)%s | %.0fs | CE tr %.4f | MACRO %.4f (best %.4f) | "
                    "CE val %.4f (code %.4f noncode %.4f) | PPL %.1f | lr %.2e | gnorm %.3f clip %.0f%%"
                    % (TAG, seen / 1e9, step + 1, TOT, (" [L=%d]" % LEXT) if PHASE == "ext" else "", time.time() - t0, run_ce / max(nrun, 1), v["macro"],
                       min(best, v["macro"]), v["all"], v["code"], v["noncode"], math.exp(min(v["macro"], 20)),
                       sched.get_last_lr()[0], run_gn / max(nrun, 1), 100.0 * n_clip / max(nrun, 1)))
            log(line); run_ce = 0.0; nrun = 0; run_gn = 0.0; n_clip = 0
            log("   domini: " + " ".join("%s %.4f" % (k, x) for k, x in sorted(v["per_dom"].items())))
            if v["per_lang"]:
                log("   codice : " + " ".join("%s %.4f" % (k, x) for k, x in sorted(v["per_lang"].items())))
            for b, sb in enumerate(st):
                log("   b%d " % b + " | ".join(
                    "%s id %.2f eH %.2f du/u %.3f dx/x %.4f cum %.3f cosu %s r90 %d/%d%s" % (
                        f["fam"], f["id"], f["eH"], f["du_rel"], f["dx_rel_prev"], f["cum_rel"],
                        ("%.4f" % f["cos_u"]) if "cos_u" in f else "  -   ", f["r90u"], f["r90x"],
                        (" rmsAB %.4f" % f["rmsAB"]) if "rmsAB" in f else "") for f in sb["fam"]))
            if v["macro"] < best:
                best = v["macro"]; torch.save(model.state_dict(), os.path.join(OUT, TAG + "_best.pt"))
            save_state(step + 1, best, seen)

    if PRANK == 0: torch.save(model.state_dict(), os.path.join(OUT, TAG + "_final.pt"))
    log("FINE %s | %.3fB token | best MACRO %.4f (PPL %.1f) | salvati %s_best.pt (MACRO L=%d) e %s_final.pt (ultimo step%s)"
        % (TAG, seen / 1e9, best, math.exp(min(best, 20)), TAG, VL, TAG, (", post-estensione L=%d" % LEXT) if LEXT else ""))
