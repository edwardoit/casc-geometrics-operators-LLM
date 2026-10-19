"""CASC inference on a trained checkpoint: passive measurements and sampling.

  MODE=extrap   per-position CE on held-out windows of LW tokens: full context versus a sliding
                96-token window (always in distribution), reported in bins of 32 positions.
  MODE=stream   incremental inference: both attentions see a 96-token window while the GRU memory runs
                continuously over the whole sequence (or is reset per window, as a control).
  MODE=sample   autoregressive generation, full context versus sliding 96-token window.

Held-out documents are read from TXT/*_validation/*.jsonl (fields "text" and "domain").
Environment: CK (checkpoint), TOK (tok64k.json), TXT, MODE, LW, NPD, BS, SEED;
sampling: TEMP, TOPP, REP, NEW, PROMPT_TOK, NOBAN.
The sampling defaults add a repetition penalty (REP=1.15) and a repeated-trigram ban. The published
NTP_A_4B samples use pure nucleus sampling instead: NOBAN=1 REP=1.0.

Usage:  CK=NTP_A_4B_final.pt TOK=tok64k.json TXT=heldout MODE=extrap python generate.py
"""
import collections, glob, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from casc import TriplexCascadeNTP, rope_cache


def long_windows():
    out = collections.defaultdict(list)
    for fp in sorted(glob.glob(f"{TXT}/*_validation/*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line); d = r["domain"].split("|")[0]
            if len(out[d]) >= NPD: continue
            ids = tok.encode(r["text"]).ids
            if len(ids) >= LW: out[d].append(ids[:LW])
    return out


@torch.no_grad()
def ce_positions(X):
    out = []
    for i in range(0, len(X), BS):
        ids = X[i:i + BS].to(dev); z = m(ids)[0]
        out.append(F.cross_entropy(z[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1).cpu())
    return torch.cat(out)


@torch.no_grad()
def stream_ce(X, W=96, CH=6, reset_gru=False):
    Lx = X.shape[1]; nW = Lx - W
    idx = torch.arange(W, Lx, device=dev)[:, None] + torch.arange(-W + 1, 1, device=dev)[None, :]
    cos, sin = rope_cache(W, m.d_head, dev, torch.float32)
    outs = []
    for i in range(0, len(X), CH):
        ids = X[i:i + CH].to(dev); B = ids.shape[0]
        x = m.tok(ids).float()
        for blk in m.blocks:
            e = blk.P_G(blk.norm_g(x)); er, ez, eh = blk.gru.proj(e); Urz = torch.cat([blk.gru.Ur, blk.gru.Uz], 1)
            if reset_gru:
                ew = [t[:, idx].reshape(B * nW, W, -1) for t in (er, ez, eh)]
                gb = x.new_zeros(B * nW, blk.gru.d_g); G = []
                for t in range(W):
                    _, gb, _ = blk.gru.step(ew[0][:, t], ew[1][:, t], ew[2][:, t], gb, Urz); G.append(gb)
                Gw = torch.stack(G, 1)
                gb = x.new_zeros(B, blk.gru.d_g); G0 = []
                for t in range(W):
                    _, gb, _ = blk.gru.step(er[:, t], ez[:, t], eh[:, t], gb, Urz); G0.append(gb)
                G0 = torch.stack(G0, 1)
            else:
                gb = x.new_zeros(B, blk.gru.d_g); G = []
                for t in range(Lx):
                    _, gb, _ = blk.gru.step(er[:, t], ez[:, t], eh[:, t], gb, Urz); G.append(gb)
                Gb = torch.stack(G, 1); G0 = Gb[:, :W]; Gw = Gb[:, idx].reshape(B * nW, W, -1)
            if hasattr(blk, "P_mem"): G0, Gw = blk.P_mem(G0), blk.P_mem(Gw)
            cA0 = blk.att(G0, cos, sin, False)[0]
            cAw = blk.att(Gw, cos, sin, False)[0][:, -1].view(B, nW, -1)
            xk = x + blk.W_O(torch.cat([cA0, cAw], 1))
            for k, br in enumerate(blk.branches):
                h = blk.norm_u(xk)
                c0 = blk.ctl(h[:, :W], cos, sin, False)[0]
                cw = blk.ctl(h[:, idx].reshape(B * nW, W, -1), cos, sin, False)[0][:, -1].view(B, nW, -1)
                pi = torch.softmax(blk.heads[k](torch.cat([c0, cw], 1)), -1).reshape(B * Lx, -1)
                hf = h.reshape(B * Lx, -1); u = F.silu(blk.P_g(hf)) * blk.P_up(hf)
                d = br.deform(u, pi); psi = d if br.fam == "S" else F.silu(d)
                xk = xk + blk.P_dn(psi).view(B, Lx, -1)
            x = xk
        z = m.readout(x)
        outs.append(F.cross_entropy(z[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(B, -1).cpu())
    return torch.cat(outs)


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__); sys.exit(0)
    CK = os.environ.get("CK", "run_ntp/NTP_A_FBMMVMS_v4_best.pt"); MODE = os.environ.get("MODE", "extrap")
    TOK = os.environ.get("TOK", "ntp_v4/tok64k.json"); TXT = os.environ.get("TXT", "ntp_v4text")
    LW = int(os.environ.get("LW", "384")); NPD = int(os.environ.get("NPD", "60")); BS = int(os.environ.get("BS", "256"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(int(os.environ.get("SEED", "1")))
    tok = Tokenizer.from_file(TOK); V = tok.get_vocab_size(); EOS = tok.token_to_id("<|eos|>")
    m = TriplexCascadeNTP(vocab=64000, d_x=1024, S=4, D=4096, rank=32, n=5, cat="FBMMVMS")
    sd = torch.load(CK, map_location="cpu", weights_only=False); m.load_state_dict(sd.get("model", sd)); m = m.to(dev).eval()

    if MODE == "stream":
        W = long_windows(); doms = sorted(W); X = torch.tensor([w for d in doms for w in W[d]]); dom = np.array([d for d in doms for _ in W[d]])
        print("finestre da %d token: %d" % (LW, len(X)), flush=True)
        t0 = time.time(); full = ce_positions(X)
        cont = stream_ce(X, reset_gru=False); print("streaming GRU continua: %.0fs" % (time.time() - t0), flush=True)
        rst = stream_ce(X, reset_gru=True); print("streaming GRU azzerata: %.0fs" % (time.time() - t0), flush=True)
        print("\nCE per bin di posizione: contesto pieno | streaming GRU-continua | streaming GRU-azzerata")
        print("%-10s %10s %12s %12s %12s" % ("posizioni", "pieno", "GRU-cont", "GRU-reset", "cont-reset"))
        for a in range(0, LW - 1, 32):
            b = min(a + 32, LW - 1)
            print("%3d-%-6d %10.4f %12.4f %12.4f %+12.4f" % (a, b - 1, float(full[:, a:b].mean()), float(cont[:, a:b].mean()), float(rst[:, a:b].mean()), float(cont[:, a:b].mean() - rst[:, a:b].mean())))
        print("\nper dominio, posizioni 192-383: pieno | GRU-cont | GRU-reset")
        for d in doms:
            sd_ = torch.from_numpy(dom == d)
            print("  %-13s %8.4f | %8.4f | %8.4f" % (d, float(full[sd_, 192:].mean()), float(cont[sd_, 192:].mean()), float(rst[sd_, 192:].mean())))

    elif MODE == "extrap":
        W = long_windows(); doms = sorted(W); X = torch.tensor([w for d in doms for w in W[d]]); dom = np.array([d for d in doms for _ in W[d]])
        print("finestre da %d token: %d (%s)" % (LW, len(X), ", ".join("%s %d" % (d, len(W[d])) for d in doms)), flush=True)
        t0 = time.time()
        full = ce_positions(X)
        sl = torch.full_like(full, float("nan"))
        sl[:, :95] = full[:, :95]
        win = []
        for t in range(95, LW - 1):
            win.append(X[:, t - 94:t + 2])
        Wn = torch.stack(win, 1).reshape(-1, 96)
        cw = ce_positions(Wn)[:, -1].view(len(X), -1)
        sl[:, 95:] = cw
        print("CE per bin di posizione (media su tutte le finestre)   [%.0fs]" % (time.time() - t0))
        print("%-10s %10s %10s %10s" % ("posizioni", "pieno", "scorrevole", "delta"))
        for a in range(0, LW - 1, 32):
            b = min(a + 32, LW - 1)
            print("%3d-%-6d %10.4f %10.4f %+10.4f" % (a, b - 1, float(full[:, a:b].mean()), float(sl[:, a:b].nanmean()), float(full[:, a:b].mean() - sl[:, a:b].nanmean())))
        print("\nper dominio, media sulle posizioni 192-383: pieno | scorrevole")
        for d in doms:
            s = torch.from_numpy(dom == d)
            print("  %-13s %8.4f | %8.4f" % (d, float(full[s, 192:].mean()), float(sl[s, 192:].nanmean())))
        print("\nriferimento in distribuzione (posizioni 64-95, contesto pieno): %.4f" % float(full[:, 64:95].mean()))

    else:
        T = float(os.environ.get("TEMP", "0.8")); TOPP = float(os.environ.get("TOPP", "0.9")); REP = float(os.environ.get("REP", "1.15"))
        NEW = int(os.environ.get("NEW", "320")); PROMPT_TOK = int(os.environ.get("PROMPT_TOK", "32"))
        NOBAN = int(os.environ.get("NOBAN", "0"))

        def sample_next(logits, hist):
            lg = logits.clone()
            if REP != 1.0 and hist:
                h = torch.tensor(sorted(set(hist)), device=lg.device)
                lg[h] = torch.where(lg[h] > 0, lg[h] / REP, lg[h] * REP)
            if len(hist) >= 2 and not NOBAN:
                a, b = hist[-2], hist[-1]; ban = {hist[i + 2] for i in range(len(hist) - 2) if hist[i] == a and hist[i + 1] == b}
                if ban: lg[torch.tensor(sorted(ban), device=lg.device)] = -float("inf")
            p = torch.softmax(lg / T, -1); sp, si = torch.sort(p, descending=True)
            cum = torch.cumsum(sp, -1); keep = cum - sp < TOPP; sp = sp * keep; sp = sp / sp.sum()
            return int(si[torch.multinomial(sp, 1)])

        @torch.no_grad()
        def generate(prompt, window=None):
            hist = list(prompt)
            for _ in range(NEW):
                ctx = hist if window is None else hist[-window:]
                z = m(torch.tensor([ctx], device=dev))[0][0, -1]
                nxt = sample_next(z, hist); hist.append(nxt)
                if nxt == EOS: break
            return hist[len(prompt):]

        W = long_windows()
        for d in [x for x in ("dolma_wiki", "dolma_stack", "dolma_books") if x in W]:
            prompt = W[d][0][:PROMPT_TOK]
            print("\n" + "=" * 100 + "\nDOMINIO %s | prompt (%d token) | T=%.2f top-p=%.2f rep=%.2f noban=%d:\n%s" % (d, PROMPT_TOK, T, TOPP, REP, NOBAN, tok.decode(prompt)))
            for nome, win in (("CONTESTO PIENO (nessun limite)", None), ("FINESTRA SCORREVOLE 96", 96)):
                torch.manual_seed(7); g = generate(prompt, win)
                words = len(tok.decode(g).split())
                print("\n--- %s | %d token generati, ~%d parole ---\n%s" % (nome, len(g), words, tok.decode(g)))
