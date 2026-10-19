"""CASC data tools: build the 64k tokenizer, pack a corpus, check a packed corpus.

    python data.py tokenizer --code 'code/*.jsonl' --noncode 'web/*.jsonl' --out corpus/tok64k.json
    python data.py pack --tok corpus/tok64k.json --out corpus --code ... --noncode ... \
                        --val-code ... --val-noncode ... --code-tokens 1.2e9 --noncode-tokens 2.8e9
    python data.py check corpus [REF] [--tok PATH] [--frozen-tokenizer]

tokenizer  byte-level BPE, vocab 64000, special tokens <|bos|> <|eos|> <|pad|> = ids 0, 1, 2, no
           normalizer (no lowercasing, no NFKC); asserts a byte-exact round trip on whitespace,
           tabs, newlines and Unicode.
pack       two uint16 token streams (code / non-code) with the code share measured AFTER tokenization,
           EOS at the end of every document, held-out windows disjoint from train by document hash
           (identical train documents are dropped and counted as leaks), exact duplicates removed,
           documents shorter than 32 tokens skipped; writes meta.json and val_dom.json.
check      meta, id range, EOS per document, code share, leaks, decoded random windows, and with REF
           a byte-identical validation set (comparable MACRO). The md5 of the tokenizer used by the
           published checkpoint is reported as a reference; it is required only with --frozen-tokenizer.

Input shards are JSONL with a "text" field (optionally "domain"), or plain text with --raw.
Nothing is tokenized or written as a side effect of importing this module.
"""
import argparse, glob, hashlib, json, os, random, sys, time
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


VOCAB = 64000
SPECIAL = ["<|bos|>", "<|eos|>", "<|pad|>"]
SAMPLE_BYTES = 2_000_000_000


def texts(paths, budget, raw, seed):
    rnd = random.Random(seed); files = sorted(sum((glob.glob(p) for p in paths), []))
    rnd.shuffle(files); got = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            if raw:
                t = f.read(); got += len(t.encode()); yield t
            else:
                for line in f:
                    try: t = json.loads(line)["text"]
                    except Exception: continue
                    got += len(t.encode()); yield t
                    if got >= budget: return
        if got >= budget: return


def tokenizer_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", nargs="+", required=True)
    ap.add_argument("--noncode", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-frac", type=float, default=0.30)
    ap.add_argument("--sample-bytes", type=int, default=SAMPLE_BYTES)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIAL, show_progress=True,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)

    nb_c = int(a.sample_bytes * a.code_frac); nb_n = a.sample_bytes - nb_c
    print("campione BPE: %.2fGB code + %.2fGB noncode" % (nb_c / 1e9, nb_n / 1e9), flush=True)

    def stream():
        yield from texts(a.code, nb_c, a.raw, a.seed)
        yield from texts(a.noncode, nb_n, a.raw, a.seed + 1)

    tok.train_from_iterator(stream(), trainer=tr)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok.save(a.out)
    v = tok.get_vocab_size()
    print("salvato %s | vocab %d | special %s" % (a.out, v, {s: tok.token_to_id(s) for s in SPECIAL}))
    probe = "def f(x):\n\treturn x**2  # commento\n\nCiao è 'ok' — naïve 日本語\r\n"
    ids = tok.encode(probe).ids
    back = tok.decode(ids)
    assert back == probe, "round-trip NON esatto:\n%r\n%r" % (probe, back)
    assert v <= 65535, "vocab > 65535: i token non entrano in uint16"
    print("round-trip byte-esatto su whitespace/tab/newline/Unicode: OK | %d token per %d byte"
          % (len(ids), len(probe.encode())))


MIN_TOKENS = 32
def tok_dtype(V):
    return np.uint16 if V <= 65535 else np.uint32
BATCH = 1000


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "big")


def docs(paths, raw):
    for fp in sorted(sum((glob.glob(p) for p in paths), [])):
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            if raw:
                t = f.read()
                if t.strip(): yield t, os.path.basename(fp)
            else:
                for line in f:
                    try: r = json.loads(line)
                    except Exception: continue
                    t = r.get("text")
                    if t and t.strip(): yield t, str(r.get("domain", os.path.basename(fp).split(".")[0]))


def pack_val(stream_name, paths, tok, eos, out, raw, L, per_dom, maxw, dt=np.uint16):
    win, dom, got, hashes = [], [], {}, set()
    for text, lab in docs(paths, raw):
        top = lab.split("|")[0]
        hashes.add(h(text))
        if got.get(top, 0) >= per_dom: continue
        ids = tok.encode(text).ids + [eos]
        for w in range(maxw):
            a = w * L
            if a + L > len(ids) or got.get(top, 0) >= per_dom: break
            win.append(np.asarray(ids[a:a + L], dtype=dt)); dom.append(lab)
            got[top] = got.get(top, 0) + 1
    if win:
        np.concatenate(win).tofile(os.path.join(out, "val_%s.bin" % stream_name))
    print("  val %s: %d finestre da %d token | %s" % (stream_name, len(win), L,
          " ".join("%s %d" % kv for kv in sorted(got.items()))), flush=True)
    return len(win) * L, dom, hashes


def pack(stream_name, paths, target, tok, eos, out, raw, dt=np.uint16, val_hashes=None):
    tr_p = os.path.join(out, "train_%s.bin" % stream_name)
    tr = open(tr_p, "wb")
    n_tr = n_doc = n_skip = n_dup = n_leak = 0; seen = set(); t0 = time.time()
    val_hashes = val_hashes or set()
    buf, meta = [], []

    def flush():
        nonlocal n_tr, n_doc, n_skip
        if not buf: return
        for enc in tok.encode_batch(buf):
            ids = enc.ids
            if len(ids) < MIN_TOKENS: n_skip += 1; continue
            tr.write(np.asarray(ids + [eos], dtype=dt).tobytes())
            n_tr += len(ids) + 1; n_doc += 1
        buf.clear(); meta.clear()

    for text, _ in docs(paths, raw):
        hh = h(text)
        if hh in val_hashes: n_leak += 1; continue
        if hh in seen: n_dup += 1; continue
        seen.add(hh)
        buf.append(text)
        if len(buf) >= BATCH:
            flush()
            if n_tr >= target: break
            if n_doc % 100000 < BATCH:
                print("  %s %.3fB/%.3fB token  %d doc  %.0fs" % (stream_name, n_tr / 1e9, target / 1e9, n_doc,
                                                                 time.time() - t0), flush=True)
    flush(); tr.close()
    print("%s: train %.4fB token, %d doc, scartati %d corti / %d duplicati / %d FUGHE verso la validazione"
          % (stream_name, n_tr / 1e9, n_doc, n_skip, n_dup, n_leak), flush=True)
    return n_tr, n_doc, n_leak


def pack_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--code", nargs="+", required=True); ap.add_argument("--noncode", nargs="+", required=True)
    ap.add_argument("--code-tokens", type=float, default=1.2e9)
    ap.add_argument("--noncode-tokens", type=float, default=2.8e9)
    ap.add_argument("--val-L", type=int, default=512)
    ap.add_argument("--val-per-domain", type=int, default=800)
    ap.add_argument("--val-maxw", type=int, default=4)
    ap.add_argument("--val-code", nargs="+", default=None)
    ap.add_argument("--val-noncode", nargs="+", default=None)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(a.tok); V = tok.get_vocab_size()

    eos = next((i for i in (tok.token_to_id(t) for t in ("<|eos|>", "<|endoftext|>", "<|end_of_text|>")) if i is not None), None)
    pad = next((i for i in (tok.token_to_id(t) for t in ("<|pad|>", "<|endoftext|>")) if i is not None), None)
    assert eos is not None, "nessun separatore di documento nel tokenizer"
    dt = tok_dtype(V)
    print("tokenizer %s | vocab %d | separatore doc %d | pad %d | dtype %s" % (a.tok, V, eos, pad, np.dtype(dt).name), flush=True)

    res, vdom, nval = {}, {}, {}
    for nm, paths, tgt, vp in (("code", a.code, a.code_tokens, a.val_code),
                               ("noncode", a.noncode, a.noncode_tokens, a.val_noncode)):
        assert vp, "servono --val-code e --val-noncode: la validazione resta il multidominio"
        nval[nm], vdom[nm], vh = pack_val(nm, vp, tok, eos, a.out, a.raw, a.val_L, a.val_per_domain, a.val_maxw, dt)
        res[nm] = pack(nm, paths, tgt, tok, eos, a.out, a.raw, dt, val_hashes=vh)

    ntr_c, ntr_n = res["code"][0], res["noncode"][0]
    frac = ntr_c / max(ntr_c + ntr_n, 1)
    json.dump(dict(vocab=V, eos_id=eos, pad_id=pad, dtype=np.dtype(dt).name, tokenizer=str(a.tok),
                   val_L=a.val_L, val_per_domain=a.val_per_domain, val_maxw=a.val_maxw,
                   tokens=dict(train_code=ntr_c, train_noncode=ntr_n, total=ntr_c + ntr_n,
                               val_code=nval["code"], val_noncode=nval["noncode"]),
                   docs=dict(code=res["code"][1], noncode=res["noncode"][1]),
                   leaks_vs_validation=dict(code=res["code"][2], noncode=res["noncode"][2]),
                   code_frac=frac, target_code_frac=a.code_tokens / (a.code_tokens + a.noncode_tokens)),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    json.dump(vdom, open(os.path.join(a.out, "val_dom.json"), "w"))
    tgt = a.code_tokens / (a.code_tokens + a.noncode_tokens)
    print("TOTALE %.4fB token | quota code IN TOKEN %.4f (bersaglio %.4f)" % ((ntr_c + ntr_n) / 1e9, frac, tgt))
    if abs(frac - tgt) > 0.005:
        print("ATTENZIONE: quota fuori tolleranza: uno dei due flussi si e' esaurito prima del bersaglio.\n"
              "  code    %.4fB / %.4fB richiesti\n  noncode %.4fB / %.4fB richiesti\n"
              "  aggiungere shard al flusso in difetto, oppure ripiegare i bersagli sulla quota ottenibile."
              % (ntr_c / 1e9, a.code_tokens / 1e9, ntr_n / 1e9, a.noncode_tokens / 1e9), file=sys.stderr)
        sys.exit(2)


TOK_MD5 = "0f773030a2ff4e2ba08cf0b8a317ccd0"


def check_main():
    ap = argparse.ArgumentParser(description="sanity check of a packed corpus before a long run")
    ap.add_argument("DATA"); ap.add_argument("REF", nargs="?", default=None, help="corpus whose validation must be byte-identical")
    ap.add_argument("--tok", default=None, help="tokenizer file (default: the 'tokenizer' path recorded in meta.json)")
    ap.add_argument("--frozen-tokenizer", action="store_true", help="fail unless the tokenizer is the reference tok64k.json")
    a = ap.parse_args()
    DATA, REF = a.DATA, a.REF
    m = json.load(open(os.path.join(DATA, "meta.json"))); ok = True
    def chk(c, msg):
        nonlocal ok; ok &= bool(c); print(("  OK   " if c else "  FALLITO ") + msg)
    chk(int(m["vocab"]) == 64000, "vocab 64000 (%s)" % m["vocab"])
    chk(m.get("dtype", "uint16") == "uint16", "dtype uint16 (%s)" % m.get("dtype"))
    chk(m.get("tokenizer", "tok64k.json").endswith("tok64k.json"), "tokenizer tok64k.json (%s)" % m.get("tokenizer"))
    tokf = a.tok or (m.get("tokenizer") if os.path.exists(m.get("tokenizer", "")) else "ntp_v4/tok64k.json")
    same = hashlib.md5(open(tokf, "rb").read()).hexdigest() == TOK_MD5
    if a.frozen_tokenizer:
        chk(same, "tokenizer identico a quello congelato (md5 %s...)" % TOK_MD5[:8])
    else:
        print("  INFO tokenizer %s il tok64k.json di riferimento (md5 %s...)" % ("identico a" if same else "diverso da", TOK_MD5[:8]))
    chk(int(m["eos_id"]) == 1 and int(m["pad_id"]) == 2, "EOS=1 PAD=2 (%s, %s)" % (m["eos_id"], m["pad_id"]))
    if REF is None or os.path.abspath(REF) != os.path.abspath(DATA):
        chk(m.get("leaks_vs_validation") is not None, "fughe train->validazione conteggiate (packer aggiornato)")
    tt = m["tokens"]; print("  token impacchettati: code %.4fB, noncode %.4fB, totale %.4fB" % (tt["train_code"] / 1e9, tt["train_noncode"] / 1e9, tt["total"] / 1e9))
    if REF:
        for f in ("val_code.bin", "val_noncode.bin", "val_dom.json"):
            a = hashlib.md5(open(os.path.join(DATA, f), "rb").read()).hexdigest(); b = hashlib.md5(open(os.path.join(REF, f), "rb").read()).hexdigest()
            chk(a == b, "validazione %s identica a %s" % (f, REF))
    tok = Tokenizer.from_file(tokf)
    eos = int(m["eos_id"]); rng = np.random.default_rng(0)
    for k in ("code", "noncode"):
        t = np.memmap(os.path.join(DATA, "train_%s.bin" % k), dtype=np.uint16, mode="r")
        n = len(t); ne = int((t == eos).sum()); mx = int(t.max())
        chk(mx < 64000, "%s: id massimo %d < 64000" % (k, mx))
        chk(abs(ne - m["docs"][k]) <= 1, "%s: EOS %d ~ documenti %d" % (k, ne, m["docs"][k]))
        chk(n == m["tokens"]["train_" + k], "%s: %d token = meta" % (k, n))
        for i in rng.integers(0, n - 96, 2):
            print("      [%s @%d] %r" % (k, i, tok.decode(t[i:i + 96].astype(int).tolist())[:160]))
    cf = m["code_frac"]; chk(0 < cf < 1, "quota codice in token %.4f (bersaglio %.4f)" % (cf, m.get("target_code_frac", 0.3)))
    lk = m.get("leaks_vs_validation"); print("  fughe train->validazione scartate:", lk)
    print("\nCORPUS %s: %s" % (DATA, "PRONTO" if ok else "NON PRONTO"))
    sys.exit(0 if ok else 1)


COMMANDS = {"tokenizer": tokenizer_main, "pack": pack_main, "check": check_main}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__); sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv.pop(1); sys.argv[0] = "data.py " + cmd
    COMMANDS[cmd]()
