"""Benchmark scores computed on encrypted labels: R2, capped NLL and MAE.

Use case: the score is computed in the same process as the submission, and
the hidden labels must not be readable there. The organizers encrypt label
features offline (CKKS, OpenFHE), the evaluator computes the score on
ciphertexts from clear predictions, and only the secret key holder decrypts
it. This hides the labels. Against a forged score, the output stays one
loss per value, and a secret audit mask under its own key opens the
predictions at a few slots, where the key holder recomputes the loss. See
``tools/encrypted_scoring.md`` for which metrics fit which route, the audit,
and measured costs.

Three routes, from the cheapest to the most costly:

- R2, linear: SSE = sum(p^2 - 2 p y) + sum(y^2). The evaluator multiplies
  Enc(y) by clear p, the key holder adds sum(y^2) and divides by the SST.
- Negative-Binomial NLL, low rank: the only term coupling the label y and the
  predicted dispersion r is g(y, r) = gammaln(y + r) - gammaln(r), which is
  close to sum_k a_k(r) b_k(y) (SVD on public grids, Nystrom extension). The
  organizers encrypt b_k(y), the evaluator computes a_k(r) in the clear.
  Without a cap, the NLL stays on this route.
- MAE and the NLL cap, comparison: |p - y| = (p - y)(2c - 1) with c = [y < p].
  The organizers encrypt the L bits of the label, and c is a comparison tree
  of depth log2(L) against the clear bits of p. The NB pmf is unimodal, so
  min(nll, cap) only needs y in [lo, hi], two comparisons with bounds that the
  evaluator computes from the prediction.

The script prints, for each score, its error on all values (exact numpy),
then the single-core cost on one ciphertext and the HE error. When HEonGPU
finds a GPU, it runs the same circuits there too, with the same ring, depth
and modulus sizes. For R2 and the capped NLL, it then audits the honest
output and three forgeries (on the CPU only).

Requirements: ``pip install openfhe numpy scipy``, and ``heongpu`` for the
GPU (its wheels need a Turing or newer GPU and a 580+ driver). The openfhe
wheel needs Python 3.12 and glibc 2.39 (Ubuntu 24.04). Labels are a 1D
array. With 1D predictions, it benchmarks R2 and MAE. With (n, 2)
predictions of NB counts (mean, dispersion), as in the virtual bacteria
challenge, it adds the NLL and computes R2 and MAE on log1p. Without
arguments, it uses synthetic NB data::

    OMP_NUM_THREADS=1 python tools/encrypted_score_bench.py [y.npy p.npy]
"""
import argparse
import operator
import os
import tempfile
import time

import numpy as np
import openfhe as o
from scipy.special import gammaln

try:  # optional GPU backend, used when a GPU is visible
    from heongpu import ckks
    ckks.Session(ckks.Params(multiplicative_depth=1))
except (ImportError, RuntimeError):
    ckks = None

MAE_BITS = (12, 16, 20)
NLL_CAP = 100.0
R_RANGE = (1e-2, 1e3)  # public clip of the NB dispersion
RANK = 16
EPS = 1e-8
CHUNK = 4096
AUDIT_SLOTS = 1400  # catches 1% of faked values with probability 1 - 1e-6
SNAP = 2.0 ** -20  # public grid of the scored predictions
MC_DRAWS = 1000


class Backend:
    """What the circuits share across CKKS libraries."""

    def sync(self):
        pass

    def timed(self, fn, *args, reps=3):
        """Median wall time of fn(*args), and its output."""
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            out = fn(*args)
            self.sync()
            times.append(time.perf_counter() - start)
        return float(np.median(times)), out

    def lt(self, ct_bits, q):
        """Enc([x < q]) per slot, from Enc(bits of x) and clear integers q."""
        cc = self.cc
        theta = bits_of(q, len(ct_bits))
        # Per bit: lt = t (1 - b) and eq = [b == t], linear in b.
        lt = [cc.EvalAdd(cc.EvalMult(b, self.pt(-t)), self.pt(t))
              for b, t in zip(ct_bits, theta)]
        eq = [cc.EvalAdd(cc.EvalMult(b, self.pt(2 * t - 1)), self.pt(1 - t))
              for b, t in zip(ct_bits, theta)]
        # Merge (high, low) blocks: lt = lt_hi + eq_hi lt_lo, eq = eq_hi eq_lo.
        while len(lt) > 1:
            merged_lt, merged_eq = [], []
            for i in range(0, len(lt) - 1, 2):
                merged_lt.append(
                    cc.EvalAdd(lt[i], cc.EvalMult(eq[i], lt[i + 1])))
                merged_eq.append(cc.EvalMult(eq[i], eq[i + 1])
                                 if len(lt) > 2 else None)
            if len(lt) % 2:
                merged_lt.append(lt[-1])
                merged_eq.append(eq[-1])
            lt, eq = merged_lt, merged_eq
        return lt[0]


class HE(Backend):
    """OpenFHE CKKS context and keys at 128-bit security."""

    name = "CPU"

    def __init__(self, depth):
        params = o.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(depth)
        params.SetScalingModSize(50)
        params.SetFirstModSize(60)
        params.SetSecurityLevel(o.SecurityLevel.HEStd_128_classic)
        self.cc = o.GenCryptoContext(params)
        for feature in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH,
                        o.PKESchemeFeature.LEVELEDSHE):
            self.cc.Enable(feature)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.slots = self.cc.GetRingDimension() // 2

    def pt(self, x):
        return self.cc.MakeCKKSPackedPlaintext(np.asarray(x, float).tolist())

    def encrypt(self, *arrays):
        return [self.cc.Encrypt(self.keys.publicKey, self.pt(x))
                for x in arrays]

    def decrypt(self, ct, n):
        dec = self.cc.Decrypt(ct, self.keys.secretKey)
        dec.SetLength(n)
        return np.array(dec.GetRealPackedValue())

    def size(self, ct):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ct.bin")
            o.SerializeToFile(path, ct, o.BINARY)
            return os.path.getsize(path)


class GPUOps:
    """OpenFHE's evaluation calls, on HEonGPU ciphertexts."""

    EvalAdd = staticmethod(operator.add)
    EvalSub = staticmethod(operator.sub)
    EvalMult = staticmethod(operator.mul)
    EvalNegate = staticmethod(operator.neg)


class GPU(Backend):
    """HEonGPU session with the ring, depth and modulus sizes of OpenFHE.

    Products are relinearized and rescaled, and operands brought to the same
    level, automatically, as in OpenFHE.
    """

    name = "GPU"
    cc = GPUOps

    def __init__(self, depth, ring_dim):
        self.session = ckks.Session(ckks.Params(
            multiplicative_depth=depth, scale_bit_size=50,
            first_modulus_bit_size=60, poly_modulus_degree=ring_dim))
        self.slots = self.session.slot_count

    def pt(self, x):
        return np.asarray(x, float)

    def encrypt(self, *arrays):
        return [self.session.encrypt(self.pt(x)) for x in arrays]

    def decrypt(self, ct, n):
        return np.asarray(self.session.decrypt(ct))[:n]

    def size(self, ct):
        return len(ct.to_bytes())

    def sync(self):
        self.session.synchronize()


def backends(depth):
    """OpenFHE on the CPU, then HEonGPU with the same ring when available."""
    he = HE(depth)
    yield he
    if ckks is not None:
        yield GPU(depth, 2 * he.slots)


def bits_of(q, n_bits):
    """Bits of the integers q in [0, 2^n_bits), most significant first."""
    q = np.asarray(q).astype(np.int64)
    return [((q >> (n_bits - 1 - k)) & 1).astype(np.float64)
            for k in range(n_bits)]


def quantize(x, n_bits):
    """floor(x * 2^n_bits) for x in [0, 1)."""
    return np.minimum(np.floor(x * 2 ** n_bits), 2 ** n_bits - 1)


def g_exact(y, r):
    return gammaln(y + r) - gammaln(r)


def nb_nll(y, mu, r):
    """Uncapped NB NLL, as gammaln(y + 1) - g(y, r) plus terms linear in y."""
    return (gammaln(y + 1) - g_exact(y, r) + y * np.log1p(r / mu)
            + r * np.log1p(mu / r))


def first_y(lo, hi, mu, r, above):
    """Smallest integer y in [lo, hi) with (nll(y) > cap) == above, else hi."""
    while (lo < hi).any():
        active = lo < hi
        mid = np.floor((lo + hi) / 2)
        hit = (nb_nll(mid, mu, r) > NLL_CAP) == above
        hi = np.where(active & hit, mid, hi)
        lo = np.where(active & ~hit, mid + 1, lo)
    return lo


def nll_basis(y):
    """Public grids and the rank-RANK SVD of g on them."""
    grid_y = np.unique(np.r_[np.arange(64.0),
                             np.geomspace(64, max(y.max(), 64), 512).round()])
    grid_r = np.geomspace(*R_RANGE, 512)
    u, s, vt = np.linalg.svd(g_exact(grid_y[:, None], grid_r[None]),
                             full_matrices=False)
    return grid_y, grid_r, u[:, :RANK], s[:RANK], vt[:RANK]


def nll_label_features(y, grid_r, s, vt):
    """Organizers: the b_k(y), from the Nystrom extension on grid_r."""
    return np.concatenate([g_exact(y[i:i + CHUNK, None], grid_r[None])
                           @ vt.T / s for i in range(0, len(y), CHUNK)])


def nll_clear(mu, r, grid_y, u, n_bits):
    """Evaluator, in the clear: the a_k(r) and the cap interval [lo, hi]."""
    a = np.concatenate([g_exact(grid_y[None], r[i:i + CHUNK, None]) @ u
                        for i in range(0, len(r), CHUNK)])
    n_max = 2.0 ** n_bits - 2
    mode = np.minimum(np.where(r > 1, np.floor(mu * (r - 1) / r), 0), n_max)
    lo = first_y(np.zeros_like(mu), mode + 1, mu, r, above=False)
    hi = first_y(mode, np.full_like(mu, n_max + 1), mu, r, above=True) - 1
    return a, lo, hi


def nll_approx(y, b, a, mu, r, lo, hi):
    """Capped NLL per value, as the circuit computes it."""
    nll = (gammaln(y + 1) - (a * b).sum(1) + y * np.log1p(r / mu)
           + r * np.log1p(mu / r))
    return np.where((lo <= y) & (y <= hi), nll, NLL_CAP)


def report(name, he, n, t_enc, t_eval, cts, got, want):
    per_value = 1e6 / n
    kb = len(cts) * he.size(cts[0]) / he.slots / 1e3
    print(f"  {name}, {he.name}, N={2 * he.slots}: eval "
          f"{t_eval * per_value:.1f} us/value, encrypt "
          f"{t_enc * per_value:.1f} us/value, labels {kb:.2f} KB/value "
          f"({len(cts)} ct), HE error on the score "
          f"{abs(got - want) / abs(want):.1e}", flush=True)


def r2_eval(he, ct_y, p):
    """Enc(p^2 - 2 p y) per slot."""
    return he.cc.EvalAdd(he.cc.EvalMult(ct_y, he.pt(-2 * p)), he.pt(p ** 2))


def nll_uncapped_eval(he, ct_y, ct_gl1, ct_b, a, mu, r):
    """Enc(nll) per slot, linear in y, gammaln(y + 1) and the b_k(y)."""
    cc = he.cc
    nll = cc.EvalAdd(cc.EvalMult(ct_y, he.pt(np.log1p(r / mu))),
                     he.pt(r * np.log1p(mu / r)))
    nll = cc.EvalAdd(nll, ct_gl1)
    for k, ct in enumerate(ct_b):
        nll = cc.EvalAdd(nll, cc.EvalMult(ct, he.pt(-a[:, k])))
    return nll


def nll_eval(he, ct_bits, ct_gl1, ct_b, a, mu, r, lo, hi):
    """Enc(min(nll, cap)) per slot, with nll <= cap iff lo <= y <= hi."""
    cc, n_bits = he.cc, len(ct_bits)
    nll = cc.EvalAdd(ct_gl1, he.pt(r * np.log1p(mu / r) - NLL_CAP))
    for k, ct in enumerate(ct_b):
        nll = cc.EvalAdd(nll, cc.EvalMult(ct, he.pt(-a[:, k])))
    slope = np.log1p(r / mu)
    for k, ct in enumerate(ct_bits):  # y = sum_k 2^(L-1-k) bit_k
        nll = cc.EvalAdd(nll, cc.EvalMult(
            ct, he.pt(slope * 2.0 ** (n_bits - 1 - k))))
    inside = cc.EvalMult(cc.EvalAdd(cc.EvalNegate(he.lt(ct_bits, lo)), 1.0),
                         he.lt(ct_bits, hi + 1))
    return cc.EvalAdd(cc.EvalMult(inside, nll), NLL_CAP)


def mae_eval(he, ct_u, ct_bits, v):
    """Enc(|v - u|) per slot, as (v - u)(2 [u < v] - 1)."""
    cc = he.cc
    lt = he.lt(ct_bits, quantize(v, len(ct_bits)))
    sign = cc.EvalSub(cc.EvalAdd(lt, lt), 1.0)
    return cc.EvalMult(sign, cc.EvalAdd(cc.EvalNegate(ct_u), he.pt(v)))


def bench_r2(y, p):
    # Public standardization keeps the CKKS values small.
    p, y = (p - y.mean()) / y.std(), (y - y.mean()) / y.std()
    exact = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"R2 {exact:.6g}, linear route: exact up to HE noise")
    for he in backends(depth=1):
        ys, ps = y[:he.slots], p[:he.slots]
        t_enc, cts = he.timed(he.encrypt, ys, reps=1)
        t_eval, ct = he.timed(r2_eval, he, cts[0], ps)
        # The key holder adds sum(y^2) and divides by the SST.
        sst = ((ys - ys.mean()) ** 2).sum()
        got = 1 - (he.decrypt(ct, len(ys)).sum() + (ys ** 2).sum()) / sst
        want = 1 - ((ps - ys) ** 2).sum() / sst
        report("linear", he, len(ys), t_enc, t_eval, cts, got, want)


def bench_nll(y, mu, r):
    assert (y == np.round(y)).all() and y.min() >= 0, "the NLL needs counts"
    n_bits = int(y.max() + 1).bit_length()
    exact_nll = nb_nll(y, mu, r)
    exact = np.minimum(exact_nll, NLL_CAP).mean()

    grid_y, grid_r, u, s, vt = nll_basis(y)
    b = nll_label_features(y, grid_r, s, vt)
    start = time.perf_counter()
    a, lo, hi = nll_clear(mu, r, grid_y, u, n_bits)
    t_clear = time.perf_counter() - start

    g = (a * b).sum(1)
    approx_nll = exact_nll + g_exact(y, r) - g
    inside = (lo <= y) & (y <= hi)
    approx = nll_approx(y, b, a, mu, r, lo, hi)
    print(f"NB NLL capped at {NLL_CAP:g}: {exact:.6g}, rank {RANK} and cap "
          f"circuit (L={n_bits}): relative error "
          f"{abs(approx.mean() - exact) / exact:.1e}, uncapped "
          f"{abs(approx_nll.mean() - exact_nll.mean()) / exact_nll.mean():.1e}"
          f", max |g error| {np.abs(g - g_exact(y, r)).max():.1e} nats, cap "
          f"mismatches {(inside != (exact_nll <= NLL_CAP)).sum()}, clear "
          f"features {t_clear * 1e6 / len(y):.0f} us/value")
    print(f"  uncapped NLL {exact_nll.mean():.6g}, max per value "
          f"{exact_nll.max():.3g}")

    # Depth 2 leaves room for the large intermediate terms a_k b_k.
    for he in backends(depth=2):
        sl = slice(0, he.slots)
        t_enc, cts = he.timed(he.encrypt, y[sl], gammaln(y[sl] + 1),
                              *b[sl].T, reps=1)
        t_eval, ct = he.timed(nll_uncapped_eval, he, cts[0], cts[1], cts[2:],
                              a[sl], mu[sl], r[sl])
        n = len(y[sl])
        report("low rank, uncapped", he, n, t_enc, t_eval, cts,
               he.decrypt(ct, n).mean(), approx_nll[sl].mean())

    for he in backends(depth=int(np.ceil(np.log2(n_bits))) + 3):
        sl = slice(0, he.slots)
        t_enc, cts = he.timed(he.encrypt, *bits_of(y[sl], n_bits),
                              gammaln(y[sl] + 1), *b[sl].T, reps=1)
        t_eval, ct = he.timed(nll_eval, he, cts[:n_bits], cts[n_bits],
                              cts[n_bits + 1:], a[sl], mu[sl], r[sl], lo[sl],
                              hi[sl])
        n = len(y[sl])
        report("low rank and cap", he, n, t_enc, t_eval, cts,
               he.decrypt(ct, n).mean(), approx[sl].mean())


def bench_mae(y, p):
    # Public bounds of the labels; predictions are clipped to them.
    lo, width = y.min(), (y.max() - y.min()) * 1.001
    u = (y - lo) / width
    v = np.clip((p - lo) / width, 0, 1 - 1e-12)
    exact = np.abs(v - u).mean()
    print(f"MAE {width * exact:.6g}, comparison route")
    for n_bits in MAE_BITS:
        sign = 2 * (quantize(u, n_bits) < quantize(v, n_bits)) - 1
        approx = sign * (v - u)
        err = abs(approx.mean() - exact) / exact
        print(f"  L={n_bits}: relative error {err:.1e}")
        for he in backends(depth=int(np.ceil(np.log2(n_bits))) + 3):
            us, vs = u[:he.slots], v[:he.slots]
            t_enc, cts = he.timed(he.encrypt, us,
                                  *bits_of(quantize(us, n_bits), n_bits),
                                  reps=1)
            t_eval, ct = he.timed(mae_eval, he, cts[0], cts[1:], vs)
            report(f"L={n_bits}", he, len(us), t_enc, t_eval, cts,
                   he.decrypt(ct, len(us)).mean(), approx[:he.slots].mean())


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def snap(x):
    """Round to the public grid, so that a noisy opening rounds back."""
    return np.round(x / SNAP) * SNAP


class Audit:
    """Secret audit mask, under its own key: no evaluation key involves it.

    The evaluator can only multiply it by plaintexts. It cannot combine it
    with the score ciphertexts, so the slots it fakes do not depend on it.
    """

    def __init__(self, he, n, rng):
        self.keys = he.cc.KeyGen()
        self.idx = rng.choice(n, AUDIT_SLOTS, replace=False)
        self.alpha = rng.uniform(1, 2, AUDIT_SLOTS)
        mask = np.zeros(n)
        mask[self.idx] = self.alpha
        self.ct = he.cc.Encrypt(self.keys.publicKey, he.pt(mask))

    def open(self, he, ct):
        """Key holder: the opened prediction at each audited slot."""
        dec = he.cc.Decrypt(ct, self.keys.secretKey)
        dec.SetLength(self.idx.max() + 1)
        return snap(np.array(dec.GetRealPackedValue())[self.idx] / self.alpha)


def audit_eval(he, ct_mask, *preds):
    """Evaluator: Enc(m alpha p) for each prediction array p."""
    return [he.cc.EvalMult(ct_mask, he.pt(p)) for p in preds]


def audit_check(name, he, audit, ct_loss, offset, ref, ref_all, bounds, tol,
                rng, t_audit, cts_open):
    """Key holder's check on the honest output and on forged ones.

    The loss per value is the decrypted output plus a label-only offset. It
    must be in bounds everywhere, and match the loss recomputed from the
    openings, ref, within tol at the audited slots. ref_all is the honest
    loss at all slots, which only the benchmark knows: it gives the faked
    slots, to estimate the detection rate over random masks.
    """
    cc, n = he.cc, len(offset)
    print(f"{name}, audit of {AUDIT_SLOTS} values out of {n}")
    try:
        cc.EvalMult(ct_loss, audit.ct)
        print("  cross-key product accepted")
    except Exception as err:
        print(f"  cross-key product refused: {str(err).split(': ')[-1]}")

    start = time.perf_counter()
    loss = he.decrypt(ct_loss, n) + offset
    err = np.abs(loss - ref_all).max()
    for ct in cts_open:
        audit.open(he, ct)
    t_check = time.perf_counter() - start
    print(f"  audit eval {t_audit * 1e6 / n:.2f} us/value, output "
          f"{len(cts_open) * he.size(cts_open[0]) / n / 1e3:.2f} KB/value, "
          f"check {t_check * 1e6 / n:.2f} us/value, honest HE error "
          f"{err:.1e} per value at most (tolerance {tol:g})")

    # The forger guesses its mean loss, and claims half of it on faked slots.
    guess = ref_all.mean()
    forged = {"honest": ct_loss,
              "fresh encryption": cc.Encrypt(he.keys.publicKey,
                                             he.pt(np.full(n, guess / 2)))}
    for eps in (1e-2, 1e-3):
        faked = rng.random(n) < eps
        forged[f"{eps:.1%} of values faked"] = cc.EvalAdd(
            ct_loss, he.pt(-guess / 2 * faked))

    draws = np.argsort(rng.random((MC_DRAWS, n)), axis=1)[:, :AUDIT_SLOTS]
    for kind, ct in forged.items():
        loss = he.decrypt(ct, n) + offset
        in_bounds = ((bounds[0] - tol <= loss) & (loss <= bounds[1] + tol)
                     ).all()
        audit_ok = (np.abs(loss[audit.idx] - ref) <= tol).all()
        bad = np.abs(loss - ref_all) > tol
        # P(the mask hits one of the f faked values), hypergeometric.
        f = bad.sum()
        miss = np.exp(log_comb(n - f, AUDIT_SLOTS) - log_comb(n, AUDIT_SLOTS))
        print(f"  {kind}: mean loss {loss.mean():.6g}, faked {f} values, "
              f"range check {'ok' if in_bounds else 'FAILED'}, audit "
              f"{'ok' if audit_ok else 'FAILED'}, audit catches "
              f"{bad[draws].any(axis=1).mean():.1%} of masks (theory "
              f"{1 - miss:.1%})")


def audit_r2(y, p, rng):
    p, y = (p - y.mean()) / y.std(), (y - y.mean()) / y.std()
    he = HE(depth=1)
    n = he.slots
    ys, ps = y[:n], snap(p[:n])
    ct_y, = he.encrypt(ys)
    audit = Audit(he, n, rng)
    ct_loss = r2_eval(he, ct_y, ps)
    t_audit, cts_open = he.timed(audit_eval, he, audit.ct, ps)
    q = audit.open(he, cts_open[0])
    audit_check("R2", he, audit, ct_loss, ys ** 2, (q - ys[audit.idx]) ** 2,
                (ps - ys) ** 2, (0, np.inf), 1e-8, rng, t_audit, cts_open)


def audit_nll(y, mu, r, rng):
    n_bits = int(y.max() + 1).bit_length()
    grid_y, grid_r, u, s, vt = nll_basis(y)
    he = HE(depth=int(np.ceil(np.log2(n_bits))) + 3)
    n = he.slots
    y = y[:n]
    b = nll_label_features(y, grid_r, s, vt)
    cts = he.encrypt(*bits_of(y, n_bits), gammaln(y + 1), *b.T)
    audit = Audit(he, n, rng)

    # Evaluator: predictions on the grid, in log since they are positive.
    log_mu, log_r = snap(np.log(mu[:n])), snap(np.log(r[:n]))
    mu, r = np.exp(log_mu), np.exp(log_r)
    a, lo, hi = nll_clear(mu, r, grid_y, u, n_bits)
    ct_loss = nll_eval(he, cts[:n_bits], cts[n_bits], cts[n_bits + 1:], a,
                       mu, r, lo, hi)
    t_audit, cts_open = he.timed(audit_eval, he, audit.ct, log_mu, log_r)

    # Key holder: the same clear pipeline on the opened predictions.
    q_mu, q_r = (np.exp(audit.open(he, ct)) for ct in cts_open)
    q_a, q_lo, q_hi = nll_clear(q_mu, q_r, grid_y, u, n_bits)
    ref = nll_approx(y[audit.idx], b[audit.idx], q_a, q_mu, q_r, q_lo, q_hi)
    audit_check(f"NB NLL capped at {NLL_CAP:g}", he, audit, ct_loss,
                np.zeros(n), ref, nll_approx(y, b, a, mu, r, lo, hi),
                (0, NLL_CAP), 1e-3, rng, t_audit, cts_open)


def main(y, p):
    print(f"{len(y)} values")
    rng = np.random.default_rng(1)
    if p.ndim == 2:
        mu, r = np.clip(p[:, 0], EPS, None), np.clip(p[:, 1], *R_RANGE)
        bench_nll(y, mu, r)
        audit_nll(y, mu, r, rng)
        y, p = np.log1p(y), np.log1p(mu)
    bench_r2(y, p)
    audit_r2(y, p, rng)
    bench_mae(y, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("labels", nargs="?", help=".npy of the labels")
    parser.add_argument("preds", nargs="?",
                        help=".npy of the predictions, (n,) or (n, 2)")
    args = parser.parse_args()

    if args.labels:
        y = np.load(args.labels).ravel().astype(np.float64)
        p = np.squeeze(np.load(args.preds)).astype(np.float64)
    else:
        # NB counts with noisy (mean, dispersion) predictions.
        rng = np.random.default_rng(0)
        mu = np.exp(rng.normal(3, 2, 1_000_000)).clip(None, 1e4)
        r = np.exp(rng.normal(0.5, 1, mu.shape)).clip(*R_RANGE)
        y = rng.negative_binomial(r, r / (r + mu)).astype(np.float64)
        p = np.c_[mu * np.exp(rng.normal(0, 0.3, mu.shape)),
                  r * np.exp(rng.normal(0, 0.3, mu.shape))]
    main(y, p)
