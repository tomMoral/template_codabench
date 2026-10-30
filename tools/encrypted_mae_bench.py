"""Benchmark a mean absolute error computed on encrypted labels.

Use case: the score is computed in the same process as the submission, and
the hidden labels must not be readable there. The organizers encrypt the
labels offline (CKKS, OpenFHE), the evaluator computes the score on
ciphertexts from clear predictions, and only the secret key holder decrypts
it. This hides the labels but does not prevent score forging: anyone can
encrypt a fake score.

|yhat - y| = (yhat - y) * (2c - 1) with c = [y < yhat]. The labels are
rescaled to u in [0, 1) and encrypted with the L bits of u. The evaluator
knows the bits of yhat in the clear, so c is a comparison tree of depth
log2(L) on the encrypted bits, exact up to a resolution of 2^-L of the label
range. Smooth scores (squared error, likelihoods) need no such circuit: they
are linear in a few encrypted features of the labels.

The script prints, for each L, the MAE error of the bit circuit on all
values (exact numpy), then the single-core cost of one ciphertext and the HE
noise against numpy.

Requirements: ``pip install openfhe numpy``. The openfhe wheel needs Python
3.12 and glibc 2.39 (Ubuntu 24.04). Run single-threaded to read per-core
costs::

    OMP_NUM_THREADS=1 python tools/encrypted_mae_bench.py  # synthetic
    OMP_NUM_THREADS=1 python tools/encrypted_mae_bench.py y.npy yhat.npy

Reference, on a 12.5M-value regression test set with a 3.4% relative MAE,
one Xeon E5-2695 v3 core: L=12 gives a relative score error of 8e-5 for
0.23 ms and 13 ciphertexts per value (about 3.8 KB). A Chebyshev polynomial
of |x| of degree 1007 gives 3e-3 for 1.2 ms per value.
"""
import argparse
import time

import numpy as np
import openfhe as o


def bits_of(x, n_bits):
    """Bits of floor(x * 2^n_bits), most significant first, for x in [0, 1)."""
    q = np.minimum(np.floor(x * 2 ** n_bits), 2 ** n_bits - 1).astype(np.int64)
    return [((q >> (n_bits - 1 - k)) & 1).astype(np.float64)
            for k in range(n_bits)]


def abs_bits(u, v, n_bits):
    """|v - u| as the bit circuit computes it."""
    qu = np.minimum(np.floor(u * 2 ** n_bits), 2 ** n_bits - 1)
    qv = np.minimum(np.floor(v * 2 ** n_bits), 2 ** n_bits - 1)
    return (2 * (qu < qv) - 1) * (v - u)


def context(depth):
    params = o.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetSecurityLevel(o.SecurityLevel.HEStd_128_classic)
    cc = o.GenCryptoContext(params)
    for feature in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH,
                    o.PKESchemeFeature.LEVELEDSHE):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def encrypted_abs(cc, ct_u, ct_bits, v):
    """Encrypted |v - u| per slot, from Enc(u), Enc(bits of u) and clear v."""
    def pt(x):
        return cc.MakeCKKSPackedPlaintext(np.asarray(x, float).tolist())

    theta = bits_of(v, len(ct_bits))
    # Per bit: lt = [u_k < v_k] = t (1 - b), eq = [u_k == v_k], linear in b.
    lt = [cc.EvalAdd(cc.EvalMult(b, pt(-t)), pt(t))
          for b, t in zip(ct_bits, theta)]
    eq = [cc.EvalAdd(cc.EvalMult(b, pt(2 * t - 1)), pt(1 - t))
          for b, t in zip(ct_bits, theta)]
    # Merge (high, low) blocks: lt = lt_hi + eq_hi * lt_lo, eq = eq_hi * eq_lo.
    while len(lt) > 1:
        merged_lt, merged_eq = [], []
        for i in range(0, len(lt) - 1, 2):
            merged_lt.append(cc.EvalAdd(lt[i], cc.EvalMult(eq[i], lt[i + 1])))
            merged_eq.append(cc.EvalMult(eq[i], eq[i + 1])
                             if len(lt) > 2 else None)
        if len(lt) % 2:
            merged_lt.append(lt[-1])
            merged_eq.append(eq[-1])
        lt, eq = merged_lt, merged_eq
    sign = cc.EvalSub(cc.EvalAdd(lt[0], lt[0]), 1.0)
    diff = cc.EvalAdd(cc.EvalNegate(ct_u), pt(v))
    return cc.EvalMult(sign, diff)


def main(y, yhat, n_bits_list, reps=3):
    # Public bounds of the labels; predictions are clipped to them.
    lo, width = y.min(), (y.max() - y.min()) * 1.001
    u = (y - lo) / width
    v = np.clip((yhat - lo) / width, 0, 1 - 1e-12)
    exact = width * np.abs(v - u).mean()
    print(f"{len(y)} values, MAE {exact:.6g}")
    for n_bits in n_bits_list:
        err = abs(width * abs_bits(u, v, n_bits).mean() - exact) / exact
        print(f"L={n_bits:2d}: relative error of the bit circuit {err:.1e}")

    for n_bits in n_bits_list:
        cc, keys = context(depth=int(np.ceil(np.log2(n_bits))) + 3)
        slots = cc.GetRingDimension() // 2
        us, vs = u[:slots], v[:slots]

        def encrypt(x):
            return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(
                np.asarray(x, float).tolist()))

        start = time.perf_counter()
        ct_u = encrypt(us)
        ct_bits = [encrypt(b) for b in bits_of(us, n_bits)]
        t_enc = time.perf_counter() - start
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            ct = encrypted_abs(cc, ct_u, ct_bits, vs)
            times.append(time.perf_counter() - start)
        t_eval = float(np.median(times))
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(len(us))
        got = np.array(dec.GetRealPackedValue())
        noise = np.abs(got - abs_bits(us, vs, n_bits)).max()
        per_value = 1e6 / len(us)
        print(f"L={n_bits:2d}, N={2 * slots}: eval {t_eval * per_value:.0f} "
              f"us/value, encrypt {t_enc * per_value:.0f} us/value, "
              f"{n_bits + 1} ciphertexts/value, max HE noise {noise:.1e}",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("labels", nargs="?", help=".npy of the labels")
    parser.add_argument("preds", nargs="?", help=".npy of the predictions")
    parser.add_argument("--bits", type=int, nargs="+", default=[12, 16, 20])
    args = parser.parse_args()

    if args.labels:
        y = np.load(args.labels).ravel().astype(np.float64)
        yhat = np.load(args.preds).ravel().astype(np.float64)
    else:
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, 1_000_000)
        yhat = y + rng.laplace(0, 0.03, y.shape)
    main(y, yhat, args.bits)
