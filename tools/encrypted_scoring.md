# Scoring on encrypted labels

This note explains how a competition score can be computed on encrypted
labels, and what this costs for different kinds of metrics. The benchmark
`tools/encrypted_score_bench.py` measures the three routes described below.

## Setting

The organizers encrypt some features of the hidden labels offline, with a
public key. The evaluator runs in the same process as the submission. It
computes the score from the clear predictions and the ciphertexts, and only
the holder of the secret key can decrypt the result. The ciphertexts are
CKKS ciphertexts (OpenFHE, 128-bit security). Each one packs N/2 values, and
the evaluator works on all of them at once.

This hides the labels from the submission, which can read everything the
evaluator reads. It does not protect the score by itself. A submission that
controls the process can replace the encrypted score by the encryption of any
number, or shift some of its values. The audit described below detects this.

## Three routes

Write the score as a sum over values of a per-value loss f(p, y), with p the
prediction and y the label. The cost depends on how f mixes p and y.

1. **Linear.** f is a sum of clear functions of p times functions of y:
   f(p, y) = sum_k a_k(p) b_k(y). The organizers encrypt the b_k(y). The
   evaluator multiplies each ciphertext by the clear a_k(p) and adds them up.
   Any function of y alone is free, since the organizers compute it before
   encrypting. Any function of p alone is free too. The result is exact up to
   the CKKS noise.
2. **Low rank.** f has a smooth term that couples p and y, such as
   gammaln(y + r) in the Negative-Binomial NLL. An SVD of this term on public
   grids gives a sum of K separable terms, and the linear route applies with
   K features. The error is set by the rank K.
3. **Comparison.** f is not smooth where p = y, like |p - y|, or it involves a
   threshold of the loss, like a cap. Polynomials approximate such kinks
   badly. Instead, the organizers encrypt the L bits of the label, and a
   comparison tree computes [y < p] against the clear bits of p, with depth
   log2(L). The error is set by the resolution 2^-L of the label range.

## Measured costs

The numbers come from `tools/encrypted_score_bench.py` on 1M synthetic NB
counts, on one Xeon E5-2695 v3 core. "Eval" is the evaluator's time for the
encrypted part, "labels" is the size of the encrypted labels.

| Score | Route | Score error | Eval (us/value) | Encrypt (us/value) | Labels (KB/value) |
| --- | --- | --- | --- | --- | --- |
| R2 | linear | 2e-13 | 1.6 | 1.4 | 0.10 |
| NB NLL, rank 16 | low rank | 9e-9 | 22 | 31 | 2.3 |
| NB NLL capped at 100, rank 16, L = 17 | low rank and comparison | 6e-8 | 766 | 113 | 10.9 |
| MAE, L = 12 | comparison | 1e-6 | 221 | 38 | 3.8 |
| MAE, L = 16 | comparison | 6e-9 | 297 | 52 | 4.9 |
| MAE, L = 20 | comparison | 9e-12 | 407 | 67 | 6.7 |

The score error is relative, and it includes both the approximation and the
CKKS noise. For the NLL, the rank-16 approximation of gammaln(y + r) is off by
at most 2e-6 nats, and the cap interval matches the exact capped loss on all
values. The evaluator also spends 20 us/value in the clear, to compute the
prediction features and the cap interval. For comparison, a float64 label
takes 0.008 KB.

As a rule of thumb, 1M values cost about 2 s on one core on the linear
route, 20 s for the uncapped NLL, 3.5 to 7 min for the MAE, and 13 min for
the capped NLL. The ciphertexts are independent, so the time divides by the
number of cores.

## Bounding a loss without a cap

The cap makes the NLL 35 times slower, because it needs two comparisons.
Without it, the NB NLL is linear in y, gammaln(y + 1) and the 16 features
b_k(y), so it stays on the low-rank route. Bounding the predictions instead of
the loss keeps this cheap route:

- For the log loss, clipping the predicted probabilities to [eps, 1 - eps]
  bounds each term by -log(eps). It needs no comparison at all.
- For the NB NLL, clipping the mean to mu >= mu_min and the dispersion to a
  fixed range also bounds the loss, with no comparison. The bound grows with
  the count, though, like y log(y / mu_min). A lower bound on mu tames the
  worst case, a highly expressed gene predicted at mu near 0, but it does not
  give a fixed bound like 100 nats.
- If the cap only has to catch that case, a one-sided cap y <= hi halves the
  comparison cost.

On the synthetic data, the largest per-value NLL is 53 nats, so the cap never
triggers. Whether a cap is needed depends on how bad the worst submissions
are on the real data.

## Checking the score

Any prediction is a valid submission, so running the circuit on some other
predictions is not a forgery. A forgery is an output that no prediction
produces: a fresh encryption, an output shifted on some values, or features
a_k(p) computed from different p for different k. The audit catches these,
because the computation is fixed in advance:

- The evaluator returns one loss per value, not their sum. The key holder
  sums after decryption, and checks that every loss is within its bounds.
- The organizers draw a secret mask m with random weights alpha on k slots,
  and encrypt m alpha under a second secret key. The evaluator returns
  Enc(m alpha) times its clear predictions. The key holder decrypts it,
  divides by alpha and gets the predictions at the k audited slots.
- The key holder recomputes the loss at these slots, with the same clear
  code as the evaluator, and compares it with the decrypted loss.
- The predictions are rounded to a public grid (2^-20, in log for positive
  quantities) before scoring. The opened prediction then rounds back to the
  exact value the evaluator used, despite the CKKS noise.

No evaluation key involves the audit key, and OpenFHE refuses to add or
multiply ciphertexts under different keys. Forcing it gives a ciphertext that
fails to decrypt. So the evaluator can only multiply the mask by plaintexts,
even on the comparison route where it holds a relinearization key. It cannot
compute (1 - m) times a shift, which it would need to fake only the slots
that are not audited. To pass the check at an audited slot, it needs the
loss of the opened prediction there, which depends on the label. So the
values it fakes do not depend on the mask. With f faked values out of n, the
audit misses them with the hypergeometric probability C(n - f, k) / C(n, k),
close to (1 - f/n)^k. With k = 1400, it catches 1% of faked values with
probability 1 - 1e-6. The check runs offline with a secret mask, so the
submission gets a single try.

A faked value moves the score by at most the width of the loss bounds, so a
bounded loss also bounds what a forgery can gain. A shift smaller than the
tolerance is not detected, but the tolerance is set by the CKKS noise, so it
is negligible.

The numbers come from `tools/encrypted_score_bench.py`, on one ciphertext of
synthetic data, with k = 1400. The forger claims half of its mean loss on the
values it fakes.

| Score | Values | Honest HE error per value | Tolerance | Fresh encryption | 1% faked | 0.1% faked |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | 8192 | 5e-11 | 1e-8 | caught | caught | caught by 75% of masks (theory 73%) |
| NB NLL capped at 100 | 16384 | 3e-4 nats | 1e-3 nats | caught | caught | caught by 86% of masks (theory 83%) |

The honest outputs pass. The fresh encryptions pass the range check but fail
the audit. The audit adds 1.1 us/value to the evaluator on the R2 route and
6.2 us/value on the NLL route, for two opened predictions. The key holder's
check takes 5 to 15 us/value. The audit outputs can be summed over the
ciphertexts if each slot position is audited in at most one ciphertext, so
they take one ciphertext per opened prediction. The losses themselves are
not summed, so the output is about as large as one encrypted label feature.

The audit does not prove that the submission's model made the predictions.
It only shows that the score is the loss of the predictions it opens.

## Which route for which metric

| Metric | Route | How |
| --- | --- | --- |
| MSE, RMSE, R2 | linear | Encrypt y. The key holder adds sum(y^2), then takes a square root or divides by the SST. |
| Gaussian NLL with predicted mean and variance | linear | Encrypt y and y^2. |
| Poisson NLL, Gamma and Tweedie deviances with a fixed power | linear | Encrypt y and the label-only terms, such as log y!. |
| Accuracy, log loss, Brier score | linear | Encrypt the one-hot label. The argmax or the log of the clear probabilities are prediction features. |
| Precision, recall, F1, IoU, Dice | linear | TP, FP and FN are sums of products of clear predictions and one-hot labels. The key holder computes the ratios. |
| ROC AUC with binary labels, NDCG | linear | The ranks come from the clear predictions. AUC is a sum of the ranks of the positives, and the ideal DCG only depends on the labels. |
| Pearson and Spearman correlations | linear | The organizers encrypt y, or the ranks of y. |
| Per-group or weighted averages, slices | linear | Clear group masks or weights multiply the per-value terms. With a hidden group, encrypt y times the one-hot group. |
| Negative-Binomial NLL | low rank | gammaln(y + r) is close to a sum of 16 separable terms. A cap on the loss adds two comparisons. |
| Student-t NLL with a predicted degree of freedom, Gaussian CRPS | low rank | Check the rank with an SVD on the prediction and label ranges first. |
| MAE, MAPE, pinball and Huber losses | comparison | One comparison gives the sign of p - y. MAPE also needs 1/y as a label feature. |
| Capped or clipped losses | comparison | With a unimodal loss, the cap is a label interval that the evaluator computes from the prediction. |
| Hit rate within a tolerance | comparison | Two comparisons, against p - tol and p + tol. |
| Kendall tau | comparison | It needs n^2 comparisons, or n^2 encrypted label pairs. Not practical beyond a few thousand values. |
| Edit distances, DTW, matching-based detection metrics | none | Each cell of the dynamic program or of the matching needs a minimum over encrypted values. Not practical. |

The key holder decrypts sums, never per-value terms, except when the
ciphertexts are summed slot by slot after decryption. This is harmless
because the key holder already has the labels.

## Practical limits

The evaluator's cost scales with the number of scored values, not with the
number of samples. A test set of 1M values costs seconds on the linear route
and minutes on the comparison route. A dense signal, such as 2 kHz EMG over
hours of recording, has billions of values and hours of comparison circuits.
The encrypted labels also take more space than the clear ones: 0.1 KB per
value and per feature on the linear route, 4 to 11 KB per value on the
comparison route, against 0.008 KB for a float64.

For a competition where only the private score is sensitive, passing the
private predictions to a separate scoring container is often simpler. Use
the encrypted route when the predictions are too heavy to store, or when the
scoring must happen in the submission's process.
