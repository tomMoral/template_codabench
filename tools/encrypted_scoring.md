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
evaluator reads. It does not protect the score itself. A submission that
controls the process can replace the encrypted score by the encryption of any
number with the public key. Integrity needs another mechanism, for instance
rerunning the top submissions in a separate container.

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
| R2 | linear | 2e-13 | 1.7 | 1.4 | 0.10 |
| NB NLL capped at 100, rank 16, L = 17 | low rank and comparison | 2e-7 | 735 | 111 | 10.9 |
| MAE, L = 12 | comparison | 1e-6 | 212 | 38 | 3.8 |
| MAE, L = 16 | comparison | 6e-9 | 284 | 49 | 4.9 |
| MAE, L = 20 | comparison | 9e-12 | 389 | 66 | 6.7 |

The score error is relative, and it includes both the approximation and the
CKKS noise. For the NLL, the rank-16 approximation of gammaln(y + r) is off by
at most 2e-6 nats, and the cap interval matches the exact capped loss on all
values. Most of the NLL cost comes from the two comparisons of the cap. The
evaluator also spends 20 us/value in the clear, to compute the prediction
features and the cap interval. For comparison, a float64 label takes 0.008 KB.

As a rule of thumb, 1M values cost about 2 s on one core on the linear
route, 3.5 to 6.5 min for the MAE, and 12 min for the capped NLL. The
ciphertexts are independent, so the time divides by the number of cores.

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
| Negative-Binomial NLL | low rank | gammaln(y + r) is close to a sum of 16 separable terms. |
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
