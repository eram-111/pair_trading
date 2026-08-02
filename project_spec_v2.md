# Predicting Temporary Stock-Price Gap Convergence
## CSC311 Summer 2026 — Option 2 Final Project — Full Specification (v2)

**Team:** Alex Bachynsky, Golam Eram, Jaskaran Narula
**Deadline:** Friday, August 7, 2026, before 11pm EST

**What changed in v2:** Two optional extension tracks added at the end (Part 8): a partial-correlation distance metric for clustering (Rotondi & Russo 2024, ~5h) and autoencoder residuals replacing PCA (Krause & Calliess 2024, ~12–15h). Both are gated — we only build them if we are ahead of schedule at defined checkpoints. The core project (Parts 0–7) is unchanged and complete without them. Part 2 gains the two source papers, and Parts 5–7 are updated so the novelty accounting, expectations, and limitations stay honest whether or not the extensions run.

---

# PART 0 — THE SHORT SUMMARY

## What we are doing, in one paragraph

Some stocks normally move together. Sometimes one drifts away from the other for no good reason, and later they come back together. Traders bet on that coming-back-together. The classic way to decide when to bet is a **fixed rule**: "if the gap is bigger than usual, bet." We are replacing that fixed rule with a **model that learns from history** which gaps actually close and which ones don't. Then we measure — honestly — whether that learned rule is genuinely better, or whether it only *looks* better because it trades less often and therefore pays fewer fees.

## What we are building

A pipeline with five stages:

1. **Find the market-wide movement** in stock returns and remove it (using PCA), so we're left with each stock's own individual behaviour.
2. **Group similar stocks** (using k-means clustering) so we only look at pairs that plausibly belong together.
3. **Measure the gap** between two stocks in a group, and record whether that gap historically closed.
4. **Train a classifier** (logistic regression, plus comparisons) to predict whether a new gap will close.
5. **Test it honestly** — no peeking at the future, realistic trading costs, and a control experiment that separates "better signal" from "just trades less."

The core comparison runs two pair-selection tracks (behavioural residuals vs company characteristics) against four entry models. **If and only if we are ahead of schedule**, Part 8 defines two additional selection tracks — a partial-correlation distance metric and autoencoder residuals — that expand the grid from 2×4 up to 4×4. They are optional by design: the project is complete without them.

## Where the ideas come from

| Stage | Source |
|---|---|
| The problem (gaps close) | Gatev, Goetzmann & Rouwenhorst (2006) |
| PCA on returns → residuals | Avellaneda & Lee (2010) |
| Clustering to pick pairs | Zhang, Dai, Pan & Djabirov (2014) |
| Purged cross-validation | López de Prado (2018) |
| **Supervised entry filter** | Recent work (2025); our version differs — see Part 5 |
| **The factorial comparison + turnover control** | **Ours** |
| *Optional:* partial-correlation distance for clustering | Rotondi & Russo (2024) — Part 8 only |
| *Optional:* autoencoder residuals replacing PCA | Krause & Calliess (2024) — Part 8 only |

## Why it matters

Every paper we found that uses a classifier to time entries reports "our model beat the simple rule" — but none of them check whether the improvement is real signal or just an artifact of trading less. We check.

---

# PART 1 — THE PROBLEM

## 1.1 What is a "temporary price gap"?

Imagine two companies that are similar — say, two large soft-drink makers. Because they sell similar products to similar customers under similar economic conditions, their share prices usually move in roughly the same direction.

Now suppose one day Company A goes up 3% and Company B stays flat. That's a **gap**. Two possibilities:

- **Temporary:** nothing real happened. Maybe a big investor sold B that morning for unrelated reasons. Within a few days, the prices drift back into their usual relationship.
- **Permanent:** something real happened. A won a lawsuit, or B lost a major customer. The old relationship is broken and they will not reconverge.

The whole game is telling those two apart.

## 1.2 Why anyone cares

If you can identify a temporary gap, you buy the one that fell behind and sell the one that ran ahead. When they reconverge, you profit — and importantly, you profit *regardless of whether the overall market goes up or down*, because you hold one long position and one short position. That's called being **market-neutral**.

This strategy is called **pairs trading** or **statistical arbitrage**. It's been used by hedge funds since the 1980s.

## 1.3 What the classic approach does

Gatev, Goetzmann & Rouwenhorst (2006) formalized the classic version:

1. Find pairs of stocks whose prices historically tracked each other closely.
2. Watch the gap between them.
3. When the gap gets unusually wide (typically 2 standard deviations), place the bet.
4. Close when the gap normalizes.

Step 3 is a **fixed rule**. It doesn't matter what's happening in the market, what kind of pair it is, or whether gaps like this one have historically closed. Gap is big → bet. Every time.

## 1.4 What we think is missing

Not all wide gaps are equal. A 2.5-standard-deviation gap during a calm market between two stable stocks is probably different from a 2.5-standard-deviation gap during a market crash between two volatile stocks.

The fixed rule cannot tell the difference. **A model trained on history can.**

That is our central question:

> **Does a learned, probabilistic entry rule beat the classical fixed-threshold rule — and does any advantage survive transaction costs?**

## ⭐ In plain words

Two friends usually walk to school together. One day they're far apart. Will they come back together, or did they have a fight? The old method says "if they're far apart, bet they'll come back." We say "let's look at the last thousand times friends drifted apart, learn which ones came back, and only bet when this looks like one of those."

---

# PART 2 — THE PAPERS AND WHAT EACH GIVES US

We are not inventing statistical arbitrage. We are standing on four papers, with two more held in reserve for the optional extensions. Here is exactly what each one contributes.

## 2.1 Gatev, Goetzmann & Rouwenhorst (2006)

**Full citation:** Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *The Review of Financial Studies*, 19(3), 797–827.

**What they did:** Tested pairs trading on US daily data from 1962 to 2002. They matched stocks into pairs by finding the minimum distance between normalized historical prices, then traded a simple rule. They found average annualized excess returns of up to 11%, and the profits typically exceeded conservative transaction-cost estimates.

**What we take from them:**
- The problem definition itself
- The **fixed-threshold baseline** we compare against (this is our E0 model)
- The idea of a **random control**: they bootstrapped random pairs to check that their results weren't just generic mean-reversion. This is the direct ancestor of our turnover-matched control.

**What we do differently:** they used raw price distance to pick pairs. We remove market-wide factors first.

## 2.2 Avellaneda & Lee (2010)

**Full citation:** Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.

**This is our most important methodological source.** Steps 2, 3, and 6 of our pipeline are theirs.

**What they did:** Generated trading signals two ways — using PCA on returns, or by regressing stock returns on sector ETFs. In both cases, the idiosyncratic (leftover) returns are modelled as mean-reverting processes, which naturally leads to contrarian strategies. Their ETF-based strategy with volume information achieved a Sharpe ratio of 1.51 from 2003 to 2007.

**What we take from them:**
- **Rolling 252-day PCA** on the return correlation matrix
- Using the **correlation** matrix rather than raw covariance (so volatile stocks don't dominate)
- **Eigenportfolios** — turning eigenvectors into factor returns
- **Residuals** as the thing you actually trade
- The **s-score** concept, which we simplify into a rolling z-score

**What we do differently:** they fit an Ornstein-Uhlenbeck model to the residual and read off a threshold. We use a plain rolling z-score and then add a learned filter on top. We state this simplification explicitly in the report.

## 2.3 Zhang, Dai, Pan & Djabirov (2014)

**Full citation:** Zhang, W., Dai, Z., Pan, B., & Djabirov, M. (2014). A Multi-factor Adaptive Statistical Arbitrage Model. arXiv:1405.2384. Tepper School of Business, Carnegie Mellon University.

**What they did:** Argued that finding "similar" stocks by price history alone leaves information on the table. Instead they described each company by **19 fundamental characteristics** (valuation ratios, size, growth, profitability, analyst ratings), compressed those with PCA, and clustered with k-means. They tested this against graphical lasso and hybrid approaches on 109 S&P 500 stocks, 2004–2011.

**Their key results:**
- Clustering found **fewer but more profitable** portfolios than graphical lasso
- Clustering on **principal components** beat clustering on selected raw factors decisively (statistical arbitrage test: p = 0.01 passed vs p = 0.785 failed)
- Their "adaptive" experiment — re-selecting portfolios mid-period — **made things worse**, and they reported that honestly

**What we take from them:**
- The idea of **clustering to constrain the pair search space**
- The **characteristics pipeline** (our Track B): merge analyst recommendations, log-transform size columns, z-score everything, PCA, then k-means
- Their **cluster-to-portfolio rules** (skip singletons, take clusters of 2–4 whole, split larger ones)
- Their **survivorship-bias defense**, which we reuse

**Important note:** Zhang's PCA runs on *company characteristics*, not on returns. This is a completely different application from Avellaneda & Lee's. We use both, separately.

## 2.4 López de Prado (2018)

**Full citation:** López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (Chapter 7 in particular.)

**What we take:** the **purged cross-validation** and **embargo** methodology. Our labels look 5 days into the future, which means consecutive observations overlap in time and share future information. Without purging, a naive train/test split leaks. This is the single most important thing protecting our results from being fake.

## 2.5 Papers we cite but do not copy

**Han, He & Toh (2023),** *European Journal of Operational Research* 307(2), 929–947 — applied k-means, DBSCAN, and agglomerative clustering to US stocks 1980–2020. Critically, they compared clustering on **price information only** versus **price plus firm characteristics**, and found characteristics improved performance significantly (Sharpe fell from 2.34 to 1.76 for k-means when using price only).

We must cite this because it partly overlaps our Track A vs Track B comparison. Our version differs: they clustered on *raw returns*, we cluster on *factor-neutralized residuals*. Our question becomes: **does the characteristic advantage survive once common factors are already removed?**

Note also: their reported Sharpe ratios (2.69, 2.34) are very high and come from a much larger universe over 40 years. We should not expect anything close.

**ICBDEIM (2025) hybrid framework** — uses Engle-Granger cointegration plus Random Forest and Gradient Boosting classifiers to predict mean-reversion probabilities, explicitly reframing the problem from "which pairs to trade" to "when to trade a given pair," benchmarked against z-score rules.

**Ekinci et al. (2025),** *Computational Economics* — classified optimal-threshold ranges for high-frequency pairs trading using logistic regression, SVM, KNN, decision trees, random forests, and Naive Bayes.

**These two mean our "learned entry filter" is not novel as of 2025.** We cite them honestly and reposition our contribution around the experimental design instead (see Part 5).

**Sarmento & Horta (2020),** *ESWA* 158, 113490 — OPTICS/DBSCAN on PCA-reduced returns. Density-based clustering isn't in our course; cite in the lit review, don't build.

## 2.6 Optional extension source: Rotondi & Russo (2024)

**Full citation:** Rotondi, F., & Russo, F. (2024). Machine Learning for Pairs Trading: a Clustering-based Approach. SSRN Working Paper 5080998.

**What they did:** Clustered S&P 500 constituents (2000–2023) with k-means under **three different distance metrics** — plain Euclidean, a PCA-based Euclidean, and a **partial-correlation-based distance**, the last being a novel application in this context. They report modest average monthly excess returns of roughly 36–41 basis points — believable numbers, useful calibration for our own expectations.

**What we would take (Part 8, Track C only):** the partial-correlation distance as an alternative input to the *same* k-means step we already run. It isolates a single variable — the distance metric — with everything else in the pipeline fixed. Conceptually, partial correlation is the accessible cousin of Zhang's graphical lasso: it asks how two stocks co-move *after controlling for every other stock*, computed from the inverse of the correlation matrix we already estimate for PCA.

**Status:** cite in the literature review regardless; implement only if the Part 8 entry gate passes.

## 2.7 Optional extension source: Krause & Calliess (2024)

**Full citation:** Krause, F., & Calliess, J.-P. (2024). End-to-End Policy Learning of a Statistical Arbitrage Autoencoder Architecture. arXiv:2402.08233.

**What they did:** Observed that classical statistical arbitrage hinges on linear (asset-pricing or PCA-based) models to define the synthetic asset whose mean the strategy reverts to, and replaced that linear step with an **autoencoder** trained on US stock returns; they also develop an end-to-end policy-learning version.

**What we would take (Part 8, Track D only):** just their first idea — an autoencoder in place of PCA at the factor-extraction stage, with the residual defined as actual return minus reconstruction, and everything downstream unchanged. We explicitly **skip** their end-to-end policy learning: it is reinforcement-learning-flavoured and outside course scope.

**Why it's the best-matched extension:** it creates a controlled **linear-vs-nonlinear comparison at the identical pipeline position** (PCA residuals vs autoencoder residuals), mirroring at the representation stage the same bias-variance question our E1-vs-E3 comparison asks at the decision stage. Autoencoders are also literally in our course toolkit.

**Status:** cite in the literature review regardless; implement only if the Part 8 entry gate passes.

## ⭐ In plain words

Four cookbooks. Gatev tells us what dish we're making. Avellaneda gives us the main technique. Zhang gives us a different way to pick ingredients. López de Prado tells us how to taste-test without fooling ourselves. Two newer cookbooks already tried our "special twist," so we're honest about that and focus on doing the taste-testing better than anyone has. And two more cookbooks sit on the shelf — we'll only cook from them if dinner is ahead of schedule.

---

# PART 3 — THE PIPELINE, STEP BY STEP

This is the core of the document. Every step: what we do, how, why, and where it came from.

---

## STEP 0 — Choosing and downloading the data

### What we do

Select **40 US stocks** from the S&P 500, spread across **4 sectors, 10 stocks each**. Download **daily adjusted closing prices and volume** for **10 years (2015–2025)** using the free `yfinance` Python library.

### How we do it

```python
import yfinance as yf
prices = yf.download(tickers, start="2015-01-01", end="2025-01-01")["Close"]
```

We must confirm `auto_adjust=True` (the default) so prices are adjusted for stock splits and dividends. Unadjusted prices create fake jumps on split dates that would look like enormous returns.

### Cleaning checks

- Drop any ticker missing more than 2% of trading days
- Forward-fill isolated single-day gaps
- Align all series to a common trading calendar
- Verify no daily return exceeds ±50% without a real corresponding event
- Verify roughly 252 rows per year

### Why 40 stocks and not 500

Three reasons. First, PCA on a 40×40 correlation matrix estimated from 252 days is statistically reasonable; a 500×500 matrix from the same window is not. Second, we have 8 days. Third, the comparison between methods is what matters, and that comparison is valid on a small universe.

### The honest problem: survivorship bias

If we pick "40 S&P 500 stocks" using **today's** index membership and then test on data from 2015, we've selected companies that we already know survived and did well. That's using future information.

Zhang faced the same issue and handled it this way: they noted that obtaining historical index composition required a paid subscription, acknowledged the bias, and argued that with roughly 10 ticker changes per year out of 500 (about 2% turnover), the effect is small — **especially in a comparative setting where all strategies face the same data**.

We use the same argument. Our baseline and our model see the identical biased universe, so the *comparison* remains valid even if absolute numbers are optimistic. We state this plainly in the limitations section.

### Source

Standard practice. Universe construction is not from any specific paper. The survivorship-bias argument is adapted from Zhang et al. (2014).

### ⭐ In plain words

We pick 40 companies and download 10 years of their daily prices. We're honest that picking today's successful companies and testing them on the past is slightly cheating — but since every method we compare faces the same cheat, the comparison is still fair.

---

## STEP 1 — Converting prices into returns

### What we do

Convert each price series into **daily returns** — the percentage change from one day to the next.

### How

For simple returns: `r_t = (P_t - P_{t-1}) / P_{t-1}`

Or log returns: `r_t = ln(P_t / P_{t-1})`

We pick one and state it. Log returns add up neatly over time and behave slightly better statistically; simple returns are more intuitive. Either is defensible.

### Why

Prices aren't comparable across stocks. A $500 stock and a $20 stock can't be meaningfully compared. But "went up 2% today" means the same thing for both.

### What we end up with

A table called the **returns matrix**:
- Rows = trading days (about 2,500 of them)
- Columns = the 40 stocks
- Each cell = that stock's return on that day

Everything downstream operates on this table.

### ⭐ In plain words

Instead of "the stock costs $147," we say "the stock went up 1.3% today." Now we can compare a big expensive stock to a small cheap one fairly.

---

## STEP 2 — PCA on returns: finding and removing market-wide movement

**This is Track A, and it comes directly from Avellaneda & Lee (2010).**

### The idea first

On most days, most stocks move together. If the whole market drops 3%, almost every stock drops. That shared movement tells you nothing about any individual stock — it's noise for our purposes.

We want to find that shared movement and **subtract it out**, leaving each stock's own individual behaviour.

**Principal Component Analysis (PCA)** is the tool that finds shared patterns.

### The mechanics, in order

#### 2a. Set up the rolling window

For each trading day *t*, we use only the **previous 252 trading days** (one year). Avellaneda & Lee use a 252-day estimation window, so adopting it is defensible and citable.

```python
WINDOW = 252
for t in range(WINDOW, len(returns)):
    window = returns.iloc[t-WINDOW:t]   # excludes day t itself
```

**Critical:** we do *not* run PCA once on the entire dataset. That would use future data to define our factors, which is leakage. The slice excluding `t` is deliberate — an off-by-one error here is a leakage bug.

#### 2b. Standardize inside the window

```python
mu = window.mean()          # from THIS window only
sigma = window.std()        # from THIS window only
Z = (window - mu) / sigma
```

**This is leakage trap #1.** If we compute the mean over the full sample instead of the window, every single observation now contains information from the future. It won't throw an error. It will just quietly make our results better than reality.

#### 2c. Build the correlation matrix

Because we standardized, the covariance matrix *is* the correlation matrix:

```python
C = np.corrcoef(Z.values, rowvar=False)   # 40 x 40
```

Avellaneda & Lee deliberately use correlation rather than raw covariance so that high-volatility stocks don't dominate the factor structure. A biotech stock swinging 5% daily would otherwise define everything while a utility swinging 0.5% would be invisible.

#### 2d. Eigendecomposition

```python
eigenvalues, eigenvectors = np.linalg.eigh(C)
idx = np.argsort(eigenvalues)[::-1]      # eigh returns ascending; flip
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
```

We use `eigh`, not `eig`. The correlation matrix is symmetric, so `eigh` is faster and guarantees real-valued output. `eig` can return complex numbers with tiny imaginary parts from floating-point noise, which makes everything downstream confusing.

**What we're looking at:**
- **Eigenvector 1** is almost always all-positive across every stock. That's "the whole market moves together." It typically explains 30–50% of all variance.
- **Eigenvectors 2, 3, 4** capture finer structure, often sector-like (tech positive, energy negative).
- `eigenvalues[k] / eigenvalues.sum()` = the fraction of variance explained by component *k*.

#### 2e. Choose how many components to keep

We keep **3 to 5**. Options for deciding: a fixed count, a variance threshold (e.g. enough for 60%), or the scree-plot elbow.

**We pick one rule a priori, apply it consistently, and report it.** We never tune this on test performance — that would be selecting on the test set.

#### 2f. Fix the sign ambiguity

Eigenvectors are only defined up to sign. `[0.3, 0.5, 0.2]` and `[-0.3, -0.5, -0.2]` represent the same direction. Across rolling windows, numpy may return one on Tuesday and the flipped version on Wednesday, making our factors appear to jump violently for no reason.

```python
for k in range(n_components):
    if eigenvectors[:, k].sum() < 0:
        eigenvectors[:, k] *= -1
```

We apply the same convention every single window. This bug is subtle, common, and produces nonsense downstream.

#### 2g. Build eigenportfolio returns

Each eigenvector's elements become portfolio weights. Avellaneda & Lee weight each element by the inverse of that stock's volatility:

```python
weights = eigenvectors[:, k] / sigma.values
factor_return_k = (returns.iloc[t] * weights).sum()
```

### Output

For each day: 3–5 numbers representing what the market did.

### Sanity checks we must run

1. **Scree plot** — variance explained by each component. Component 1 should be clearly dominant.
2. **PC1 loadings** — should be roughly uniformly positive across all 40 stocks.
3. **PC1 factor return vs SPY** — these should track very closely. **This is our single best check** that the "market factor" is actually the market.
4. **Variance-explained over time** — plot the top-3 cumulative variance across rolling windows. It should *spike during 2020*, because correlations rise in crises. If we see that, PCA is working, and it's a good figure for the report.

### ⭐ In plain words

The fire alarm rings and everyone in the class stands up. That's not information about any individual kid. PCA finds "the fire alarm" — the thing that moved everybody at once — so we can ignore it and look at what each kid did on their own.

---

## STEP 3 — Residuals: what's left after removing the market

**Also from Avellaneda & Lee (2010).**

### What we do

For each stock, run a **linear regression** of its returns on the factor returns from Step 2:

```
stock_return = β₁·factor₁ + β₂·factor₂ + ... + βₖ·factorₖ + residual
```

The β coefficients say how sensitive that stock is to each factor. The **residual** is the part the factors do not explain — the stock's own idiosyncratic movement.

### How

Ordinary least squares, estimated on the trailing window. Then apply those betas going forward to compute out-of-sample residuals. This train-then-apply structure is what keeps it honest.

### A worked example

Suppose the market fell 3% today.

| Stock | Actual move | Market explains | **Residual** |
|---|---|---|---|
| Apple | −2.0% | −3.6% (beta 1.2) | **+1.6%** |
| Meta | −2.0% | −3.0% (beta 1.0) | **+1.0%** |
| Exxon | −0.5% | −1.5% (beta 0.5) | **+1.0%** |

Notice that Apple and Meta both fell 2%, but their residuals differ because their betas differ. The subtraction is **per-stock**, not one-size-fits-all.

### Why this matters enormously

If Apple and Meta both went up because the market went up, that tells us nothing. We only care when Apple's *own* behaviour diverges from Meta's *own* behaviour. Residuals isolate exactly that.

### Known problem to disclose

Betas drift. A beta estimated in January may be badly stale by June. Re-estimating every window helps, but stale betas turn a supposedly market-neutral spread into a directional bet. We say this in limitations.

### Output

A **residual matrix** — same shape as the returns matrix (days × stocks), but now each cell is the stock's leftover movement.

Each stock's column is its **fingerprint**: its personality after the market is stripped out.

### ⭐ In plain words

The bell rang and everyone stood up. But some kids stood up faster than the bell alone would explain, and some slower. That extra bit — faster or slower than expected — is the residual. It's personal to each kid.

---

## STEP 4 — Track A clustering: grouping stocks by behaviour

**The idea of clustering to constrain the pair search comes from Zhang et al. (2014) and Sarmento & Horta (2020).**

### What we do

Use **k-means** to group stocks whose residuals behave similarly.

### What we cluster on

Two options, and we should try both:

- **Option 1:** each stock's 252-length residual series (a long vector)
- **Option 2:** each stock's 3–5 factor loadings (a short, compressed vector)

Zhang's evidence favours the compressed version. When they compared clustering on selected raw factors versus clustering on principal components, the PC-based version passed the statistical-arbitrage test (p = 0.01) while the raw-factor version failed badly (p = 0.785).

The reason: principal components are **uncorrelated by construction**. Raw features overlap, so overlapping ideas get counted multiple times in the distance calculation, distorting the clusters.

### How k-means works

1. Pick a number *k* (how many groups)
2. Initialize *k* centroids
3. Assign each stock to its nearest centroid (Euclidean distance)
4. Recompute each centroid as the mean of its members
5. Repeat 3–4 until assignments stop changing

**Implementation details that matter:**
- Use **k-means++** initialization. Plain random init frequently drops two centroids inside the same natural group, wasting a cluster.
- Use **multiple restarts** (`n_init=10`), because k-means converges to local optima. Keep the lowest-inertia run.
- **Handle empty clusters** — reseed on the point furthest from its centroid.

### Choosing k

Zhang used K=30 for 109 stocks to get clusters of about 2–4 stocks. Scaled to our 40 stocks, that suggests **k = 8 to 13**.

We choose using the **elbow method** (inertia vs k) or **silhouette score**, computed on the **formation window only**. Choosing k by trading performance would be tuning on the test set.

### The label-switching problem

Re-running k-means shuffles the labels. "Cluster 2" this window might be "Cluster 5" next window with identical members. This makes tracking impossible if we rely on labels.

**Solution:** never track raw labels. Only ask "are stocks i and j in the same cluster, yes or no?" That's a **co-membership matrix**, and it's invariant to relabelling.

### Stability analysis

We measure: what fraction of stock pairs that were co-clustered in window *w* are still co-clustered in window *w+1*?

This is honest analysis and turns an annoyance into a reportable finding. We expect Track B (characteristics) to be more stable than Track A (behaviour), because fundamentals change slowly.

### Why cluster at all?

With 40 stocks there are 780 possible pairs. Testing all of them and reporting the best is **multiple testing** — with enough attempts, something always looks good by luck. Clustering narrows the candidates *before* we look at outcomes, which is honest.

### ⭐ In plain words

We watch each kid's personal behaviour for a year and group together the ones who act alike. We don't compare every kid to every other kid — that's 780 comparisons and we'd find fake patterns. We only compare kids who are already in the same group.

---

## STEP 4B — Track B: grouping stocks by what kind of company they are

**This is Zhang et al. (2014), implemented separately from Track A.**

### The core difference

| | Track A | Track B |
|---|---|---|
| PCA runs on | 252 days × 40 stocks of **returns** | 40 stocks × 19 **characteristics** |
| Correlation matrix is | 40×40 (stock by stock) | ~18×18 (characteristic by characteristic) |
| An eigenvector is | a portfolio (weight per stock) | a **recipe** (weight per characteristic) |
| Groups stocks by | how they **behave** | what kind of **company** they are |

### 4B.1 — Getting the data

From Bloomberg (which we have access to), pull **19 characteristics** per stock:

| Category | Fields |
|---|---|
| Valuation | P/E, Price/Book, Price/Sales, Price/EBITDA |
| Size | Market Cap, Shares Outstanding |
| Growth | Sales Growth, Cash Flow Growth, Free Cash Flow Growth |
| Profitability | Normalized ROE, Dividend per Share |
| Trading | Volatility (60-day), RSI, Close Price, Ask, Bid |
| Sentiment | Analyst Rating, Buy Recommendations, Sell Recommendations |

**Pull quarterly, not daily.** Fundamentals only update quarterly. That's 40 stocks × 19 fields × ~40 quarters ≈ 30,000 cells — one terminal session.

### ⚠️ Point-in-time warning

Bloomberg may return fundamentals computed from **restated** financials. A P/E pulled for March 2018 may reflect numbers nobody actually had in March 2018. That's look-ahead bias — the same class of error we're careful about everywhere else.

Prefer as-reported fields where available. If we can't verify, we disclose it in limitations. Zhang doesn't discuss this issue, so acknowledging it makes our treatment **more rigorous than the paper we're extending**.

### 4B.2 — Cleaning (Zhang's three steps)

**Step 1: Merge analyst recommendations.**

```
sentiment = (buy_count - sell_count) / (buy_count + sell_count)
```

A stock with 15 buys and 5 sells scores +0.5. Range is −1 to +1. Two columns saying the same thing become one, and it's scale-free — a stock covered by 40 analysts doesn't outweigh one covered by 5. **19 columns become 18.**

**Step 2: Log-transform the size columns.**

Raw market caps: Apple ~$2,800,000M, a mid-cap ~$40,000M. That's **70× difference**. After log: 14.85 vs 10.60 — about 1.4× difference.

Zhang's justification: Axtell showed US firm sizes follow a **Zipf-law distribution** — a few giants and a long tail of small firms. Without the log, Apple's size alone would swamp every distance calculation and our clusters would just be "big" versus "everyone else."

**Step 3: Z-score every column.**

This one cannot be skipped. Market cap is in the millions. P/E is around 20. ROE is around 0.4. PCA finds directions of maximum variance — so on raw data, component 1 would be **100% market cap and nothing else.** Useless.

For each column: subtract the mean, divide by the standard deviation. Now every characteristic competes on equal terms.

**Our addition:** Zhang dropped any *stock* missing any field, going from 500 stocks to 109. With only 40 stocks we can't afford that, so we drop sparse *columns* instead and median-impute the few remaining gaps. Both choices go in limitations.

### 4B.3 — PCA on characteristics

```python
C = np.corrcoef(X, rowvar=False)      # ~18 x 18
eigenvalues, eigenvectors = np.linalg.eigh(C)
```

Zhang found **7 components explained 95.5% of the total variance** across their 19 characteristics. That's because the characteristics overlap heavily — P/E, Price/Sales, and Price/EBITDA all measure "how expensive is this stock," just three different ways.

**Caveat we must note:** estimating an 18×18 correlation matrix from only 40 observations is statistically thin. Zhang had 109 stocks. Our trailing components will be noisy. We keep fewer components as a result.

### 4B.4 — Reading the components

This is the fun part, and it's a Clarity win for the report. Look at eigenvector 1's weights:

| Characteristic | Weight on PC1 |
|---|---|
| P/E | 0.42 |
| P/B | 0.39 |
| P/S | 0.41 |
| P/EBITDA | 0.38 |
| Sales Growth | 0.21 |
| Market Cap | 0.05 |
| Volatility | −0.02 |

All four valuation ratios load heavily; everything else is near zero. **PC1 = "expensiveness."** PC2 might be growth. PC3 might be size.

We interpret and name them in the report. It proves the PCA found something meaningful rather than being a black box.

### 4B.5 — Zhang's two routes

**Route A — PCA as a detector.** Read the loadings, find which *original* characteristics carry the most weight, keep those raw columns, discard PCA. Zhang's survivors were P/E, Price/Sales, Cash Flow Growth, Price, Price/EBITDA, ROE, Volatility.

**Route B — PCA as a transformer.** Project the stocks onto the components. Each stock becomes 7 component scores, e.g. `AAPL = [+1.4 expensiveness, +0.9 growth, +2.1 size, ...]`.

**Zhang found Route B decisively better** (p = 0.01 passed vs p = 0.785 failed). We use Route B, and run Route A too if time allows, citing Zhang's comparison as precedent.

### 4B.6 — Clustering

k-means with **k = 10 to 13** for our 40 stocks. Then Zhang's group rules:

- Cluster of 1 stock → skip (can't build a spread with one leg)
- Cluster of 2, 3, or 4 → take the whole cluster
- Cluster of 5+ → split into subgroups of 2–3

Splitting large clusters keeps group sizes comparable so Track A and Track B aren't compared at different granularities.

### ⭐ In plain words

Instead of watching how kids behave, we look at facts about them — their age, height, what class they're in, what they're good at. Then we group the kids who are *similar as people*, not just kids who happened to walk the same way today. The idea is that genuinely similar kids have a real reason to stay together.

---

## STEP 5 — From clusters to candidate pairs

### What we do

Within each cluster, form all possible pairs. A cluster of 4 stocks gives 6 pairs. Across all clusters we might end up with 60–120 candidate pairs instead of 780.

### Output format

```
pair_id | stock_a | stock_b | group_id | source
```

Where `source` is "track_a" or "track_b" (extended to "track_c" / "track_d" if Part 8 runs), so merged results stay attributable.

### ⭐ In plain words

Now we have a list of "these two kids are friends, watch them."

---

## STEP 6 — Spread and z-score: measuring the gap

**Simplified from Avellaneda & Lee's s-score.**

### 6a. Build the spread

Two options:

**Simple:** `spread = residual_A - residual_B` (cumulative over the window)

**Hedge-ratio version:** regress stock A's residuals on stock B's residuals to get a ratio *h*, then `spread = residual_A - h × residual_B`. This balances the two legs so the position is properly market-neutral.

Estimate *h* on the formation window only. Either version is defensible; the simple one has less to go wrong.

### 6b. Compute the z-score

```
z_t = (spread_t - rolling_mean) / rolling_std
```

Where the mean and standard deviation come from the **trailing window, not the full sample**. Same leakage trap as Step 2b.

A z-score of +2.5 means "this gap is 2.5 standard deviations wider than typical for this pair."

### What we're simplifying

Avellaneda & Lee's s-score is derived from an Ornstein-Uhlenbeck process fitted to the residual — more sophisticated than a plain rolling z-score. **We state this explicitly:** "we use a simplified rolling z-score rather than the OU-derived s-score of Avellaneda & Lee." That's honest and shows we know the difference.

### ⭐ In plain words

We measure the distance between the two friends, and then ask "is this distance weird?" A z-score of 2.5 means "much bigger gap than these two usually have."

---

## STEP 7 — Building the labeled dataset

**This is our own formulation. Neither Avellaneda & Lee nor Zhang builds a labeled dataset — both use rules and statistical tests instead.**

### 7a. The trigger

An observation is created whenever `|z|` **first crosses 2.0** after having been below it.

**Only onsets count.** If we created an observation for every day the spread stays wide, we'd get hundreds of near-duplicate overlapping examples that would corrupt training.

### 7b. The label

```
y = 1  if |z| falls to ≤ 50% of its trigger value within the next 5 trading days
y = 0  otherwise
```

So if z triggers at 2.4, we check whether `|z|` reaches ≤1.2 within 5 days.

**Every parameter must be stated and justified:** the trigger threshold (2.0), the reversion fraction (50%), the horizon (5 days). We choose these *a priori* or tune on validation — never on test.

### 7c. Check the base rate first

Before interpreting any model output, we count what fraction of triggers actually revert.

- If 80% revert → the problem is nearly trivial and our baseline will be strong
- If 25% revert → we have severe class imbalance and must handle it

Either way we need to know before trusting any number.

### 7d. The critical property

This label **looks 5 days into the future**. That's fine for training, but it means consecutive observations overlap in time and share future information. This is exactly why Step 10's purging exists.

### What the dataset looks like

| Trigger # | \|z\| | spread vol | momentum | mkt vol | ... | **Reverted?** |
|---|---|---|---|---|---|---|
| 1 | 2.3 | 0.8 | −0.4 | low | ... | **Yes** |
| 2 | 2.7 | 1.9 | +0.9 | high | ... | **No** |
| 3 | 2.1 | 0.6 | −0.2 | low | ... | **Yes** |
| ... | | | | | | |
| 847 | 2.4 | 1.2 | +0.1 | med | ... | **No** |

**That last column does not exist in any of our source papers.** Creating it is what makes this a machine learning project rather than a rule-following exercise.

### ⭐ In plain words

We go through ten years of history, find every time two friends drifted apart, and write down whether they came back together. Now we have a big list of "here's what it looked like, and here's what happened." That list is what the computer learns from.

---

## STEP 8 — Features: what the model gets to see

**Our own design. Volume as a feature is precedented by Avellaneda & Lee, who found their ETF strategy with volume information achieved a Sharpe ratio of 1.51.**

### The seven features

| Feature | How computed | Why it might matter |
|---|---|---|
| \|z\| at trigger | current z-score magnitude | how extreme the divergence is |
| Spread volatility | rolling std of spread, 60d | is this pair usually jumpy or stable |
| Residual momentum | sum of residual returns, 5–10d | is the gap still widening or already narrowing |
| Market volatility | rolling std of PC1, 20d | calm or chaotic conditions |
| Relative volume | volume vs its 20d average | is unusual activity involved |
| Days since last trigger | count | is this pair firing constantly (unstable) |
| Cluster stability | did the pair survive last window | confidence in the peer relationship |

### The rule

**All features must use only information available at or before the trigger time.** Standardize using training-window statistics only.

### The hypothesis being tested

A z-score of 2.5 means different things in different contexts. A fixed rule cannot distinguish. A model with these features can. **Whether it actually can is our research question.**

### Why no PCA here

We only have 7 features and they don't badly overlap, so there's nothing to compress. More importantly, compressing them would destroy our ability to read the model's coefficients — which is one of the main reasons we chose logistic regression.

### ⭐ In plain words

For each gap, we write down seven facts about the situation. Then the computer learns which combinations of facts usually mean "this gap will close."

---

## STEP 9 — The models

### E0 — Fixed z-score rule (the baseline)

```
Enter when |z| > 2.0
Exit when |z| < 0.5, or after 5 days, whichever comes first
```

No learning at all. This is essentially Gatev's rule and Avellaneda & Lee's approach.

**We must implement this fairly** — same data, same costs, same backtest. The handout warns against "compar[ing] unfairly against other methods." A well-implemented baseline that beats our models is a legitimate result; a strawman baseline is a rubric violation.

### E1 — Logistic regression (our primary model)

Takes the 7 features, computes a weighted sum, squashes it through the sigmoid function into a probability *p* = P(gap closes).

**Settings:**
- L2 regularization, strength tuned on validation
- `class_weight='balanced'` if the base rate is skewed
- Decision threshold τ chosen on validation, never test

**The decision rule:**

| | Enters when |
|---|---|
| Baseline (E0) | \|z\| > 2.0 |
| **Ours (E1)** | \|z\| > 2.0 **AND** p > τ |

Our model is strictly a **filter** on the baseline — it takes a subset of the same trades. That makes the comparison clean, and it makes the turnover question sharp.

**We report the coefficients.** "Market volatility carries a negative weight — gaps close less reliably in chaotic conditions" is a genuine finding, and it's why we chose an interpretable model.

### E2 — Gaussian Discriminant Analysis (generative comparison)

Same question, opposite approach. Instead of learning the boundary between classes, GDA models what each class *looks like* as a multivariate Gaussian, then applies Bayes' rule.

**Why include it:** discriminative vs generative is core course material. Theory says GDA wins with less data when its Gaussian assumption holds, and logistic regression wins when it doesn't. Testing which happens here is a real result.

### E3 — Small neural network (nonlinear baseline)

A small MLP — 1–2 hidden layers, 8–16 units, sigmoid output, early stopping, weight decay.

**Expected result: it does not beat logistic regression.** Financial data has terrible signal-to-noise, and the extra flexibility buys variance without buying signal. Demonstrating that cleanly is the bias-variance lesson from the course shown on real data — a *good* outcome, not a failure.

**Keep it small.** A large net on a few hundred training examples will memorize noise and we'll have made the point too easily.

### The bias-variance ladder

| Model | Bias | Variance |
|---|---|---|
| Fixed z-score rule | High | Low |
| Logistic regression | Medium | Medium |
| Neural network | Low | High |

Financial data is so noisy that the middle usually wins. Demonstrating this cleanly is one of the strongest things our report can do.

### ⭐ In plain words

Four ways to decide whether to bet. The dumb one: always bet when the gap is big. The smart one: bet only when history says gaps like this usually close. Plus two other smart ones we compare against, to check whether fancier is actually better. (Usually it isn't.)

---

## STEP 10 — Honest evaluation

**This is where 50% of our grade lives.**

### 10a. Chronological splits

| Period | Years | Purpose |
|---|---|---|
| Train | 2015–2020 | fit the models |
| Validation | 2021–2022 | tune hyperparameters, choose τ and k |
| Test | 2023–2025 | touched **exactly once**, at the end |

The handout explicitly warns against "set[ting] hyperparameters using test accuracy." The validation set exists so we never have to.

### 10b. Purging

Our labels look 5 days forward. An observation triggered 3 days before the train/validation boundary has a label determined by data *inside* validation.

**Fix:** remove all training observations whose label horizon extends past the boundary. With a 5-day horizon, drop the last 5 days of each training block.

### 10c. Embargo

Beyond purging, leave an additional **10 trading days** between train and validation. This handles subtler correlations — a spread's behaviour on day T is correlated with day T+3 even without direct label overlap.

**Source:** López de Prado (2018), Chapter 7.

### 10d. Metrics

**Classification:**
- **AUC** (primary) — how well the model ranks reverting gaps above non-reverting ones. Threshold-independent, robust to imbalance.
- Precision and recall at our chosen τ
- **Calibration** — when the model says 60%, do about 60% actually revert? A reliability diagram. Almost no finance-ML paper reports this.

**Explicitly not accuracy.** If 70% of gaps don't revert, predicting "never" scores 70% and is worthless.

**Strategy:**
- Hit rate, mean return per trade, cumulative return
- **Sharpe ratio** — with the smell test attached
- **Turnover** — number of trades, essential for the signal-vs-fewer-fees question
- Max drawdown

**Report uncertainty, not point estimates.** Bootstrap confidence intervals on everything. Given our sample size the intervals will be wide — and showing that is *rigor*, not weakness. "AUC 0.54 [0.49, 0.59]" tells an honest story that "AUC 0.54" doesn't.

### 10e. The smell test (post this visibly)

| Sharpe ratio | Interpretation |
|---|---|
| 0 to 1 | Realistic |
| Above 2 | **Suspect a bug** |
| Above 3 | **Almost certainly leakage** |

Realistic AUC for this problem: **0.52 to 0.58**. An AUC of 0.85 means we have a bug.

Because we don't have finance backgrounds, we lack the instinct to smell when a result is wrong. This table substitutes for that instinct. **We treat a good result as a bug report until proven otherwise.**

### ⭐ In plain words

We only let the computer study the past, never the future. We keep one chunk of data completely hidden until the very end, and only look at it once. And if our results look amazing, we assume we made a mistake, because amazing results in this field usually mean a bug.

---

## STEP 11 — Transaction costs

### What we do

Every trade costs *c* basis points, applied on **both entry and exit**, on **both legs**. A pair trade therefore costs roughly 4*c*. Realistic *c* for liquid US equities: 5–10 bps including spread and slippage.

**Applied at the point of each transaction**, not as a lump subtraction at the end. That's a common bug.

### The cost-sensitivity sweep (our signature figure)

Re-run everything with *c* from 0 to 50 bps. Plot net performance versus assumed cost, one line per strategy.

Two things emerge:

**The breakeven cost** — where each strategy's net return hits zero. "The fixed rule breaks even at 8 bps; the filtered strategy survives to 14 bps" is a concrete, interpretable finding.

**The crossover** — if our model trades less, it may lose at zero cost but win at realistic cost. That's an interesting and honest result.

### Why this matters

It directly addresses the gap between "statistically predictable" and "actually tradeable." Gatev et al. found profits typically exceeded conservative transaction-cost estimates in their era. Testing whether that still holds in ours is a meaningful question.

### ⭐ In plain words

Every bet costs a fee. We test how big the fee has to get before betting stops being worth it. That number — the breakeven fee — tells you far more than "we made X%."

---

## STEP 12 — The experiments

### 12a. The factorial grid

| | E0 Fixed | E1 Logistic | E2 GDA | E3 Neural net |
|---|---|---|---|---|
| **Track A** (residuals) | | | | |
| **Track B** (characteristics) | | | | |

Rows = how we pick pairs. Columns = how we decide to trade. **If either Part 8 extension passes its gate, the grid gains a Track C and/or Track D row (up to 4×4); the questions below don't change, they just gain data points.**

**Three questions this answers:**

1. **Row effect:** does pair selection matter? Average down each column.
2. **Column effect:** does the entry decision matter? Average across each row. **If E1 beats E0 regardless of track, our core claim holds.**
3. **Interaction:** does the best model depend on the selection method? Maybe learned filtering helps a lot for weak selection and barely at all for strong selection — because good pairs don't need filtering.

### 12b. The turnover-matched control ⭐

**This is our most important experiment and our strongest remaining originality claim.**

Our filter trades less by construction. So if it performs better, there are two possible explanations:

1. The model genuinely identifies better opportunities (**real signal**), or
2. The model just trades less often, so it pays fewer fees (**nothing to do with signal quality**)

Those look identical in a results table.

**To separate them:** build a control that trades exactly as rarely as our model but picks **randomly**. If our model beats the random-but-equally-rare filter, the gain is real signal. If it doesn't, our "improvement" was just reduced trading costs.

**We found no paper that does this.** Every recent paper reports "our classifier beat the z-score rule" without checking.

The idea traces to Gatev et al., who bootstrapped random pairs to distinguish pairs trading from pure mean-reversion.

### 12c. Consensus pairs ⭐

We have two pair lists. The obvious analysis is "which is better." The interesting one is:

**Are pairs that BOTH methods select better than pairs either method picks alone?**

Three buckets:
- **Consensus** (both methods agree)
- **Track A only** (behavioural similarity)
- **Track B only** (economic similarity)

Compare reversion rate, AUC, and net performance for each.

**The logic:** if a pair looks similar *economically* AND *behaviourally*, that's two independent pieces of evidence for a real relationship. Pairs selected by only one method might be picking up coincidence.

**Effort: about 4 hours.** It's a set intersection. Nobody has published it.

(If Part 8 runs, the consensus analysis naturally extends to more buckets — e.g. pairs selected by 3+ methods — but the two-track version is the committed deliverable.)

### 12d. Does the characteristic advantage survive factor neutralization?

Han et al. (2023) found characteristics improve clustering — but they clustered on **raw returns**, which are dominated by market beta. So there's an unresolved possibility: maybe characteristics were partly proxying for factor exposure all along.

Our setup tests this directly, because Track A gives factor-neutralized residuals:

> Once common factors are removed, do fundamental characteristics still add information?

**Either answer is a real finding:**
- Characteristics still help → they carry genuine economic information beyond factor loadings
- The advantage vanishes → Han's result may partly reflect characteristics proxying for factor exposure

That second finding would be a legitimate qualification of a 2023 *EJOR* paper.

### 12e. Cheap additions with high information value

- **Coefficient comparison** (1h): does the classifier learn the same thing in both tracks?
- **Base-rate decay** (1h): plot reversion rate by year, 2015–2025. If it's falling, the opportunity is being arbitraged away.
- **Error analysis** (3h): look at confident false positives. Clustered in crises? In specific sectors? In unstable pairs?

### ⭐ In plain words

We test every combination of "how we pick friends" × "how we decide to bet," and look for patterns. Then the important one: if our smart method wins, is it because it's actually smarter, or just because it bets less and pays fewer fees? We build a fake method that bets equally rarely but picks at random. If we beat that, we're genuinely smarter.

---

# PART 4 — THE LEAKAGE AUDIT

**Leakage = accidentally letting the model see the future.** It's the single most common way finance projects produce fake results. It doesn't throw an error — it just quietly makes everything look better than reality.

## The checklist (this goes IN the report)

1. PCA factors estimated on trailing windows only? ✓
2. Standardization statistics window-local, never full-sample? ✓
3. Regression betas estimated before the period they're applied to? ✓
4. Clustering performed on formation-window data only? ✓
5. Labels purged at split boundaries? ✓
6. Embargo applied? ✓
7. Test set touched exactly once? ✓
8. Backtest uses signal at time *t* with returns at *t+1*? ✓ **(hand-traced 3 specific dates)**
9. Costs applied at each transaction? ✓
10. Bloomberg fields point-in-time, or disclosed? ✓
11. *(If Track C runs)* Precision matrix estimated in-window only, shrinkage documented? ✓
12. *(If Track D runs)* Autoencoder trained on trailing data only, retraining schedule documented, inputs standardized window-locally? ✓

## The two decisive tests

**The noise test.** Run the entire pipeline on synthetic random-walk price series with no real relationships. **If it finds signal, we have a bug.** Nothing else catches leakage this reliably. (Any Part 8 track that runs must also pass the noise test before its results are reported.)

**The manual trace.** Pick three specific dates, follow the position logic on paper. Confirms we're not using today's signal with today's return — a bug that runs perfectly, throws no error, and produces fantastic fake results.

## Why we publish the checklist

Most student projects either don't think about leakage or quietly hope they got it right. Showing our work on this is a direct play for the Quality mark, and it's the thing that separates a credible study from a plausible-looking one.

### ⭐ In plain words

The biggest way to cheat by accident is to let the computer peek at the future. We made a checklist of every place that could happen, checked each one, and put the checklist in our report. We also ran our whole system on completely random fake data — if it "finds" a pattern in pure noise, we know something's broken.

---

# PART 5 — WHAT'S BORROWED AND WHAT'S OURS

## The honest accounting

| Step | Source | Ours? |
|---|---|---|
| 0–1. Data, returns | Standard practice | — |
| 2. Rolling PCA, eigenportfolios | **Avellaneda & Lee (2010)** | parameter choices only |
| 3. Residuals via factor regression | **Avellaneda & Lee (2010)** | — |
| 4. Clustering to constrain search | **Zhang (2014)**, Sarmento & Horta (2020) | k-means substitution, stability analysis |
| 4B. Characteristics pipeline | **Zhang (2014)** | missing-data handling |
| 5. Cluster → pairs rules | **Zhang (2014)** | — |
| 6. Spread, z-score | **Avellaneda & Lee (2010)** (s-score) | documented simplification |
| 7. Labeled dataset formulation | concept from Gatev / A&L | ✅ **our formulation** |
| 8. Feature set | volume from A&L | ✅ **our design** |
| 9. E0 fixed baseline | **Gatev (2006)** | — |
| 9. E1–E3 learned entry | direction precedented (2025) | ✅ our implementation |
| 10. Purged CV, embargo | **López de Prado (2018)** | parameter choices |
| 11. Cost sweep, breakeven framing | cost-awareness from Gatev/A&L | ✅ **ours** |
| 12b. **Turnover-matched control** | inspired by Gatev's bootstrap | ✅ **ours** |
| 12c. **Consensus pairs** | — | ✅ **ours** |
| Leakage audit in report | López de Prado | ✅ **ours** |
| *Optional* Track C: partial-correlation distance | **Rotondi & Russo (2024)** | shrinkage handling, integration |
| *Optional* Track D: autoencoder residuals | **Krause & Calliess (2024)** | linear-AE-recovers-PCA validation, integration |

## ⚠️ The honest correction on novelty

We must be upfront about this. When we first designed this project, we believed the "learned entry filter" was novel. **It is not, as of 2025.** Two papers do essentially what we planned:

- The **ICBDEIM (2025) hybrid framework** uses cointegration plus Random Forest and Gradient Boosting classifiers to predict mean-reversion probabilities, explicitly reframing the problem from "which pairs to trade" to "when to trade a given pair," and benchmarks against z-score rules with realistic transaction costs.
- **Ekinci et al. (2025)**, *Computational Economics*, classify optimal-threshold ranges using logistic regression, SVM, KNN, decision trees, random forests, and Naive Bayes.

**We cite both.** Overclaiming would be worse than the loss of novelty — and a grader checking the literature would catch it.

## What remains genuinely ours

Three things, and they're real:

1. **The factorial design.** Both papers fix one selection method and vary the model. Neither crosses selection × entry to separate row effects from column effects. Our central question — *does learned filtering help regardless of how pairs were chosen?* — is not answered anywhere we found. **Every extension row from Part 8 that runs makes this claim stronger**, because a column effect that holds across three or four selection paradigms is far more convincing than one that holds across two.

2. **The turnover-matched control.** Our strongest claim. Every paper reports "our classifier beat the rule" without checking whether that's better signal or just fewer trades. **We found no paper that disentangles this.**

3. **Consensus pairs.** Nobody has asked whether agreement between economically-motivated and behaviourally-motivated selection is itself predictive.

## The contribution sentence for our report

**If the core 2×4 grid is what ships:**

> Recent work has applied supervised classifiers to pairs-trading entry decisions (Ekinci et al., 2025; ICBDEIM, 2025), but evaluates a single selection method and does not separate improved signal quality from reduced trading frequency. We evaluate learned entry filtering factorially across pair-selection methods drawn from Avellaneda & Lee (2010) and Zhang et al. (2014), and introduce a turnover-matched control to isolate the source of any improvement.

**If one or both Part 8 extensions ship:**

> Statistical-arbitrage pair selection has been approached through PCA residuals (Avellaneda & Lee, 2010), fundamental characteristics (Zhang et al., 2014), partial-correlation clustering (Rotondi & Russo, 2024), and autoencoder representations (Krause & Calliess, 2024); recent work has separately applied supervised classifiers to the entry decision (Ekinci et al., 2025; ICBDEIM, 2025) without separating improved signal quality from reduced trading frequency. We evaluate learned entry filtering factorially across these selection methods spanning 2010–2024, and introduce a turnover-matched control to isolate the source of any improvement.

Both are accurate, cite prior work, and state a real gap. For a "novel combination of familiar techniques," either is sufficient.

### ⭐ In plain words

Most of what we built, we borrowed — and we say exactly who from. Our own contribution is small but real: we test more combinations than anyone has, and we check something everyone else forgot to check.

---

# PART 6 — WHAT WE EXPECT TO FIND

Setting expectations honestly, so a weak result doesn't feel like failure.

- **Most cells will be statistically indistinguishable.** Overlapping confidence intervals everywhere.
- **Selection method probably matters more than entry model.** Pair quality tends to dominate.
- **The neural net probably won't beat logistic regression.** Bias-variance, as predicted.
- **Net-of-cost performance will be near zero.** Realistic breakevens in the 5–15 bps range.
- **Track A and Track B pair lists may overlap very little.** That's a finding by itself.
- **AUC around 0.52–0.58.** Anything higher means a bug.
- *(If Track C runs)* **Partial-correlation clusters will probably resemble the residual clusters** more than the characteristic clusters — both are behaviour-derived. Measuring that overlap is itself a finding.
- *(If Track D runs)* **The autoencoder probably won't beat PCA on 40 stocks.** Autoencoders are data-hungry and we have little data. Rotondi & Russo's modest 36–41 bps monthly returns are the right calibration for expectations here, not Han's Sharpe ratios. Reporting a clean "nonlinear didn't help at this scale" is the bias-variance lesson at the representation stage — a good outcome, not a failure.

**All of these are legitimate, reportable results.** The handout is explicit: *"You are not being marked on how good the results are. It doesn't matter if your method is better or worse than the ones you compare to."*

---

# PART 7 — HONEST LIMITATIONS

Everything we must disclose:

**Data:**
- Survivorship bias from using today's index membership backwards
- Only 40 stocks — thin for estimating correlation matrices
- Bloomberg fundamentals may not be point-in-time (restated financials)
- We drop sparse characteristic columns and median-impute rather than dropping stocks

**Method:**
- Simplified rolling z-score instead of Avellaneda & Lee's OU-derived s-score
- Betas drift; stale betas make a supposedly neutral spread directional
- k-means converges to local optima; cluster membership is unstable across windows
- Label parameters (2.0 trigger, 50% reversion, 5 days) are choices; results may be sensitive to them
- Estimating an 18×18 characteristic correlation matrix from 40 stocks is statistically thin
- *(If Track C runs)* Inverting a 40×40 correlation matrix from 252 observations is numerically fragile; we stabilize with diagonal shrinkage, which is itself a modelling choice
- *(If Track D runs)* The autoencoder is retrained on a coarser schedule than the daily-rolling PCA for compute reasons, so the comparison is not perfectly matched; and with this little data its capacity had to be kept very small

**Evaluation:**
- Small sample of triggers → wide confidence intervals
- Transaction cost model is stylized; real costs depend on trade size and liquidity
- Short-sale constraints and borrow costs ignored
- Multiple configurations tested → some will look good by chance

**On multiple testing specifically:** with 8 configurations (up to 16 if Part 8 runs in full) we expect roughly one to appear significant at the 5% level by chance alone — and proportionally more as the grid grows. We therefore emphasize **consistent patterns across the grid** rather than individual cell performance, and we pre-register our primary comparison (Track A × E1 vs Track A × E0) before running anything. Adding extension rows does not change the pre-registered primary comparison.

---

# PART 8 — OPTIONAL EXTENSIONS (ONLY IF AHEAD OF SCHEDULE)

**These are additions, not commitments.** The project described in Parts 0–7 is complete, gradeable, and internally consistent without anything in this part. Nothing here may be started unless its entry gate passes, and nothing here may ever displace the never-cut list: purged CV, the cost model, the turnover-matched control, and the honest limitations section.

## 8.0 The entry gates

| Extension | Gate | Decision day |
|---|---|---|
| **Track C — partial-correlation distance (Rotondi)** | The full 2×4 pipeline runs end-to-end on real data with the leakage audit and noise test passed | **Day 5 evening sync** |
| **Track D — autoencoder residuals (Krause)** | Track C (or a deliberate decision to skip it) is resolved, results-freeze on Day 7 is still realistic, AND Person C's E-models are done | **Day 6 morning sync** |

If a gate fails, the extension is **cited in the literature review as related work we scoped but did not run**, with one sentence each. That is an honest and complete treatment — no apology needed in the report.

Priority order if only one can run: **Track C first.** It is roughly one-third the cost of Track D and adds a full grid row.

## 8.1 Track C — Partial-correlation distance clustering (Rotondi & Russo, 2024)

**Effort: ~5 hours. New data required: none. Owner: Person A.**

### The idea

Ordinary correlation between two stocks can be inflated by everything they share with the rest of the universe — if both load on the market and the same sector, they look connected even if there is no direct link between them. **Partial correlation** asks a sharper question: how do these two stocks co-move *after controlling for all the other stocks*? Rotondi & Russo (2024) use a distance built on this as a novel input to k-means for pair selection. Conceptually it is the accessible cousin of Zhang's graphical lasso, and it is philosophically aligned with our whole project: one more way of stripping out common influences before asking who really belongs with whom.

### The mechanics

1. Take the same in-window 40×40 correlation matrix `C` already computed for Track A's PCA (Step 2c). No new estimation.
2. **Stabilize before inverting.** A 40×40 matrix from 252 observations can be near-singular. Apply diagonal shrinkage first:

```python
lam = 1e-3                                  # small ridge; report the value used
C_shrunk = (1 - lam) * C + lam * np.eye(40)
P = np.linalg.inv(C_shrunk)                 # precision matrix
```

3. Convert precision to partial correlation:

```python
d = np.sqrt(np.diag(P))
partial_corr = -P / np.outer(d, d)
np.fill_diagonal(partial_corr, 1.0)
```

4. Build the distance and cluster with the **identical** k-means machinery from Step 4 (same k-range, same k-means++, same restarts, same co-membership tracking, same stability analysis):

```python
dist = np.sqrt(2 * (1 - partial_corr))     # standard correlation-to-distance map
```

5. Emit pairs through the **same** Step 5 rules, tagged `source = "track_c"`.

Everything downstream — spreads, labels, features, models, evaluation — is untouched. That is the entire point: one variable changes (the distance metric), so any performance difference is attributable to it.

### What it buys the report

- A third grid row for ~5 hours, strengthening the column-effect claim (Part 5, point 1)
- A clean single-variable comparison: Euclidean-on-residuals vs partial-correlation distance, everything else fixed — exactly the isolation Rotondi & Russo designed for
- An overlap measurement (Track C vs Track A pair lists) that extends the consensus-pairs analysis for free

### Honest cautions

- The shrinkage parameter is a choice; report it and note results were not tuned over it
- Rotondi & Russo's own returns (36–41 bps monthly) are modest — expect the same order, not Han-style Sharpe ratios
- The precision matrix must be estimated **in-window only** — leakage checklist item 11

### ⭐ In plain words

Two kids might look like friends just because they're both friends with the same popular kid. Partial correlation checks whether they'd still hang out if the popular kid moved away. We group kids by *that*, and see if it picks better pairs.

## 8.2 Track D — Autoencoder residuals (Krause & Calliess, 2024)

**Effort: ~12–15 hours. New data required: none. Owner: Person C (PyTorch), integrated by Person A.**

### The idea

Track A assumes the market's common structure is **linear** — that's what PCA extracts. Krause & Calliess (2024) replace the linear extraction with an **autoencoder**: a small neural network that compresses the 40 daily returns down to a few numbers and reconstructs them. Whatever the reconstruction misses is, by construction, the part common structure cannot explain — a **nonlinear residual**, playing exactly the role our PCA residual plays.

We implement **only** this first idea of theirs. Their end-to-end policy learning is reinforcement-learning-flavoured, outside course scope, and explicitly skipped.

### The mechanics

1. **Architecture, kept deliberately tiny:** input 40 → bottleneck 3–5 (matched to the PCA component count, so the comparison is capacity-fair) → output 40. One hidden layer at most on each side. MSE loss, early stopping on a held-out slice of the training window, weight decay.
2. **Training schedule:** retraining a network daily on rolling windows is computationally silly at our scale. Retrain **monthly** on the trailing 252 days and apply forward until the next retrain. Document that this makes the PCA-vs-AE comparison imperfectly matched (PCA re-estimates daily) — it goes in limitations.
3. **Standardize inputs window-locally** — same leakage trap as Step 2b, same fix.
4. **Residual:** `residual_t = return_t - reconstruction_t`, computed strictly out-of-sample (the AE never trains on the day it scores).
5. Feed the residual matrix into the **same** Step 4 clustering and everything downstream, tagged `source = "track_d"`.

### The built-in sanity check (do this FIRST)

Baldi & Hornik (1989) proved that a **linear** autoencoder recovers the PCA subspace. So before training the nonlinear version: strip the activations, train the linear AE, and verify its reconstructions and subspace match our Step 2 PCA (compare reconstruction errors; they should be near-identical). **If they don't match, we have a bug — in the AE or in our PCA.** This is a self-validating test that connects the two tracks, and it's a strong rigor point for the report whether or not the nonlinear results are interesting.

### What it buys the report

- The cleanest experiment in the whole project: **linear vs nonlinear factor extraction at the identical pipeline position**, mirroring at the representation stage what E1-vs-E3 tests at the decision stage
- Course-content alignment: autoencoders are in our toolkit, and this uses them on real data with a falsifiable expectation
- A fourth grid row, making the column-effect claim as strong as it can get

### Honest expectation

**It probably will not beat PCA.** Forty stocks and ~2,500 days is very little data for a neural representation, and the honest prior is that the nonlinearity buys variance, not signal. A clean null result here is the bias-variance lesson demonstrated twice in one project — once at the model stage (E3), once at the representation stage (Track D). That symmetry is worth a paragraph in the report on its own.

### ⭐ In plain words

PCA finds the fire alarm by drawing straight lines. The autoencoder is allowed to draw curvy lines. Curvy lines *can* capture more — but they also imagine patterns that aren't there, especially when you've only watched 40 kids. We test whether curvy beats straight here. We suspect it won't, and finding that out properly is the point.

## 8.3 How the grid and analyses absorb the extensions

**Grid:** 2×4 core → 3×4 with Track C → 4×4 with both. The pre-registered primary comparison (**Track A × E1 vs Track A × E0**) never changes, no matter how many rows exist. Everything beyond it is labeled secondary.

**Multiple testing:** each added row adds 4 cells. We restate the expected-false-positive arithmetic in the report for whichever grid size actually ran, and continue to emphasize cross-grid patterns over individual cells.

**Consensus analysis:** with 3–4 pair lists, add one figure — for each pair, how many methods selected it (1, 2, 3, 4) — and compare reversion rate by selection count. The hypothesis extends naturally: the more independent paradigms agree a pair is real, the more real it should be.

**Report space:** extensions get at most one paragraph plus their rows in the main table. If writing space runs short on Day 8, extension detail moves to the appendix before anything from Parts 0–7 does.

## 8.4 If we skip them

One sentence each in the literature review, along these lines: "Alternative selection paradigms include partial-correlation clustering (Rotondi & Russo, 2024) and autoencoder-based factor extraction (Krause & Calliess, 2024); we scoped both as extensions of our factorial design but prioritized depth of evaluation over grid width within the available time." That is honest, shows awareness of the field, and turns the omission into evidence of deliberate scoping rather than ignorance.

---

# PART 9 — EXPLAIN IT LIKE I'M FIVE

*This is the version for a friend, a parent, or a non-technical interviewer.*

## The story

Imagine two kids, Sam and Alex. They're best friends and they always walk to school together, side by side, every single day.

One morning you notice something odd. Sam is way out in front and Alex is trailing far behind.

That's weird. They're always together.

So you make a guess: **"I bet they'll be walking together again by tomorrow."** Because that's what they always do.

Sometimes you'd be right — Alex just stopped to tie a shoe. Sometimes you'd be wrong — they actually had a fight and won't walk together again.

## What we built

Instead of kids, it's **companies**. Some companies normally move together because they're similar — two soft drink makers, two airlines, two banks.

When one drifts away from its partner, you can bet they'll come back together. If you're right, you make money.

**The old way of deciding when to bet:** "They're far apart → bet." Simple. Always the same.

**Our way:** "They're far apart → but hold on. Let me look at the last thousand times this happened. What was different about the ones that came back together? Does *this* one look like those?"

## The three tricky parts

**Part one: everyone moves at once.** When the school bell rings, *every* kid runs. That doesn't tell you anything about Sam and Alex specifically. So first we figure out how much of the movement was "the bell" and subtract it. What's left is each kid's own personal behaviour. *(That's PCA and residuals.)*

**Part two: who's actually friends?** We don't compare every kid to every other kid — with 40 kids that's 780 comparisons, and you'd find fake patterns by accident. So we group kids who act alike first, and only compare within groups. *(That's clustering.)* We do this two different ways: by how they behave, and by what kind of kid they are. And if we finish early, we have two spare ways on the shelf — one that checks who'd *still* be friends without the popular kid in the middle, and one that lets a small robot look for friendships that aren't straight lines.

**Part three: are we fooling ourselves?** This is the part we care about most. We built our system so it can never peek at the future. And we made ourselves a rule: **if the results look amazing, assume something's broken.** Because in this field, amazing results almost always mean a mistake.

## The clever question at the end

Here's the thing we're proudest of.

Suppose our smart method makes more money than the dumb method. Great! But *why*?

Two possible reasons:

1. Our method is actually smarter — it picks better bets, **or**
2. Our method just bets *less often*, and since every bet costs a fee, betting less means paying fewer fees

Those look **exactly the same** in the results. Everyone who's published on this just assumed it was reason one.

So we built a fake method that bets exactly as rarely as ours, but picks **completely at random**. If we beat the random-but-equally-lazy method, we're genuinely smarter. If we don't — we were just being lazy, and lazy is cheap.

**Nobody had checked this.** That's our contribution.

## Why we did it

Two reasons.

**The honest one:** it's a course project, and we wanted to do something real rather than something that just looks impressive.

**The real one:** in this field, it's *very* easy to build something that looks like it makes money and actually doesn't. The whole skill is knowing the difference. We wanted to practice being the kind of people who check.

## If you remember one sentence

> **Everyone else bets on every gap. We tried to learn which gaps are worth betting on — and then we checked, honestly, whether we'd actually learned anything at all.**

---

*End of specification (v2).*
