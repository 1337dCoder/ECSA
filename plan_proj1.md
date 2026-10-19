# Project 1: Earnings Call Sentiment Analyzer

## 1. Problem Statement

Company executives speak on earnings calls about results, guidance and outlook, and much of the real information sits in their word choice: hedging ("we expect, subject to..."), confidence, or defensiveness in Q&A. Investors read these calls, but reading hundreds of transcripts per quarter is not practical, and most of the tone is not captured by simple numbers like EPS or revenue.

**Core question:** Can an NLP model that scores the sentiment of earnings call transcripts predict the stock's abnormal price reaction after the call, and does that signal add value beyond the reported numbers?

**Sub-questions:**
1. Can a pretrained financial language model turn transcript text into a reliable sentiment score (positive / neutral / negative)?
2. Does that score correlate with the abnormal return in the 1 to 2 trading days after the call (an event study)?
3. Does the score still carry information after controlling for EPS surprise, the market move and company characteristics?
4. Can the signal be turned into a trading rule that survives realistic assumptions (costs, no look-ahead)?

**Success criteria (define before building):**
- Sentiment model: macro F1 clearly above a majority-class baseline on a labelled hold-out set.
- Signal: statistically meaningful relationship between sentiment and next-day CAR (t-stat above about 2 after Newey-West correction), with a sign that makes economic sense.
- Backtest: positive net-of-cost Sharpe on out-of-sample data, or an honest write-up of why it fails.

**Non-goals:** This is not a live trading system, not investment advice, and not a claim that the model will make money.

---

## 2. Research Review

### 2.1 What the reference project shows

The reference image shows a three-class (negative / neutral / positive) setup using a Longformer model. Its label counts are roughly 590 negative, 250 neutral and 720 positive, so the data is imbalanced, with neutral the smallest class. The classification report shows the model is weak on class 1: precision and recall are 0 for that class, and accuracy is 0.45. Two lessons follow:

- **Class imbalance must be handled** (class weights, resampling, or evaluating with macro F1 rather than accuracy).
- **A long-context model is not automatically better.** Earnings calls often run 5,000 to 10,000+ words, so the transcript has to be split into chunks and aggregated. The aggregation method matters as much as the model.

### 2.2 Models

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **FinBERT** (ProsusAI/finbert) | BERT fine-tuned on financial news for 3-class sentiment | Domain-tuned, fast, widely cited baseline | 512-token limit, needs chunking |
| **Longformer / BigBird** | Transformer with long-context attention | Reads longer spans at once | Heavier, needs more labelled data |
| **Loughran-McDonald dictionary** | Finance-specific word lists (negative, positive, uncertainty, litigious) | Transparent, a strong academic baseline, no training | Misses context, e.g. "not bad" |
| **General LLM zero-shot scoring** | Prompted model returns a score | No labels needed | Expensive, hard to reproduce, prompt-sensitive |

**Recommendation:** Build with three levels and compare them: (1) Loughran-McDonald word-count baseline, (2) FinBERT chunk-level scoring, (3) optional fine-tuned Longformer or FinBERT on your own labels. The baseline is required because it is the standard the literature compares against.

### 2.3 Finance literature to know

- **Loughran & McDonald (2011):** showed that general-purpose sentiment word lists misclassify many financial words (e.g. "liability", "tax" are often negative in general lists but not in finance). Their finance-specific list is the standard baseline.
- **Price, Doran, Peterson & Bliss (2012):** found that the tone of earnings call transcripts is associated with abnormal returns and trading volume around the call. This is the main theoretical support for the project.
- **Event-study methodology (MacKinlay, 1997):** the standard way to measure abnormal returns around an announcement. You estimate the normal return with a market model over a window before the event, then sum the abnormal returns over the event window (CAR).
- **Q&A vs prepared remarks:** Q&A is less scripted, so many studies treat it as more informative. Test both and do not assume the answer.

Check the original papers yourself before citing them in a report. This plan summarises the general findings and does not quote them.

### 2.4 Data sources reviewed

| Source | What it contains | Notes |
|---|---|---|
| **kurry/sp500_earnings_transcripts** (Hugging Face) | About 33,000 transcripts for S&P 500 companies, 2005 to 2025, with year and date fields | Largest option found. Good for the main study. Check its licence and the exact fields before use. |
| **awinml/earnings-calls-qa** (Hugging Face Space) | Plain-text transcripts per ticker (e.g. AAPL, MSFT, MU, CSCO), files named by date | Easy to read and parse. Useful for a small prototype or for checking the parser. |
| **soumakchak/earnings_call_transcript_lite** (Hugging Face) | Small CSV sample, roughly 1K to 10K rows | Good for a quick first test of the pipeline. |
| **Revai/earnings21** (Hugging Face) | About 39 hours of earnings call audio with formatted text, built for speech recognition research | Audio data, not an ideal fit for pure text sentiment. Useful only if you want to add speech-to-text. Check the licence. |
| **Price data** | Daily adjusted close and SPY (or sector ETF) for the market return | Free options: yfinance (for learning), or a free-tier API. Use adjusted prices and confirm the source's terms. |
| **Earnings date and EPS surprise** | Call date/time, consensus EPS, actual EPS | Free options are limited. Possible sources: company IR pages, SEC filings, or a financial data API. Make sure the consensus data is point-in-time (what was known at the time). |

**Data check before starting:** For each source, verify (a) licence allows your use, (b) the call timestamp is accurate and includes whether the call was before market open or after close, (c) the ticker and date fields join cleanly with price data.

### 2.5 Key methodological risks

1. **Look-ahead bias:** Using information that was not available at the time of the call. Example: using the final quarterly revenue number that was reported later. Use only what is public at the call time.
2. **Timing errors:** An after-close call's reaction happens the next trading day. If you measure the reaction on the same day, you measure nothing. Define day 0 carefully for before-open and after-close calls.
3. **Survivorship bias:** Only using companies that are still listed today. Use a point-in-time index membership list if possible.
4. **Multiple testing / overfitting:** Trying many sentiment models, windows and thresholds until one looks good. Pre-register your main specification and report the rest as robustness checks.
5. **Confounding:** Sentiment may just reflect the EPS beat or miss. Always test sentiment controlling for the EPS surprise and the market return.
6. **Transaction costs and liquidity:** Small-cap stocks can look profitable before costs. Include costs and filter by liquidity.
7. **Label noise:** Sentiment labels from a model are not the truth. Hand-label a small sample (e.g. 200 sentences) to measure agreement.

---

## 3. Proposed Method

### 3.1 Pipeline overview

```
Transcripts  ->  Clean & split (prepared remarks / Q&A)  ->  Sentence / chunk sentiment
                                                                  |
Prices & market  ->  Event study (CAR)  <--  Aggregate to call-level score
                          |
                  Regression with controls  ->  Trading backtest  ->  Report
```

### 3.2 Step-by-step

**Step A. Data collection and cleaning**
- Load transcripts with ticker, company, call date, and time-of-day (if not available, assume after close and test both).
- Remove operator text, safe-harbor disclaimers and speaker labels where they add noise.
- Split each transcript into Prepared Remarks and Q&A (detect the start of Q&A from phrases such as "we'll now open the line for questions").
- Save cleaned data in Parquet or CSV with one row per call.

**Step B. Sentiment scoring**
- Baseline: Loughran-McDonald score = (positive words − negative words) / total words.
- FinBERT: split text into sentences, run each through the model, get P(positive), P(neutral), P(negative). Aggregate to call level by mean of (P_pos − P_neg) and by share of negative sentences.
- Handle long text by chunking at sentence boundaries with a maximum of 510 tokens per chunk.
- Save the per-sentence and per-call scores separately.

**Step C. Label validation (important)**
- Randomly sample about 200 sentences. Label them yourself (positive / neutral / negative). Compare with FinBERT and the dictionary. Report accuracy, macro F1 and a confusion matrix.
- Use this hand-labelled set as your true hold-out test for the classifier.

**Step D. Event study**
- For each call, define the event day (day 0) as the first trading day whose price reaction includes the call.
- Estimate a market model for each stock over a window of 250 days ending 30 days before the event: R_it = α + β · R_mt + ε.
- Abnormal return AR = R_it − (α + β · R_mt).
- CAR over window [0, +1] (and as a robustness check [0, +2] and [−1, +1]).

**Step E. Regression**
- CAR_i = a + b · Sentiment_i + c · EPS_surprise_i + d · log(MarketCap_i) + e · Prior_return_i + sector fixed effects + ε_i
- Standardise sentiment (z-score) so coefficients are comparable.
- Use heteroskedasticity-robust standard errors, clustered by company or by date.
- Also run a rank-based version (Spearman correlation between sentiment and CAR) for robustness.

**Step F. Trading signal and backtest**
- Each quarter (or at each call), rank stocks by sentiment. Go long the top decile (or quintile) and short the bottom, or long-only as a second version.
- Enter at the close before the reaction (or the open after the call, depending on timing) and exit after 1 to 2 days. Do not trade on information you did not have.
- Apply transaction costs (e.g. 10 to 20 bps per side as a starting assumption, and test sensitivity).
- Report annualised return, volatility, Sharpe ratio, hit rate, maximum drawdown and turnover.

**Step G. Out-of-sample testing**
- Split by time, not randomly. Example: train or tune on 2015 to 2020, test on 2021 to 2025.
- Use walk-forward evaluation if you have enough data.
- Freeze the main model before looking at the test period. Report the test result even if it is weak.

**Step H. Reporting**
- Write up the problem, data, method, results, limitations, and what you would do next.
- Include charts: sentiment distribution, confusion matrix, CAR by sentiment quintile, cumulative backtest return vs benchmark.

### 3.3 Expected deliverables

1. Clean dataset (CSV or Parquet) with transcripts, sentiment scores and CAR per call.
2. Notebook or Python scripts for each step (cleaning, scoring, event study, regression, backtest).
3. Validation report with confusion matrix and hand-label agreement.
4. Results summary with tables and charts.
5. A short written report (2 to 4 pages) with clear limitations.
6. (Optional) A small Streamlit or Gradio app where a user pastes a transcript and gets a sentiment breakdown.

### 3.4 Suggested folder structure

```
earnings-sentiment/
├── data/
│   ├── raw/                 # downloaded transcripts, prices
│   ├── processed/           # cleaned calls, scores
│   └── labels/              # hand-labelled validation sentences
├── src/
│   ├── clean.py             # parse and split transcripts
│   ├── sentiment_lm.py      # Loughran-McDonald baseline
│   ├── sentiment_finbert.py # FinBERT scoring with chunking
│   ├── event_study.py       # market model and CAR
│   ├── regression.py        # controlled regressions
│   └── backtest.py          # trading rule and costs
├── notebooks/
│   └── exploration.ipynb
├── reports/
│   ├── figures/
│   └── final_report.md
├── tests/
├── requirements.txt
└── README.md
```

---

## 4. Tech Stack

- **Language:** Python 3.10 or newer
- **Data:** pandas, numpy, pyarrow (for Parquet)
- **NLP:** Hugging Face `transformers` and `datasets`, `torch`, Loughran-McDonald word lists (download from the authors' site or use a maintained copy)
- **Finance/stats:** `yfinance` (learning use), `statsmodels` (regression, Newey-West), `scipy`
- **Visualisation:** matplotlib, seaborn
- **Optional app:** Streamlit or Gradio
- **Version control:** Git, with data folders in `.gitignore`

---

## 5. Timeline (suggested, about 4 to 5 weeks part-time)

| Week | Focus | Output |
|---|---|---|
| 1 | Data sourcing, cleaning, Prepared vs Q&A split | Clean call-level dataset |
| 2 | Sentiment scoring (dictionary + FinBERT), hand-labelling | Scores and validation report |
| 3 | Event study and controlled regressions | CAR results and tables |
| 4 | Backtest, out-of-sample testing, robustness checks | Backtest results |
| 5 | Write-up, charts, optional app, README | Final report and repo |

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Data is incomplete or missing call times | Use a smaller, cleaner subset; document exclusions |
| FinBERT scores are noisy on long text | Chunk carefully; compare with dictionary baseline; hand-label check |
| Result disappears after controlling for EPS surprise | Report it honestly. A null result is a valid finding. |
| Backtest looks great but is unrealistic | Enforce point-in-time data, add costs, test out-of-sample |
| Scope creep (audio, LLMs, live trading) | Keep audio and live trading as "future work" |

---

## 7. Future Extensions

- Add speech-based features (tone, pauses) using the Earnings21 audio data.
- Compare analyst reaction with management tone by linking to analyst notes.
- Track sentiment changes quarter over quarter (the change may matter more than the level).
- Build a sector-specific model.

---

## 8. Disclaimer

This is a learning project. The results are for research and education only and are not investment advice. Check the licence of every dataset and model before using them in any public or commercial work.
