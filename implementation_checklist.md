# Implementation Checklist: Earnings Call Sentiment Analyzer

Tick each box as you finish it. Each phase ends with a "Done when" check. Do not start the next phase until that check passes.

---

## Phase 0: Setup (Day 1)

- [ ] Create the project folder and Git repository (see folder structure in `plan_proj1.md`)
- [ ] Create a virtual environment: `python -m venv .venv` and activate it
- [ ] Install core packages:
  ```bash
  pip install pandas numpy pyarrow matplotlib seaborn scipy statsmodels
  pip install transformers datasets torch yfinance jupyter
  ```
- [ ] Add a `.gitignore` that excludes `data/raw/`, `data/processed/`, `.venv/` and model caches
- [ ] Create `README.md` with the problem statement (copy from the plan)
- [ ] Check that a small model loads, e.g. `pipeline("sentiment-analysis", model="ProsusAI/finbert")`

**Done when:** the environment runs, FinBERT returns a label for a test sentence, and the repo has its first commit.

---

## Phase 1: Data Sourcing (Week 1)

- [ ] Download a sample transcript dataset from Hugging Face (start with `soumakchak/earnings_call_transcript_lite` or `awinml/earnings-calls-qa` for testing)
- [ ] Read the dataset card and record the licence, fields and date coverage in `data/README.md`
- [ ] Decide the main dataset (recommended: `kurry/sp500_earnings_transcripts`) and confirm its licence allows your use
- [ ] Inspect 5 transcripts manually: check the format, the speaker labels, and where Q&A begins
- [ ] Collect daily adjusted prices for all tickers in your sample
- [ ] Collect SPY (or a sector ETF) daily prices for the market return
- [ ] Collect call timing (before open / after close). Record it as a column, even if it is an estimate
- [ ] Collect EPS actual and consensus for each call (needed as a control)
- [ ] Save all raw downloads to `data/raw/` with the download date in the filename

**Done when:** you have raw transcripts, prices, market data and EPS data for an agreed date range, and you have written down any gaps.

---

## Phase 2: Cleaning and Parsing (Week 1)

- [ ] Write `src/clean.py` to load transcripts into a DataFrame with: `ticker`, `company`, `call_date`, `call_timing`, `text`
- [ ] Remove operator lines, safe-harbour / forward-looking disclaimers and page artefacts
- [ ] Split each transcript into `prepared_text` and `qa_text` at the start of the Q&A section
- [ ] Count words per call and flag calls with fewer than 500 words (likely parse errors)
- [ ] Drop duplicate calls (same ticker and date)
- [ ] Join with price and EPS data on ticker and date; record how many calls are lost at each step
- [ ] Save `data/processed/calls_clean.parquet`

**Done when:** a call-level table exists with one row per call, no duplicates, and a log showing how many rows were dropped and why.

---

## Phase 3: Event Study (Week 2)

- [ ] Write `src/event_study.py`
- [ ] Define the event day: day 0 for after-close calls is the next trading day; for before-open calls it is the same day
- [ ] For each stock, estimate the market model over days [−250, −30] relative to the event (need at least 120 valid days)
- [ ] Compute daily abnormal returns: AR = R_stock − (α + β · R_market)
- [ ] Compute CAR for the windows [0, +1], [0, +2] and [−1, +1]
- [ ] Check the distribution of CARs for outliers and data errors (plot a histogram)
- [ ] Save `data/processed/car.parquet`

**Done when:** every call has CAR values, you can explain how day 0 was set for a sample of calls, and the distribution looks reasonable.

---

## Phase 4: Sentiment Scoring (Week 2)

### 4a. Dictionary baseline
- [ ] Download the Loughran-McDonald master dictionary (positive and negative word lists)
- [ ] Write `src/sentiment_lm.py`: tokenise, lowercase, count positive and negative words
- [ ] Compute `lm_score = (pos − neg) / total_words` for prepared text, Q&A text and full text
- [ ] Save the scores to `data/processed/scores_lm.parquet`

### 4b. FinBERT scoring
- [ ] Write `src/sentiment_finbert.py`
- [ ] Split text into sentences (use `nltk` or `spacy`, or a regex fallback)
- [ ] Group sentences into chunks of at most 510 tokens, never breaking a sentence
- [ ] Run each chunk through FinBERT, store P(positive), P(neutral), P(negative)
- [ ] Aggregate to call level: `finbert_score = mean(P_pos − P_neg)` and `share_negative = mean(label == negative)`
- [ ] Use a GPU if available (`torch.cuda.is_available()`); batch the inputs; save progress every 500 calls so a crash does not lose work
- [ ] Save `data/processed/scores_finbert.parquet`

**Done when:** both scores exist for every call and a spot check of 10 calls looks sensible (positive-sounding calls score higher).

---

## Phase 5: Label Validation (Week 2)

- [ ] Randomly sample 200 sentences (stratified across companies and years) into `data/labels/sample_to_label.csv`
- [ ] Label each sentence yourself as positive, neutral or negative; keep the labels in a separate file
- [ ] Run FinBERT and the dictionary on the same sentences
- [ ] Compute accuracy, macro F1 and a confusion matrix for each method
- [ ] Note the most common errors (e.g. hedged guidance labelled as negative)
- [ ] Record results in `reports/validation.md`

**Done when:** you have a table comparing both methods against your labels, and you know which one to use for the main analysis.

---

## Phase 6: Exploratory Analysis (Week 3)

- [ ] Plot the distribution of each sentiment score
- [ ] Check the correlation between `lm_score` and `finbert_score` (expect a moderate positive value)
- [ ] Check the class balance if you use categories (watch for a small neutral class)
- [ ] Plot the mean CAR by sentiment quintile
- [ ] Check whether Prepared Remarks and Q&A sentiment differ in their relationship with CAR
- [ ] Save charts to `reports/figures/`

**Done when:** you have at least 4 charts and a short list of observations written in `reports/notes.md`.

---

## Phase 7: Controlled Regression (Week 3)

- [ ] Write `src/regression.py`
- [ ] Standardise sentiment scores (z-score)
- [ ] Run the baseline regression: CAR ~ sentiment only
- [ ] Run the controlled regression: CAR ~ sentiment + EPS_surprise + log(market_cap) + prior_return + sector fixed effects
- [ ] Use robust standard errors clustered by company (and try clustering by date)
- [ ] Run the same models for [0,+1], [0,+2] and each sentiment measure (LM, FinBERT, Prepared, Q&A)
- [ ] Run a rank correlation (Spearman) as a robustness check
- [ ] Record the coefficient, t-stat, R² and number of observations in `reports/regression_results.md`
- [ ] Check residual plots for obvious problems

**Done when:** you can state clearly whether sentiment is associated with CAR after controls, with the size and significance of the effect.

---

## Phase 8: Backtest (Week 4)

- [ ] Write `src/backtest.py`
- [ ] Define the portfolio rule (e.g. go long top quintile, short bottom quintile by sentiment, rebalanced at each earnings season)
- [ ] Use only information available before the trade (entry at the close before the reaction, or the open after the call, consistent with your timing)
- [ ] Exit after a fixed holding period (1 or 2 days)
- [ ] Add transaction costs (start at 10 bps per side; test 5, 20 and 50 bps)
- [ ] Compute: annualised return, volatility, Sharpe ratio, hit rate, max drawdown, average turnover
- [ ] Compare against a benchmark (SPY buy-and-hold, equal-weight universe)
- [ ] Plot the cumulative return curve

**Done when:** a backtest exists with a clear cost assumption and a comparison to the benchmark.

---

## Phase 9: Out-of-Sample Testing (Week 4)

- [ ] Split the data by time: development period (e.g. 2015 to 2020) and test period (e.g. 2021 to 2025)
- [ ] Choose the model specification, window and thresholds using only the development period
- [ ] Write down the chosen specification in `reports/spec_frozen.md` **before** looking at the test period
- [ ] Run the frozen specification on the test period once
- [ ] Report both development and test results, even if the test result is weaker
- [ ] (Optional) Walk-forward: retrain each year on the past and test on the next year

**Done when:** a frozen specification has been tested once on held-out data and the result is reported honestly.

---

## Phase 10: Robustness and Sanity Checks (Week 4)

- [ ] Re-run the main result excluding the top and bottom 1% of CARs (winsorising)
- [ ] Re-run excluding the 2020 market shock period, and separately excluding 2008 if included
- [ ] Re-run for large caps only and for small/mid caps only
- [ ] Check for look-ahead: confirm that no feature uses data released after the call
- [ ] Check for duplicate companies or tickers that changed (e.g. mergers, renames)
- [ ] Check that results hold with a different sentiment threshold or model

**Done when:** you have a short table showing which results hold under each check and which do not.

---

## Phase 11: Report and Presentation (Week 5)

- [ ] Write `reports/final_report.md` with sections: Problem, Data, Method, Results, Limitations, Next steps
- [ ] Include the validation table, regression table, backtest chart and at least one sentiment chart
- [ ] Write a limitations section covering: data coverage, label noise, transaction costs, survivorship, and the fact that results may not persist
- [ ] Update `README.md` with setup instructions, how to run each script, and the main findings
- [ ] Check that all scripts run from a clean environment (`pip install -r requirements.txt`, then run the pipeline)
- [ ] Freeze `requirements.txt`: `pip freeze > requirements.txt`
- [ ] Tag the release in Git: `git tag v1.0`

**Done when:** a new person can clone the repo, follow the README, and reproduce the main results.

---

## Phase 12: Optional App (Stretch Goal)

- [ ] Build a simple Streamlit app: user pastes a transcript, the app shows sentence-level sentiment with colour highlights and the call-level score
- [ ] Add a chart showing the score split between Prepared Remarks and Q&A
- [ ] Add a disclaimer on the page that the output is not investment advice
- [ ] Deploy to Streamlit Community Cloud or run locally

---

## Quality Checklist (apply at the end)

- [ ] No look-ahead bias in any feature or signal
- [ ] Time-based train / test split used throughout
- [ ] Results reported with confidence intervals or t-stats, not just point estimates
- [ ] Transaction costs included in all backtest numbers
- [ ] Dataset licences checked and recorded
- [ ] Random seeds set where randomness is used
- [ ] Code has docstrings and is split into the modules listed in the plan
- [ ] Negative or null results are reported, not hidden

---

## Common Pitfalls (keep this list visible)

1. **Same-day reaction for after-close calls.** The price reaction is the next day.
2. **Using the full dataset to choose the model**, then testing on the same data. This inflates results.
3. **Ignoring the EPS surprise.** Sentiment may only be proxying for the earnings beat.
4. **Accuracy as the main metric on imbalanced classes.** Use macro F1.
5. **Truncating long transcripts** to 512 tokens and losing most of the call.
6. **Forgetting costs.** Small trades look profitable until costs are included.
7. **Not saving intermediate results.** Save after each long step.

---

## Weekly Progress Log

| Week | Phase(s) | Status | Notes |
|---|---|---|---|
| 1 | 0, 1, 2 | | |
| 2 | 3, 4, 5 | | |
| 3 | 6, 7 | | |
| 4 | 8, 9, 10 | | |
| 5 | 11, 12 | | |
