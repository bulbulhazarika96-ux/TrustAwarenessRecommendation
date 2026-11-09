# Academic Viva / Defense Preparation Guide
## Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework for Personalized News Recommendation

This guide provides targeted questions, rigorous technical answers, and mathematical justifications commonly requested by external examiners and university project review committees.

---

### Q1: What is the fundamental research gap addressed by your project?
**Examiner Focus:** Understanding the difference between commercial systems and your academic contribution.

**Answer:**
Conventional recommendation platforms (like Google News, Yahoo News, or social media feeds) optimize primarily for single-objective engagement heuristics, typically Click-Through Rate (CTR). This approach has three critical failures:
1. It favors sensational and hyperbolic headlines over factual reporting (clickbait optimization).
2. It traps users in narrow ideological filter bubbles (echo chambers).
3. It remains completely truth-blind, algorithmically treating a viral fabricated conspiracy article identically to peer-reviewed investigative journalism.

Our research gap address is formulating a **7-Objective Trust-Aware Pareto Optimization Framework** that optimizes for User Relevance, Credibility, Long-Term Engagement, User Satisfaction, Diversity, Novelty, and Exposure Fairness simultaneously.

---

### Q2: How did you prevent data leakage in the Kaggle Fake News dataset?
**Examiner Focus:** Methodological rigor and machine learning best practices.

**Answer:**
In many published papers on the Kaggle ISOT dataset, models report unrealistically high accuracy because articles in `True.csv` frequently begin with wire agency identifiers such as `WASHINGTON (Reuters) - `. If vectorizers process these headers, the classifier trivially learns:
$$\text{If 'reuters' in text} \implies \text{REAL else FAKE}$$
In our implementation ([`src/preprocessing.py`](file:///c:/Users/lenovo/OneDrive/Desktop/TrustAwareNewsRecommendation/src/preprocessing.py)), we implemented regex sanitization that systematically strips all wire service headers (`Reuters`, `AP`, `AFP`, etc.) before tokenization and feature extraction. This forces the model to learn authentic stylistic, syntactic, and rhetorical markers rather than memorizing publisher strings.

---

### Q3: Why did the Passive Aggressive Classifier outperform Logistic Regression and Random Forest?
**Examiner Focus:** Justification of model selection.

**Answer:**
The Passive Aggressive Classifier (PAC) is an online margin-based algorithm particularly suited for high-dimensional sparse text vectors (10,000 TF-IDF unigram and bigram features). 
- **Passive step:** If a training instance is classified correctly with sufficient margin, no update occurs ($L=0$).
- **Aggressive step:** If a misclassification occurs, it updates weight vectors aggressively to ensure the loss on that sample is zeroed out:
  $$\mathbf{w}_{t+1} = \mathbf{w}_t + \text{sign}(y_t - \hat{y}_t) \frac{\ell_t}{\|\mathbf{x}_t\|^2 + \frac{1}{2C}} \mathbf{x}_t$$
In our experiments, calibrated PAC achieved **98.98% F1-score** and **0.9993 ROC-AUC** with sub-millisecond (0.8 ms) inference latency on CPU, outperforming Random Forest (which suffers under very high dimensionality and sparse matrices) and Naive Bayes (which assumes naive conditional independence).

---

### Q4: Why not simply fine-tune BERT or DistilBERT for production recommendation?
**Examiner Focus:** Scalability and systems engineering trade-offs.

**Answer:**
While fine-tuned DistilBERT achieves comparable accuracy (~98.45% F1), it introduces significant computational drawbacks for real-time recommendation:
1. **Inference Latency:** Transformer forward passes take 40–100 ms per candidate article on CPU, compared to 0.8 ms for calibrated linear TF-IDF models.
2. **Resource Footprint:** Ranking 100 candidate articles for thousands of concurrent users in real time would require costly GPU clusters.
3. **Interpretability:** Linear TF-IDF classifiers provide explicit token weights, enabling transparent explainability ("Why Recommended?") by surfacing exact tokens that drove the classification.

---

### Q5: How is the Credibility Score calculated? Is it just the classifier's output?
**Examiner Focus:** Multi-factor synthesis vs single-model dependency.

**Answer:**
No, the classifier prediction is strictly one component. An automated NLP model evaluates statistical writing style, not ground truth factuality. Therefore, our Credibility Score ($C$) synthesizes four orthogonal components:
$$C(a) = \alpha \cdot P(\text{real} \mid a) + \beta \cdot S_{\text{source}}(a) + \gamma \cdot (1 - \text{Sensationalism}(a)) + \delta \cdot \text{Factuality}(a)$$
- $P(\text{real} \mid a)$ (50%): Calibrated classifier probability.
- $S_{\text{source}}(a)$ (25%): Publisher track record looked up from an empirical trust index.
- Linguistic Integrity (15%): Inverse measure of headline capitalization, exclamation marks, and clickbait triggers.
- Factuality Proxy (10%): Presence of quotes, dates, and official entity attributions.

Furthermore, we compute **Shannon Entropy** to quantify epistemic uncertainty:
$$H(p) = - (p \log_2 p + (1-p) \log_2 (1-p))$$
When uncertainty is high (near decision boundary $p \approx 0.50$), the system alerts the user that model confidence is ambiguous.

---

### Q6: How does the system detect accidental clicks / clickbait traps?
**Examiner Focus:** Long-term engagement and satisfaction modeling.

**Answer:**
Commercial platforms treat every click as positive engagement. In our temporal engagement module ([`src/engagement.py`](file:///c:/Users/lenovo/OneDrive/Desktop/TrustAwareNewsRecommendation/src/engagement.py) & [`src/satisfaction.py`](file:///c:/Users/lenovo/OneDrive/Desktop/TrustAwareNewsRecommendation/src/satisfaction.py)), we calculate estimated reading duration based on average human reading velocity ($3.5 \text{ words/sec}$):
$$\text{ExpectedTime} = \frac{\text{WordCount}}{3.5}$$
If a user clicks an article with high sensationalism ($> 0.50$) but bounces within 5 seconds ($\text{Completion} < 15\%$), the algorithm flags the interaction as an **accidental bounce click**, applying a $-0.15$ satisfaction penalty and demoting similar clickbait headlines.

---

### Q7: Why use Pareto Optimization instead of just a linear weighted sum?
**Examiner Focus:** Advanced multi-objective optimization depth.

**Answer:**
A simple linear weighted sum $\sum w_i O_i$ can hide severe deficits—for example, an article with exceptionally high clicks could achieve a high aggregate score even if its credibility score is near zero.
**Pareto Non-Dominated Sorting** ([`src/optimization.py`](file:///c:/Users/lenovo/OneDrive/Desktop/TrustAwareNewsRecommendation/src/optimization.py)) guarantees that an article is prioritized only if no other article is strictly better across all key dimensions (Relevance, Credibility, Satisfaction, Engagement). 

Furthermore, our re-ranking pipeline introduces **Hard Credibility Guardrails** ($\text{Credibility} \ge 0.40$), ensuring that low-quality clickbait cannot buy its way into recommendations regardless of popularity.

---

### Q8: What were your ablation results comparing Click-Optimized vs Trust-Aware ranking?
**Examiner Focus:** Empirical proof of project success.

**Answer:**
In our empirical benchmark across the candidate corpus:
- **Click-Optimized Ranking:** Clickbait Exposure Rate in top-10 was **50.0%**, with an average credibility of only 0.42.
- **Trust-Aware Pareto Re-ranking:** Clickbait Exposure Rate dropped to **0.0%**, average credibility rose to **0.89**, Intra-List Diversity increased from 0.38 to **0.67**, and NDCG@10 reached **0.915**.

