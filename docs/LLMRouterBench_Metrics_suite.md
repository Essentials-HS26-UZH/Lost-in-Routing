# LLM Router Evaluation Metrics Suite

> **Source Reference:** Formalized based on **LLMRouterBench** (*Yuli et al., 2025* / *LLMRouterBench.pdf*).

## 1. Overview & Notation

Let $D$ be the set of evaluation datasets or language subsets ($d \in D$).  
Let $a$ denote a specific routing classifier/method.  
Let $\text{Acc}(a, d)$ denote the accuracy of router $a$ on dataset/language subset $d$.

### Baseline Reference Models
1. **Best Single Model ($B$):** The single candidate model (e.g., Large Model) achieving the highest average accuracy across all datasets/queries in hindsight.
2. **Oracle Baseline ($O$):** An ideal hindsight router that, for every query, selects the candidate model that produces a correct prediction (preferring the lowest-cost model when multiple candidates are correct).
3. **Random Router ($R$):** A lower-bound baseline that selects a candidate model uniformly at random for each query.

---

## 2. Performance Metrics

### 2.1 Average Accuracy ($\text{AvgAcc}$)
* **Description:** The primary metric measuring the overall mean correctness of the model outputs selected by the router across all datasets or language subsets.
* **Formula:**
  $$\text{AvgAcc} = \frac{1}{|D|} \sum_{d \in D} \text{Acc}(a, d)$$
* **Computation:** Average the classification/task accuracy of router $a$ across all evaluated dataset subsets.

### 2.2 Gain over Best Single Model ($\text{Gain@B}$)
* **Description:** The relative percentage accuracy improvement achieved by router $a$ over always calling the single best candidate model ($B$).
* **Formula:**
  $$\text{Gain@B} = \frac{1}{|D|} \sum_{d \in D} \left( \frac{\text{Acc}(a, d)}{\text{Acc}(B, d)} - 1 \right)$$
* **Computation:** Compute the relative accuracy ratio $\frac{\text{Acc}(a, d)}{\text{Acc}(B, d)} - 1$ for each subset $d$, then take the mean across all subsets.

### 2.3 Gain over Random Router ($\text{Gain@R}$)
* **Description:** The relative percentage accuracy improvement of router $a$ compared to a random model selection baseline ($R$).
* **Formula:**
  $$\text{Gain@R} = \frac{1}{|D|} \sum_{d \in D} \left( \frac{\text{Acc}(a, d)}{\text{Acc}(R, d)} - 1 \right)$$
* **Computation:** Compute the relative gain over the random baseline across all datasets $d \in D$.

### 2.4 Gap to Oracle ($\text{Gap@O}$)
* **Description:** Quantifies the distance between router $a$ and the theoretical maximum accuracy achieved by the hindsight Oracle ($O$). Lower values indicate higher optimality.
* **Formula:**
  $$\text{Gap@O} = \frac{1}{|D|} \sum_{d \in D} \left( 1 - \frac{\text{Acc}(a, d)}{\text{Acc}(O, d)} \right)$$
* **Computation:** Measure $1 - \frac{\text{Acc}(a, d)}{\text{Acc}(O, d)}$ for each subset $d$ and average the result across $|D|$.

---

## 3. Performance-Cost Tradeoff Metrics

In a performance-cost routing configuration, the router $a$ possesses a tunable threshold parameter $\theta \in \Theta$ (e.g., willingness-to-pay or probability cutoff) that controls the trade-off between accuracy and inference cost.

### 3.1 Peak Performance Gain ($\text{PerfGain}$)
* **Description:** The maximum relative accuracy improvement achievable over the Best Single Model ($B$) across all threshold settings $\theta \in \Theta$, irrespective of cost.
* **Formula:**
  $$\theta^* = \arg\max_{\theta \in \Theta} \text{AvgAcc}(\theta)$$
  $$\text{PerfGain} = \frac{\text{AvgAcc}(\theta^*)}{\text{AvgAcc}(B)} - 1$$
* **Computation:** Evaluate the router across a grid of thresholds $\Theta$, find the configuration $\theta^*$ yielding the highest overall accuracy, and compute its relative gain over $B$.

### 3.2 Maximum Cost Savings ($\text{CostSave}$)
* **Description:** The maximum percentage reduction in total inference dollar cost achievable by the router without dropping accuracy below the Best Single Model ($B$).
* **Formula:**
  $$\theta^\dagger = \arg\min_{\theta \in \Theta} \text{Cost}(\theta) \quad \text{s.t.} \quad \text{AvgAcc}(\theta) \ge \text{AvgAcc}(B)$$
  $$\text{CostSave} = 1 - \frac{\text{Cost}(\theta^\dagger)}{\text{Cost}(B)}$$
* **Computation:** Identify the cheapest threshold configuration $\theta^\dagger$ whose average accuracy meets or exceeds $\text{AvgAcc}(B)$, then calculate the proportional cost reduction relative to $\text{Cost}(B)$.

### 3.3 Distance to Pareto Frontier ($\text{ParetoDist}$)
* **Description:** Measures how close the operating points of a router's threshold configurations are to the empirical Pareto frontier $P$ formed by all candidate models and router configurations.
* **Formula:**
  Let $\tilde{\theta}$ and $\tilde{y}^*$ represent $L_1$-normalized (cost, accuracy) coordinates in $[0, 1]^2$:
  $$\text{ParetoDist} = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \min_{y^* \in P} \|\tilde{\theta} - \tilde{y}^*\|_1$$
* **Computation:** Normalize cost and accuracy to $[0, 1]$, extract the non-dominated Pareto frontier points $P$, and calculate the mean $L_1$ distance from each threshold configuration $\theta$ to its nearest point on $P$.

---

## 4. Diagnostic Metrics

### 4.1 Recall@k
* **Description:** Evaluates whether a correct candidate model (or the optimal model choice) is ranked within the top-$k$ predictions generated by the router classifier.
* **Formula:**
  $$\text{Recall@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \text{TopK}(a(q)) \cap C(q) \neq \emptyset \right)$$
  where $Q$ is the set of queries, $C(q)$ is the set of correct models for prompt $q$, and $\text{TopK}(a(q))$ are the top-$k$ candidate choices predicted by router $a$.
* **Computation:** Count the fraction of test queries where at least one correct model is included in the top-$k$ ranked choices of the classifier.

---

## 5. Summary Table

| Metric | Category | Primary Goal | Baseline Benchmark | Ideal Value |
| :--- | :--- | :--- | :--- | :--- |
| **$\text{AvgAcc}$** | Performance | Measure overall accuracy | Mean accuracy | $100\%$ |
| **$\text{Gain@B}$** | Performance | Relative gain over Large Model | Best Single Model ($B$) | $> 0\%$ |
| **$\text{Gap@O}$** | Performance | Distance to theoretical limit | Hindsight Oracle ($O$) | $0\%$ |
| **$\text{PerfGain}$** | Cost-Tradeoff | Peak accuracy improvement | Best Single Model ($B$) | Maximize |
| **$\text{CostSave}$** | Cost-Tradeoff | Max cost savings at equal accuracy | Best Single Model ($B$) | Maximize ($1.0$) |
| **$\text{ParetoDist}$** | Cost-Tradeoff | Proximity to optimal Pareto curve | Empirical Pareto Frontier ($P$) | $0.0$ |
| **$\text{Recall@k}$** | Diagnostic | Identify model-recall failures | Correct Model Set $C(q)$ | $1.0$ |
