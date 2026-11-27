# Decision note (DRAFT, week 10)

Half a page when finished. Fill the [brackets] after you run `python train.py`. Final version is due before the v1 release (week 13).

## 1. Who the client is
The instructor confirmed we decide this ourselves. The client is a school that decides the evening before whether PE lessons move indoors. We act as that school's adviser.

## 2. Metric and threshold: why they fit the client
- **Number forecast:** MAE in µg/m³ (the average size of the mistake, in the unit people understand).
- **Warning flag:** **recall** is the main metric. A missed unhealthy day (children outside in bad air) costs more than a false alarm (one lesson moved indoors for nothing). We also watch **precision** so the channel does not cry wolf every day.
- **Threshold:** share of training days at or above each level: 15 µg/m³ is 51%, 25 is 13%, 35 is 6%, 55 is 1%. The WHO value of 15 flags half of all days, so it cannot work as a warning. We propose **25 µg/m³** (about one day in eight). Agreed between A and B on [date]. Reason to write here: [your own words].

## 3. Baselines to beat (from `train.py`, time-ordered cross-validation, train data only)
| Model | MAE (µg/m³) | Recall | Precision |
|---|---|---|---|
| Dummy (median) | [6.17] | [0.00] | [n/a] |
| Persistence (tomorrow = today) | [4.42] | [0.52] | [0.62] |
| Linear regression | [3.64] | [0.64] | [0.66] |

Persistence is the rule to beat. If no course model beats it, we say so honestly in the report.

## 4. Data fix we tried
[Week 11: what the error worksheet showed, which fix we tried, and how it changed the score.]

## 5. Model change we tried
[Week 12: kNN, decision tree, logistic regression for the flag; what we changed and why.]

## 6. What the learning curve says
[Week 12: does more data help?]

## 7. What we release as v1
[Week 13: the chosen model, its threshold, and why.]
