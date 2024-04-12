**Node classification results (%). We report the results of average Macro-F1 and Micro-F1 over ten trials and highlight the best score on each dataset in bold.**   

(*And *\* denote p-value<0.05 and p-value<0.01 with T-test, respectively)

- **ACM**

| Metric   | Training | HGMAE          | HGCA                | GraMI               |
| -------- | -------- | -------------- | ------------------- | ------------------- |
| Macro-F1 | 10%      | 84.37&plusmn;1.02 | **91.94&plusmn;1.02**  | 91.31&plusmn;0.34      |
|          | 20%      | 84.88&plusmn;0.38 | **92.13&plusmn;0.54*** | 91.64&plusmn;0.28      |
|          | 40%      | 85.74&plusmn;0.46 | 92.56&plusmn;0.52      | **92.71&plusmn;0.44*** |
|          | 60%      | 86.94&plusmn;0.49 | 93.08&plusmn;0.41      | **93.16&plusmn;0.59***  |
|          | 80%      | 87.54&plusmn;0.87 | 93.11&plusmn;0.77      | **93.23&plusmn;1.01*** |
| Micro-F1 | 10%      | 84.87&plusmn;0.95 | **91.85&plusmn;0.92**  | 91.22&plusmn;0.42      |
|          | 20%      | 85.12&plusmn;0.43 | **92.01&plusmn;0.46*** | 91.57&plusmn;0.30      |
|          | 40%      | 86.03&plusmn;0.55 | 92.58&plusmn;0.44      | **92.73&plusmn;0.61*** |
|          | 60%      | 87.14&plusmn;0.62 | 93.02&plusmn;0.31      | **93.10&plusmn;0.63***  |
|          | 80%      | 88.11&plusmn;0.94 | 92.97&plusmn;0.85      | **93.35&plusmn;0.98*** |

- **DBLP**

| Metric   | Training | HGMAE          | HGCA           | GraMI                |
| -------- | -------- | -------------- | -------------- | -------------------- |
| Macro-F1 | 10%      | 88.14&plusmn;0.45 | 91.23&plusmn;0.82 | **93.67&plusmn;0.63**** |
|          | 20%      | 88.71&plusmn;0.55 | 92.25&plusmn;0.74 | **93.85&plusmn;0.39**** |
|          | 40%      | 89.32&plusmn;0.61 | 93.01&plusmn;0.53 | **94.13&plusmn;0.41***  |
|          | 60%      | 89.83&plusmn;0.93 | 93.17&plusmn;0.45 | **94.03&plusmn;0.39***  |
|          | 80%      | 90.41&plusmn;1.16 | 94.12&plusmn;0.64 | **95.00&plusmn;0.48**** |
| Micro-F1 | 10%      | 89.41&plusmn;0.37 | 92.01&plusmn;0.81 | **94.12&plusmn;0.60**** |
|          | 20%      | 89.65&plusmn;0.52 | 92.94&plusmn;0.72 | **94.31&plusmn;0.37**** |
|          | 40%      | 90.13&plusmn;0.53 | 93.55&plusmn;0.53 | **94.56&plusmn;0.35***  |
|          | 60%      | 90.62&plusmn;0.80 | 92.96&plusmn;0.47 | **94.45&plusmn;0.39**** |
|          | 80%      | 91.15&plusmn;1.03 | 93.18&plusmn;0.63 | **95.36&plusmn;0.50**** |

- **YELP**

| Metric   | Training | HGMAE          | HGCA           | GraMI               |
| -------- | -------- | -------------- | -------------- | ------------------- |
| Macro-F1 | 10%      | 57.58&plusmn;3.38 | 91.37&plusmn;0.81 | **91.48&plusmn;0.40**  |
|          | 20%      | 60.43&plusmn;3.17 | 91.91&plusmn;0.56 | **92.05&plusmn;0.36***  |
|          | 40%      | 63.92&plusmn;2.57 | 92.81&plusmn;0.57 | **92.91&plusmn;0.56*** |
|          | 60%      | 67.23&plusmn;1.82 | 93.22&plusmn;0.71 | **93.43&plusmn;0.81*** |
|          | 80%      | 68.37&plusmn;2.24 | 93.41&plusmn;1.28 | **93.74&plusmn;0.40*** |
| Micro-F1 | 10%      | 73.81&plusmn;0.93 | 90.69&plusmn;0.81 | **91.01&plusmn;0.46**  |
|          | 20%      | 74.92&plusmn;1.26 | 91.03&plusmn;0.56 | **91.74&plusmn;0.41***  |
|          | 40%      | 75.35&plusmn;1.30 | 92.33&plusmn;0.55 | **92.58&plusmn;0.62*** |
|          | 60%      | 76.34&plusmn;1.24 | 92.63&plusmn;0.72 | **92.97&plusmn;0.80*** |
|          | 80%      | 77.87&plusmn;1.81 | 92.98&plusmn;1.31 | **93.35&plusmn;0.40*** |
