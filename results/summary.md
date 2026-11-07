| Model                        | Train → Test          | Macro-F1 [95% CI]    |   Accuracy |
|:-----------------------------|:----------------------|:---------------------|-----------:|
| Loughran-McDonald lexicon    | No training → News    | 0.462 [0.409, 0.511] |      0.608 |
| Loughran-McDonald lexicon    | No training → Twitter | 0.444 [0.422, 0.467] |      0.585 |
| TF-IDF + logistic regression | News → News           | 0.884 [0.846, 0.917] |      0.914 |
| TF-IDF + logistic regression | News → Twitter        | 0.415 [0.392, 0.438] |      0.668 |
| TF-IDF + logistic regression | Twitter → News        | 0.661 [0.610, 0.711] |      0.746 |
| TF-IDF + logistic regression | Twitter → Twitter     | 0.770 [0.750, 0.791] |      0.832 |
| DistilBERT (fine-tuned)      | News → News           | 0.951 [0.925, 0.972] |      0.96  |
| DistilBERT (fine-tuned)      | News → Twitter        | 0.594 [0.571, 0.618] |      0.691 |
| DistilBERT (fine-tuned)      | Twitter → News        | 0.843 [0.801, 0.875] |      0.872 |
| DistilBERT (fine-tuned)      | Twitter → Twitter     | 0.822 [0.803, 0.841] |      0.867 |
| FinBERT (fine-tuned)         | News → News           | 0.951 [0.923, 0.973] |      0.962 |
| FinBERT (fine-tuned)         | News → Twitter        | 0.677 [0.656, 0.700] |      0.739 |
| FinBERT (fine-tuned)         | Twitter → News        | 0.839 [0.799, 0.876] |      0.867 |
| FinBERT (fine-tuned)         | Twitter → Twitter     | 0.825 [0.807, 0.844] |      0.867 |
