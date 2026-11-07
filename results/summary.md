| Model                        | Train → Test          | Macro-F1 [95% CI]    |   Accuracy |
|:-----------------------------|:----------------------|:---------------------|-----------:|
| Loughran-McDonald lexicon    | No training → News    | 0.462 [0.409, 0.511] |      0.608 |
| Loughran-McDonald lexicon    | No training → Twitter | 0.444 [0.422, 0.467] |      0.585 |
| TF-IDF + logistic regression | News → News           | 0.884 [0.846, 0.917] |      0.914 |
| TF-IDF + logistic regression | News → Twitter        | 0.415 [0.392, 0.438] |      0.668 |
| TF-IDF + logistic regression | Twitter → News        | 0.661 [0.610, 0.711] |      0.746 |
| TF-IDF + logistic regression | Twitter → Twitter     | 0.770 [0.750, 0.791] |      0.832 |
