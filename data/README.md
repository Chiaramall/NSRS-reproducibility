# Data

`editkg-filtered` contains the reconstructed datasets used for the strict EditKG reproduction. Each dataset directory contains `train.txt`, `test.txt`, `kg_final.txt`, identifier lists, and a filtering manifest.

`editkg-validation` contains deterministic per-user validation views generated from the training set with seed 2024. At least one interaction remains in training for every user. Its `test.txt` file is the validation split expected by the official baseline loaders; `final_test.txt` points to the untouched held-out test set.

The tuning code must use `editkg-validation`. Final evaluation must use `editkg-filtered`.
