# Datasets

This directory contains the processed training, validation and test splits. Files are tab-separated despite the `.csv` suffix. The data loader regenerates metadata and grouped histories.


Interactions are ordered by timestamp for each user. The last positive interaction (rating > 3) is held out for testing, and the second-to-last for validation. Users with fewer than five interactions are kept in the training set only.

Interactions with label 0 are excluded from validation and test metrics. They remain in the files because negative feedback in the validation period can be part of the history used to predict the test interaction.
