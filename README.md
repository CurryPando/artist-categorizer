## Purpose
Create a model that takes in lyrics, and spits out a prediction for the artist it thinks wrote those lyrics.

## Data
This uses the genius-song-lyrics dataset from sebastiandizon on Hugging Face.

I narrowed this dataset to just 27 hip-hop artists, since the genre tends to put more focus on lyrical content than others.
The hip-hop artists chosen are generally well-known artists with enough data and/or a distinct lyrical pen.

## Implementation
This uses a base ModernBERT model from answerdotai on hugging face fine-tuned for classification.

Flash attention 2 was used to speed up training time on my single A1000 6GB.

Multi-fidelity Bayesian Optimization was used via HyperbandPruner and Optuna to get the best hyperparameters.

## Results
The total f1-score for this is ~0.45. From personal testing and considering the task at hand, the performance of this model is certainly good enough, and exceeded my expectations for what was possible with a simple BERT model.
