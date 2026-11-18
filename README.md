# nlp-text-analytics

Four small text experiments in one notebook (`nlp_text_analytics.ipynb`): language detection
with stop words and with Naive Bayes, and movie title search with character n-grams and with
sentence transformers.

I wanted to put a simple classical method next to a learned one on the same task each time,
and see where each one breaks, instead of assuming the neural model is always better.

## What is in it

- **a)** Detect the language of a text by counting NLTK stop words per language.
- **b)** Search IMDB movie titles with character 2- to 4-grams (accents removed with
  `unidecode`, word start and end marked) and Jaccard similarity.
- **c)** The same search with sentence embeddings (`all-MiniLM-L6-v2` and
  `all-mpnet-base-v2`).
- **d)** Naive Bayes language detection on character n-grams for English, German, Italian,
  French and Spanish (top 1000 n-grams per language, equal prior), compared with a).

## Results

Title search, top result:

| query | n-grams | sentence transformer (MiniLM) |
|---|---|---|
| "the dark night" | The Dark Knight | Night on Earth |
| "shawshenk" | The Shawshank Redemption | Zwartboek |
| "toys that come alive" | All That Jazz | Toy Story |
| "space adventure" | The Adventures of Robin Hood | Star Trek |

n-grams handle typos, embeddings handle meaning. The two fail in opposite cases. The embedding
model reads "night" as night, so it looks for night films and never gets to The Dark Knight.

Language detection, accuracy on 1500 held-out European Parliament sentences (300 per
language), also cut to their first words:

| text length | stop words | Naive Bayes |
|---|---|---|
| 1 word | 60.0% | 75.7% |
| 3 words | 81.3% | 94.5% |
| 6 words | 93.3% | 98.9% |
| full sentence | 99.7% | 100.0% |

On short texts there are often no stop words at all, so the stop word method has nothing to
count. When all languages get 0 hits it falls back to the first language (English), which
makes its English score look better than it is.

## What I would try next

- Combine both searches (e.g. add the two scores), since their mistakes do not overlap.
- Embed the plot summary (`Overview` column) instead of only the title, so that queries like
  "a boy who is a wizard" have something to match.
- Use proper smoothing in the Naive Bayes model instead of a fixed floor, and train on more
  everyday text than parliament speeches ("Merci beaucoup" is still classified as English).

## Data

- IMDB top 1000 movies: `data/imdb_top_1000.csv`, from
  [Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
  (Harshit Shankhdhar, CC0: Public Domain).
- European Parliament texts and stop word lists: NLTK corpora `europarl_raw` and `stopwords`,
  downloaded by the notebook.

## Run

```bash
pip install nltk unidecode pandas numpy sentence-transformers jupyter
jupyter notebook nlp_text_analytics.ipynb
```
