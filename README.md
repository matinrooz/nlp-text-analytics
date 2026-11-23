# nlp-text-analytics

**What:** A classical and a neural method side by side on language detection and movie title search.  
**Result:** Naive Bayes on character n-grams got 75.7% on single words vs. 60.0% for stop-word counting; n-grams handle typos, embeddings handle meaning.  
**Stack:** Python, NLTK, sentence-transformers, pandas

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

For title search I compared the top result of both methods. With the typo "shawshenk",
n-grams find The Shawshank Redemption while the embedding model returns Zwartboek, and for
"the dark night" n-grams find The Dark Knight while embeddings return Night on Earth. It goes
the other way when the query is about meaning: "toys that come alive" gives Toy Story with
embeddings but All That Jazz with n-grams, and "space adventure" gives Star Trek vs. The
Adventures of Robin Hood.

n-grams handle typos, embeddings handle meaning. The two fail in opposite cases. The embedding
model reads "night" as night, so it looks for night films and never gets to The Dark Knight.

For language detection I used 1500 held-out European Parliament sentences (300 per
language), and also cut them down to their first words. With full sentences both methods are
almost perfect (99.7% for stop words, 100% for Naive Bayes). The difference shows on short
texts: with a single word, stop-word counting gets 60.0% and Naive Bayes 75.7%, with three
words 81.3% vs. 94.5%, and with six words 93.3% vs. 98.9%.

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
