# Natural Language Processing (NLP) — Interview Prep Syllabus

Natural Language Processing sits at the core of all three roles this syllabus targets. A **Data Scientist** uses NLP to extract insight from unstructured text — customer feedback, clinical notes, support tickets — via classification, topic modeling, and statistical text analysis, and must be able to justify preprocessing and modeling choices with rigor. A **Machine Learning Engineer** builds, trains, evaluates, and productionizes NLP pipelines — tokenizers, embedding lookups, sequence labelers, fine-tuned classifiers — and is expected to reason about latency, vocabulary size, OOV handling, and evaluation metrics precisely. An **AI Engineer** increasingly builds on top of pretrained transformer models (semantic search, RAG, intent classifiers, extraction pipelines) and needs to understand what happens *underneath* the API call: how tokenization fragments text, why BERT and GPT differ, how embeddings are compared, and how classical baselines (TF-IDF, n-gram LMs) still matter for speed, interpretability, and cost. This file covers NLP fundamentals from classical statistical methods through the transformer era at the *task and technique* level; deep architectural internals of modern LLMs (multi-head attention math, positional encodings, RLHF, etc.) are covered in the Generative AI / LLM syllabus file.

## Table of Contents

1. [Text Preprocessing](#text-preprocessing)
2. [Text Representation](#text-representation)
3. [Classical NLP Tasks and Models](#classical-nlp-tasks-and-models)
4. [Sequence-to-Sequence and Neural NLP](#sequence-to-sequence-and-neural-nlp)
5. [Transformer-based NLP](#transformer-based-nlp)
6. [Applied NLP Tasks](#applied-nlp-tasks)
7. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Text Preprocessing

Preprocessing determines what information ever reaches the model. Every choice here is a lossy transformation, and interviewers probe whether you understand the tradeoffs, not just the API calls.

### Tokenization Approaches: Word, Subword, Character-level

**Word-level tokenization** splits text on whitespace/punctuation into whole words (`"don't stop"` → `["do", "n't", "stop"]` with a rule-based tokenizer, or `["don't", "stop"]` naively). It is intuitive and interpretable but suffers from two structural problems:

1. **Vocabulary explosion** — every inflection (`run`, `runs`, `running`, `runner`) is a separate ID, so vocabularies balloon to hundreds of thousands of entries for morphologically rich languages.
2. **Out-of-vocabulary (OOV) words** — any word not seen during training (typos, new slang, rare names) becomes `<UNK>`, destroying information.

**Character-level tokenization** splits into individual characters/bytes. This eliminates OOV entirely (any string is representable) and keeps the vocabulary tiny (~256 for bytes, ~100 for a language's alphabet), but sequences become very long, which hurts compute (attention/RNN cost grows with sequence length) and makes it harder for the model to capture word-level semantics directly — it must *learn* words from characters.

**Subword tokenization** is the modern compromise: split rare/complex words into meaningful sub-units while keeping common words whole. `"unhappiness"` → `["un", "happi", "ness"]`. This bounds vocabulary size, eliminates true OOV (any word decomposes to known subwords, down to characters/bytes as a fallback), and preserves morphological structure the model can exploit.

#### Byte Pair Encoding (BPE) — mechanics

BPE (Sennrich et al., 2016, adapted from a 1994 compression algorithm) builds a vocabulary bottom-up by iteratively merging the most frequent adjacent symbol pair.

**Algorithm:**
1. Initialize vocabulary as the set of individual characters (or bytes), plus an end-of-word marker.
2. Represent the corpus as sequences of these symbols, with frequency counts per word.
3. Count all adjacent symbol pairs across the corpus.
4. Merge the single most frequent pair into a new symbol; add it to the vocabulary.
5. Repeat steps 3–4 for a fixed number of merges (a hyperparameter, e.g., 30,000–50,000) or until target vocabulary size is reached.

```python
# Minimal illustrative BPE trainer (toy scale — real implementations use tries/heaps for speed)
from collections import defaultdict, Counter

def get_vocab(corpus):
    vocab = Counter()
    for word in corpus:
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_pair_freqs(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

corpus = ["low", "lower", "lowest", "newest", "widest"] * 5
vocab = get_vocab(corpus)
num_merges = 10
for i in range(num_merges):
    pairs = get_pair_freqs(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print(f"Merge {i+1}: {best_pair}")
```

At inference time, a word is tokenized by greedily applying the learned merges in the order they were learned. BPE is **frequency-driven and deterministic** — it has no probabilistic model of segmentation.

#### WordPiece — mechanics (used by BERT)

WordPiece looks similar to BPE but differs in the merge criterion: instead of merging the most *frequent* pair, it merges the pair that maximizes the **likelihood of the training corpus** under a unigram language model, i.e., it picks the pair `(a, b)` that maximizes:

```
score(a, b) = freq(ab) / (freq(a) * freq(b))
```

This normalizes for the fact that some symbols are frequent on their own; WordPiece prefers merges where the combined unit is disproportionately more common than its parts would predict independently (similar to pointwise mutual information). WordPiece also marks continuation subwords with `##` (e.g., `playing` → `play`, `##ing`), which lets you unambiguously reconstruct whether a token started a new word.

```python
# Conceptual scoring difference vs BPE
# BPE:        pick pair maximizing freq(a, b)
# WordPiece:  pick pair maximizing freq(a, b) / (freq(a) * freq(b))
```

#### SentencePiece — mechanics

SentencePiece is a *library/framework* (not a single algorithm) that implements both BPE and a **Unigram Language Model** tokenizer, with two crucial engineering differences from the above:

1. **Language-agnostic, no pre-tokenization**: it treats input as a raw stream of Unicode characters (or bytes) including whitespace itself as a symbol (usually mapped to `▁`), so no whitespace-based pre-tokenization step is required. This makes it work naturally for languages without whitespace-delimited words (Japanese, Chinese, Thai).
2. **Unigram LM mode**: starts with a large seed vocabulary and *iteratively prunes* subwords to maximize corpus likelihood under a unigram model (each token has an independent probability), using the EM algorithm and Viterbi-like decoding to find the best segmentation. This is the opposite direction from BPE (top-down pruning vs bottom-up merging), and it gives a *probabilistic* tokenizer — segmentation can be sampled, enabling regularization ("subword regularization": train on multiple valid segmentations of the same word for robustness).

| Property | BPE | WordPiece | SentencePiece (Unigram) |
|---|---|---|---|
| Direction | Bottom-up merge | Bottom-up merge | Top-down prune |
| Merge/score criterion | Pair frequency | Pair likelihood ratio | Corpus likelihood (unigram LM) |
| Whitespace handling | Needs pre-tokenizer | Needs pre-tokenizer | Built-in (`▁` marker), works without pre-tokenization |
| Probabilistic segmentation | No | No | Yes (can sample alternative segmentations) |
| Used by | GPT-2/GPT-3, RoBERTa | BERT, DistilBERT | T5, ALBERT, XLNet, many multilingual models |

#### Character-level tokenization — when it's used

Used in some early CNN/RNN character-aware models (e.g., ELMo's character CNN, char-RNN language models) and in byte-level BPE (GPT-2 operates on UTF-8 *bytes*, not Unicode characters, guaranteeing zero OOV across any input including emoji and unseen scripts, at the cost of longer sequences for non-Latin scripts).

**Pitfalls:**
- Comparing tokenizer vocabulary sizes without accounting for average tokens-per-word is misleading — a "smaller" vocabulary can still produce longer sequences.
- Subword tokenization can split semantically atomic tokens (e.g., a company name split into three fragments), hurting few-shot/zero-shot performance on rare entities. This is why domain-specific vocabularies (BioBERT, SciBERT, legal-BERT) retrain the tokenizer on in-domain corpora.
- Non-English/morphologically rich languages get worse "compression" from BPE trained mostly on English corpora (this is measured as tokens-per-word, and directly affects both cost and context-window usage in production LLM systems).

### Normalization: Lowercasing, Stemming vs Lemmatization, Stopwords

**Lowercasing** collapses case variants (`"Apple"` vs `"apple"`) to reduce vocabulary size, but it destroys information: `"Apple"` (company) vs `"apple"` (fruit), and it hurts NER, where capitalization is a strong signal. Cased models (`bert-base-cased`) retain case for this reason.

**Stemming** crudely chops word endings using rule-based heuristics to reach a common (not necessarily real) root. The classic algorithm is the **Porter Stemmer** (1980), which applies an ordered sequence of suffix-stripping rules in multiple passes (e.g., strip `-sses` → `-ss`, `-ies` → `-i`, `-ational` → `-ate`, subject to conditions on syllable count of the stem). It's fast and rule-based but can produce non-words.

```python
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')

porter = PorterStemmer()
words = ["running", "runs", "ran", "better", "studies", "universal", "caresses"]
print([porter.stem(w) for w in words])
# ['run', 'run', 'ran', 'better', 'studi', 'univers', 'caress']
# Note: "ran" -> "ran" (irregular form not normalized), "better" unchanged (not "good"),
# "studies" -> "studi" (not a real word), "universal" -> "univers"

lemmatizer = WordNetLemmatizer()
print([lemmatizer.lemmatize(w, pos='v') for w in ["running", "runs", "ran", "studies"]])
# ['run', 'run', 'run', 'study']   -- requires POS to resolve correctly (ran -> run only if pos='v')
```

**Lemmatization** maps a word to its dictionary base form (*lemma*) using vocabulary and morphological analysis (e.g., WordNet, or a full morphological analyzer), correctly handling irregular forms (`ran` → `run`, `better` → `good` if treated as adjective, `mice` → `mouse`). It's linguistically correct but slower and typically requires POS tagging as input to disambiguate (`"leaves"` as noun-plural of *leaf* vs verb form of *leave*).

| | Stemming | Lemmatization |
|---|---|---|
| Output | Possibly non-word root | Valid dictionary word |
| Speed | Fast (rule-based) | Slower (dictionary/morphology lookup) |
| Accuracy | Crude, can over/under-stem | More accurate, context/POS aware |
| Irregular forms | Not handled (`ran`→`ran`) | Handled (`ran`→`run`) |
| Typical use | Search/IR, fast pipelines | Downstream linguistic tasks, chatbots |

**Stopword removal** drops high-frequency, low-information words (`the`, `is`, `a`, `and`). This helps classical bag-of-words/TF-IDF pipelines by reducing noise and dimensionality for tasks like topic modeling or document retrieval where function words rarely help discriminate topics.

**When NOT to remove stopwords:**
- **Sentiment/negation-sensitive tasks**: "not good" vs "good" — removing "not" flips meaning entirely.
- **Machine translation, generation, summarization**: function words are structurally essential for fluent output.
- **Phrase/collocation-sensitive tasks**: "to be or not to be" becomes empty after aggressive stopword removal.
- **Any modern transformer-based pipeline**: contextual models (BERT, GPT) are trained on natural, un-stripped text; removing stopwords before feeding them to a pretrained transformer is *harmful* — it creates a distribution mismatch with pretraining data and destroys the syntactic cues the model relies on.
- **POS tagging, parsing, NER, QA**: stopwords carry grammatical structure needed for correct labeling.

### Handling of Punctuation, Special Tokens, and OOV Words

**Punctuation** handling depends on the task: for TF-IDF/BoW pipelines it's usually stripped; for sentiment (emphasis via `!!!`, `?`), sarcasm detection, or code/formula-heavy text, punctuation carries signal and should be preserved or explicitly featurized (e.g., "exclamation count" as a feature). Tokenizer-level, most subword tokenizers keep punctuation as separate tokens rather than stripping it, since transformers can learn to use or ignore it.

**Special tokens** are reserved vocabulary entries with fixed roles, not natural text:

| Token | Purpose |
|---|---|
| `[CLS]` / `<s>` | Sequence-start marker; in BERT its final hidden state is used as the pooled sequence representation for classification |
| `[SEP]` / `</s>` | Separates two segments (e.g., question/context in QA, sentence pairs) or marks sequence end |
| `[PAD]` | Pads shorter sequences to a fixed batch length; must be masked out of loss/attention |
| `[MASK]` | Placeholder for masked-language-modeling pretraining (BERT) |
| `[UNK]` | Fallback for tokens absent from vocabulary (rare with subword tokenizers, common with word-level ones) |
| `<bos>` / `<eos>` | Beginning/end of sequence, critical for autoregressive generation to know when to stop |

**OOV handling strategies:**
1. **`<UNK>` mapping** (word-level models): simple but lossy — many distinct rare words collapse to one symbol.
2. **Subword fallback** (BPE/WordPiece/SentencePiece): decompose unknown words into known subwords/characters, so true OOV essentially disappears (worst case: falls back to individual bytes/characters).
3. **Character n-gram hashing** (FastText): represent a word as the sum of its character n-gram embeddings, so *any* string — including misspellings and unseen morphological variants — gets a reasonable vector.
4. **Hashing trick**: hash tokens into a fixed-size bucket space (used in some large-scale feature hashing pipelines) to bound memory regardless of vocabulary growth, at the cost of hash collisions.

**Pitfalls:**
- Forgetting to mask `[PAD]` tokens in attention/loss computation silently leaks padding into training signal.
- Mismatched preprocessing between train and serving (e.g., lowercasing in training but not inference) is one of the most common silent production bugs — always pin the exact tokenizer artifact (vocab file + merges file + config) alongside the model.
- Truncating sequences without care can cut off exactly the informative part of the text (e.g., truncating a long review might drop the conclusion where sentiment is stated).

### Text Data Augmentation for NLP

Data augmentation synthesizes additional labeled training examples from existing ones — useful when labeled data is scarce, classes are imbalanced, or the model needs to be robust to input noise/variation. It is a training-data-side lever, distinct from architectural regularization (dropout, weight decay).

**EDA (Easy Data Augmentation, Wei & Zou, 2019)** bundles four simple, model-free operations, typically applied with a small probability per word:

| Technique | Operation | Example |
|---|---|---|
| Synonym Replacement (SR) | Replace `n` random non-stopwords with a synonym (WordNet or embedding nearest-neighbor) | "The movie was **great**" → "The movie was **fantastic**" |
| Random Insertion (RI) | Insert a synonym of a random word at a random position | "The movie was great" → "The **excellent** movie was great" |
| Random Swap (RS) | Swap the positions of two random words | "The movie was great" → "Movie the was great" |
| Random Deletion (RD) | Randomly delete each word with probability `p` | "The movie was great" → "The movie great" |

```python
import random
from nltk.corpus import wordnet

def synonym_replacement(words, n):
    new_words = words.copy()
    candidates = [w for w in words if wordnet.synsets(w)]
    random.shuffle(candidates)
    replaced = 0
    for word in candidates:
        synonyms = {lemma.name().replace('_', ' ')
                    for syn in wordnet.synsets(word) for lemma in syn.lemmas()
                    if lemma.name().lower() != word.lower()}
        if synonyms:
            new_words = [random.choice(list(synonyms)) if w == word else w for w in new_words]
            replaced += 1
        if replaced >= n:
            break
    return new_words

print(synonym_replacement("the movie was great and fun".split(), n=1))
```

**Back-translation**: translate the source text into an intermediate (pivot) language and then back into the original language using machine translation models (e.g., English → French → English). The round-trip introduces lexical and syntactic paraphrasing while — usually — preserving meaning, giving diverse but semantically faithful augmented examples; widely used in low-resource text classification and NLI augmentation.

```
Original:        "This film was absolutely fantastic."
      ↓ MT (en→fr)
Pivot:            "Ce film était absolument fantastique."
      ↓ MT (fr→en)
Back-translated:  "This movie was absolutely amazing."
```

**Contextual/model-based augmentation**: mask random tokens and have a pretrained MLM (e.g., BERT) fill them in with contextually plausible substitutes (more fluent, context-aware than WordNet synonym lookup); or use a paraphrase-generation model (e.g., a fine-tuned T5) to rewrite whole sentences.

**Other techniques**: character-level noise injection (simulate typos/OCR errors for robustness), random word/character deletion for regularization, and — increasingly — prompting a large generative LLM to produce paraphrases or synthetic examples for a target class (covered from the generation/prompting angle in the Generative AI/LLM syllabus file; here the focus is the classical, model-agnostic techniques and why one augments text data at all).

**Pitfalls:**
- **Label invalidation**: naive synonym replacement can flip the label if it touches a negation or sentiment-bearing word ("not good" → replacing "good" changes sentiment, but replacing a neutral word is usually safe) — augmentation must be applied carefully around label-critical tokens.
- **Semantic drift in back-translation**: round-trip translation isn't guaranteed to preserve meaning exactly, especially for idioms, named entities, or numbers — always spot-check a sample of augmented outputs.
- **Diminishing/negative returns**: over-augmenting (especially with aggressive random swap/deletion) adds noise that can hurt performance past a certain point; augmentation is not a substitute for fixing fundamental label noise or class-imbalance issues at the source.
- **Train/test contamination**: augmented variants of a training example must stay strictly within the training split — leaking near-duplicate augmented examples across train/test splits inflates evaluation metrics.

### Interview Questions

1. **Q: Why do modern NLP systems prefer subword tokenization over pure word-level tokenization?**
   A: Word-level tokenization causes vocabulary explosion for morphologically rich languages and produces true OOV tokens for words not seen in training, both of which degrade downstream performance. Subword tokenization (BPE/WordPiece/SentencePiece) keeps a bounded vocabulary (e.g., 30–50K tokens), decomposes rare/unseen words into known pieces (eliminating true OOV — worst case falls back to characters/bytes), and lets the model exploit shared morphology (e.g., `un-`, `-ing`) across many words.

2. **Q: Explain the core algorithmic difference between BPE and WordPiece.**
   A: Both build vocabulary bottom-up via iterative merges, but BPE merges the pair with the highest raw frequency, while WordPiece merges the pair that maximizes `freq(ab) / (freq(a) * freq(b))` — a likelihood-ratio/PMI-like score that favors merges where the combined unit occurs disproportionately more often together than its parts would predict independently.

3. **Q: How does SentencePiece differ fundamentally from BPE/WordPiece in its approach?**
   A: SentencePiece can operate top-down: it starts with a large seed vocabulary and prunes tokens to maximize likelihood of the corpus under a unigram language model (EM + Viterbi), rather than bottom-up merging. It's also whitespace-agnostic — it treats the input as a raw character/byte stream (mapping space to `▁`), so it needs no separate word pre-tokenization step, making it well-suited for languages without whitespace-delimited words.

4. **Q: What is the Porter Stemmer and what's a concrete weakness of stemming vs lemmatization?**
   A: Porter Stemmer is a rule-based algorithm applying an ordered sequence of suffix-stripping rules (e.g., `-ational`→`-ate`) in multiple passes to reduce a word to a crude root. Weakness: it can produce non-dictionary words (`"studies"`→`"studi"`) and doesn't handle irregular forms (`"ran"` stays `"ran"` instead of becoming `"run"`), whereas lemmatization uses vocabulary/morphology (often with POS context) to return valid dictionary lemmas correctly, including irregulars.

5. **Q: Give two scenarios where removing stopwords would hurt model performance.**
   A: (1) Sentiment analysis / negation handling — removing "not" from "not good" flips the sentiment. (2) Any task using a pretrained transformer (BERT/GPT) — these were pretrained on natural, unstripped text, so removing stopwords creates a train/inference mismatch and destroys syntactic structure the self-attention mechanism relies on.

6. **Q: What is the role of the `[CLS]` token in BERT-style models?**
   A: It is prepended to every input sequence; after passing through all transformer layers, its final hidden state is treated as an aggregate/pooled representation of the whole sequence, typically fed into a classification head for sentence- or pair-level tasks (e.g., sentiment classification, NLI).

7. **Q: How does FastText handle out-of-vocabulary words differently from Word2Vec?**
   A: Word2Vec assigns one atomic vector per whole word, so an unseen word has no vector at all (or a generic `<UNK>` vector). FastText represents each word as the sum of embeddings of its character n-grams (plus the whole word itself), so it can synthesize a reasonable vector for *any* string, including misspellings and unseen morphological variants, by composing from n-grams it has seen.

8. **Q: Why must `[PAD]` tokens be masked during attention and loss computation?**
   A: Padding tokens carry no real information; without masking, self-attention would attend to padding positions (introducing noise) and the loss would be computed on predictions for padding positions (introducing meaningless gradient signal), both of which degrade training quality and can bias the model, especially in variable-length-heavy batches.

9. **Q: What is byte-level BPE and why did GPT-2 use it?**
   A: Byte-level BPE performs the BPE merge algorithm over raw UTF-8 bytes (256 possible base symbols) rather than Unicode characters. This guarantees the tokenizer can represent *any* input — any language, emoji, or malformed text — with zero true OOV, without needing a large base character inventory for multi-script coverage.

10. **Q: What's the practical production risk of tokenizer/preprocessing mismatch between training and serving?**
    A: If preprocessing steps (lowercasing, tokenizer version, special-token handling, truncation length) differ between training and inference, the model receives inputs from a different distribution than it was trained on, causing silent accuracy degradation that's hard to detect without dedicated preprocessing-consistency tests. Best practice: version and ship the exact tokenizer artifacts (vocab + merges + config) with the model weights.

11. **Q: Why does WordPiece use a `##` prefix on continuation tokens?**
    A: It disambiguates whether a subword token starts a new word or continues the previous one, which is necessary to correctly detokenize/reconstruct the original text and to let downstream logic (e.g., NER span aggregation) know where word boundaries fall.

12. **Q: In what situation would a character-level model be preferable to subword tokenization?**
    A: When the domain has extreme OOV rates or noisy text (e.g., misspelling-heavy user-generated content, DNA/protein sequences, or arbitrary code/text mixtures) where subword vocabularies trained on clean corpora would fragment poorly; character-level models trade longer sequences for guaranteed robustness and zero OOV, at increased compute cost.

13. **Q: Does lemmatization always need POS tagging? Why?**
    A: For accurate results, yes for many words — lemmatization is often POS-dependent (`"leaves"` lemmatizes to `"leaf"` as a noun but `"leave"` as a verb). Without POS context, lemmatizers must guess (usually defaulting to noun), producing incorrect lemmas for words whose correct root depends on grammatical role.

14. **Q: What is "subword regularization" in SentencePiece's Unigram mode?**
    A: Because the Unigram LM tokenizer is probabilistic, the same word can be segmented multiple valid ways with different probabilities. During training, sampling different segmentations for the same input acts as a data augmentation/regularization technique, exposing the model to varied subword decompositions and improving robustness.

15. **Q: Why is naive whitespace tokenization inadequate for Chinese or Japanese text?**
    A: These languages don't delimit words with whitespace, so whitespace-based splitting either produces one giant "word" per sentence or is meaningless. Tools use character-based segmentation, statistical word-segmenters, or whitespace-agnostic subword tokenizers like SentencePiece that treat the input as a raw character stream.

16. **Q: What four operations does EDA (Easy Data Augmentation) apply, and why is each typically capped at a low replacement probability?**
    A: EDA applies synonym replacement, random insertion, random swap, and random deletion, each usually applied to only a small fraction of words per sentence. Keeping the probability low preserves the sentence's core meaning and grammaticality — aggressive augmentation risks flipping the label (e.g., replacing a sentiment-bearing word) or producing a sentence too corrupted to be a useful training signal.

17. **Q: How does back-translation generate augmented training data, and what's its main risk?**
    A: Back-translation translates a sentence into a pivot language and then back into the original language using machine translation, producing a paraphrase that usually preserves meaning while varying vocabulary and syntax. Its main risk is semantic drift — the round trip through MT isn't guaranteed to preserve meaning exactly, especially for idioms, numbers, or named entities, so augmented examples should be spot-checked rather than trusted blindly.

18. **Q: Why can naive synonym replacement invalidate the label of a training example?**
    A: If the replaced word is label-critical — a negation, a sentiment-bearing adjective, or a key entity — swapping it for a "synonym" can change the sentence's actual meaning or sentiment even though the replacement is lexically valid, silently corrupting the label. Augmentation pipelines need to be sentiment/label-aware, not purely lexical.

19. **Q: Why does data augmentation not substitute for fixing label noise or class imbalance at the source?**
    A: Augmentation multiplies existing examples with small perturbations; if the underlying labels are wrong or a class is fundamentally under-sampled relative to the real distribution, augmentation just multiplies that same noise/bias rather than correcting it — it improves robustness to lexical variation but can't manufacture information that isn't already present in the labeled data.

---

## Text Representation

Once text is tokenized, it must become numeric. This section spans from sparse counting-based representations to dense trained embeddings to contextual vectors.

### Bag-of-Words and TF-IDF

**Bag-of-Words (BoW)** represents a document as a vector of word counts (or binary presence), ignoring order and grammar entirely. For vocabulary size `V`, each document becomes a `V`-dimensional sparse vector.

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["the cat sat on the mat", "the dog sat on the log"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())
# ['cat' 'dog' 'log' 'mat' 'on' 'sat' 'the']
# [[1 0 0 1 1 1 2]
#  [0 1 1 0 1 1 2]]
```

BoW's weakness: frequent but uninformative words (`the`, `on`) dominate the vector magnitude, and it can't distinguish a word that appears in every document (uninformative) from one that appears rarely but is highly discriminative.

**TF-IDF (Term Frequency–Inverse Document Frequency)** reweights raw counts to downweight ubiquitous terms and upweight discriminative ones.

```
TF(t, d)  = count(t in d) / total_terms(d)              [or just raw count(t, d)]
IDF(t)    = log( N / (1 + df(t)) )                       [df(t) = # docs containing t, N = total docs]
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

Intuition: a term that appears often in *this* document (high TF) but rarely across *all* documents (high IDF, since `df(t)` is small) is a strong signal for that document's topic. A term appearing in every document (`df(t) = N`) gets `IDF ≈ log(1) = 0`, nulling its contribution.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())
print(X_tfidf.toarray().round(3))
```

TF-IDF remains a strong, cheap, interpretable baseline for classification/retrieval and is still widely used in production for its speed and lack of training data requirements (no embeddings to train).

### N-grams and Tradeoffs

An **n-gram** is a contiguous sequence of `n` tokens. Unigrams (`n=1`) are BoW; bigrams (`n=2`) capture short-range word pairs (`"not good"`, `"New York"`); trigrams and beyond capture more context but multiply the feature space combinatorially.

```python
from sklearn.feature_extraction.text import CountVectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
X = bigram_vectorizer.fit_transform(["not good", "very good"])
print(bigram_vectorizer.get_feature_names_out())
# ['good' 'not' 'not good' 'very' 'very good']
```

| N | Captures | Tradeoff |
|---|---|---|
| Unigram | Individual word presence | Loses all word order/context; "not good" ≈ "good" |
| Bigram | Local word pairs, negation, simple phrases | Vocabulary size grows ~quadratically; sparser features |
| Trigram+ | Longer local patterns | Extreme sparsity; most trigrams seen once (data sparsity), huge feature space, high overfitting risk |

**Tradeoff summary**: higher-order n-grams capture more local syntax at the cost of exponentially growing, extremely sparse feature spaces, which increases overfitting risk and memory/compute cost, and still cannot capture long-range dependencies (that requires sequence models). In practice, `(1,2)` or `(1,3)` n-gram ranges combined with feature selection/hashing are the sweet spot for classical pipelines.

### Word Embeddings: Word2Vec, GloVe, FastText

Word embeddings map each word to a dense, low-dimensional (typically 100–300-d) vector such that geometric relationships encode semantic relationships (`vec("king") - vec("man") + vec("woman") ≈ vec("queen")`).

#### Word2Vec (Mikolov et al., 2013)

Word2Vec learns embeddings via a shallow neural network trained on a *self-supervised* prediction task over a sliding context window, in two architectural variants:

**CBOW (Continuous Bag of Words)**: predict the *center* word from the *average of surrounding context words' embeddings*. Faster to train, works well for frequent words.

```
Objective: maximize P(w_t | w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2})
```

**Skip-gram**: predict *context* words from the *center* word — the inverse direction. Slower but performs better for rare words since each (center, context) pair is an independent training example, giving rare words more gradient updates relative to their frequency.

```
Objective: maximize sum over context positions of P(w_{t+j} | w_t),  j in window, j != 0
```

Both use a shallow architecture: input word → embedding lookup (the matrix we want) → predict output word(s) via a softmax over the *entire vocabulary*. This full softmax is computationally prohibitive for vocabularies of hundreds of thousands of words (cost scales with `V` per training step), which motivates:

**Negative Sampling**: reframe the prediction as binary classification: "is `(center, context)` a real pair observed in the corpus, or not?" For each true (positive) pair, sample `k` random *negative* words (not the true context) from a noise distribution — commonly a unigram distribution raised to the 3/4 power (`P(w) ∝ freq(w)^0.75`), which smooths out the raw frequency skew so that very common words aren't sampled disproportionately more, while very rare words still get more negatives than pure uniform sampling.

```
Loss for one positive pair (w, c) with k negative samples c_1..c_k:
L = -log σ(v_w · v_c) - Σ_{i=1}^{k} log σ(-v_w · v_{c_i})

where σ is the sigmoid function, v_w and v_c are the "input" and "output" embedding vectors.
```

This turns an O(V) softmax into an O(k) computation per step (typically k = 5–20), making training on billion-word corpora tractable.

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "sat", "on", "the", "log"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1,
                  sg=1,              # 1 = skip-gram, 0 = CBOW
                  negative=10,       # negative sampling count
                  epochs=50)
print(model.wv.most_similar("cat", topn=3))
```

#### GloVe (Global Vectors, Pennington et al., 2014)

GloVe takes a different, matrix-factorization-flavored approach: it first builds a global **word-word co-occurrence matrix** `X` over the entire corpus (`X_ij` = how often word `j` appears in word `i`'s context window, aggregated over the whole corpus), then learns embeddings such that the dot product of two word vectors approximates the **log of their co-occurrence probability ratio**:

```
w_i · w_j + b_i + b_j ≈ log(X_ij)
```

The training objective is a weighted least-squares regression over all observed (nonzero) co-occurrence pairs, with a weighting function `f(X_ij)` that down-weights extremely frequent co-occurrences (to avoid stopword pairs dominating) and caps the influence of very rare/noisy pairs:

```
J = Σ_{i,j} f(X_ij) * (w_i · w̃_j + b_i + b̃_j - log(X_ij))^2
```

**Word2Vec vs GloVe**: Word2Vec is a *predictive, local-context, online* method (streams through text with a sliding window, one training example at a time); GloVe is a *count-based, global-statistics* method (builds the full co-occurrence matrix upfront, then factorizes it). Empirically they produce embeddings of similar quality; GloVe explicitly leverages global corpus statistics which can help with rarer co-occurrence patterns, while Word2Vec scales more easily to very large streaming corpora without holding a huge co-occurrence matrix in memory.

#### FastText (Bojanowski et al., 2017, Facebook)

FastText extends skip-gram Word2Vec by representing each word as the **sum of embeddings of its character n-grams** (typically n = 3 to 6) plus a whole-word vector, rather than a single atomic vector per word.

```
vec("apple") = vec(<ap) + vec(app) + vec(ppl) + vec(ple) + vec(le>) + ... + vec(<apple>)
```

(`<` and `>` mark word boundaries so that n-grams like `"her"` inside `"where"` don't collide with the standalone word `"her"`.)

**Benefits**: (1) handles OOV words gracefully by composing from known n-grams; (2) captures morphology — related words (`"play"`, `"playing"`, `"played"`) share n-grams and thus have similar vectors even with limited training data; (3) works well for morphologically rich languages (Finnish, Turkish, German compounds) and misspelling-heavy text.

| Method | Training signal | Handles OOV? | Captures morphology? | Granularity |
|---|---|---|---|---|
| Word2Vec | Local context window prediction | No | No | Whole word |
| GloVe | Global co-occurrence statistics | No | No | Whole word |
| FastText | Local context + character n-grams | Yes (composes from n-grams) | Yes | Subword (char n-grams) |

```python
from gensim.models import FastText
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, min_n=3, max_n=6, epochs=50)
print(ft_model.wv["catz"])  # works even though "catz" was never seen — composed from n-grams
```

### Contextual Embeddings: ELMo and the Transition to Transformers

Static embeddings (Word2Vec/GloVe/FastText) assign **one fixed vector per word regardless of context** — `"bank"` gets the same vector in "river bank" and "bank account." This is a fundamental limitation for polysemous words.

**ELMo (Embeddings from Language Models, Peters et al., 2018)** was the pivotal step toward contextual representations: it trains a deep **bidirectional multi-layer LSTM** language model (a forward LSTM predicting the next word, and a separate backward LSTM predicting the previous word, trained jointly but not truly jointly conditioned — this "shallow" bidirectionality is a key contrast with BERT's deep bidirectionality). For a given word in a given sentence, ELMo produces its embedding as a **learned weighted combination of the hidden states from all LSTM layers** (plus the base character-CNN embedding), so the same word gets a *different* vector depending on its sentence context.

```
ELMo_k = γ * Σ_j s_j * h_{k,j}
  where h_{k,j} is the hidden state for token k at layer j (from both directions),
  s_j are softmax-normalized, task-specific learned weights per layer,
  γ is a task-specific scaling factor.
```

ELMo embeddings were typically used as *additional input features* concatenated with existing task-specific architectures (a "feature-based" approach), rather than the whole model being fine-tuned end-to-end.

**Transition to transformers**: ELMo's LSTMs process sequences step-by-step, which is (a) slow to train (no parallelization across time steps) and (b) limited in how far back context can effectively propagate (vanishing gradients over long sequences, even with LSTM gating). The **Transformer** architecture (Vaswani et al., 2017) replaced recurrence with **self-attention**, allowing every token to directly attend to every other token in the sequence in parallel, regardless of distance. This enabled BERT/GPT-style models to build much deeper, more effective bidirectional (or unidirectional) contextual representations, trained on far larger corpora, far faster (parallelizable), leading to the current transformer-dominated era of NLP. (Full transformer architecture — multi-head attention, positional encoding, etc. — is covered in depth in the Generative AI/LLM syllabus file; here the key point is *why* NLP representation learning moved from static → shallow-contextual (ELMo) → deep-contextual (transformer) embeddings.)

| | Static (Word2Vec/GloVe/FastText) | Shallow contextual (ELMo) | Deep contextual (BERT/GPT) |
|---|---|---|---|
| Vector per word | Fixed, one per word | Context-dependent | Context-dependent |
| Architecture | Shallow log-linear / matrix factorization | Bi-LSTM | Transformer self-attention |
| Parallelizable training | Yes (trivially) | No (sequential recurrence) | Yes (attention is parallel over sequence) |
| Bidirectionality | N/A | Shallow concat of independent fwd/bwd LSTMs | Deep joint bidirectional (BERT) or unidirectional (GPT) |
| Typical usage pattern | Feature input | Feature input | Fine-tuning / prompting the whole model |

### Similarity Measures: Cosine, Jaccard, Edit Distance

**Cosine similarity** measures the angle between two vectors, ignoring magnitude — the standard metric for comparing dense embeddings:

```
cos_sim(A, B) = (A · B) / (||A|| * ||B||)
```

Ranges from -1 (opposite) to 1 (identical direction); for typical NLP embeddings values are usually positive. It's magnitude-invariant, which matters because embedding norm often correlates with word frequency rather than meaning, so comparing direction (not scale) gives a cleaner semantic comparison.

```python
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Jaccard similarity** measures set overlap — useful for comparing token sets, n-gram sets, or shingles (e.g., near-duplicate document detection), not embeddings:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

```python
def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0

a = set("the cat sat".split())
b = set("the cat ran".split())
print(jaccard_similarity(a, b))  # {'the','cat'} / {'the','cat','sat','ran'} = 2/4 = 0.5
```

**Edit distance (Levenshtein distance)** counts the minimum number of single-character insertions, deletions, and substitutions needed to transform one string into another — used for spelling correction, fuzzy matching, OCR error correction, and entity matching/deduplication.

```python
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],     # deletion
                                    dp[i][j - 1],     # insertion
                                    dp[i - 1][j - 1]) # substitution
    return dp[m][n]

print(levenshtein("kitten", "sitting"))  # 3
```

| Metric | Compares | Typical use |
|---|---|---|
| Cosine similarity | Dense vectors (direction) | Embedding/semantic similarity, semantic search |
| Jaccard similarity | Sets (token sets, n-gram shingles) | Near-duplicate detection, simple lexical overlap |
| Edit distance | Character sequences | Spelling correction, fuzzy string matching, dedup |

**Pitfalls:**
- Using Euclidean distance on high-dimensional, non-normalized text embeddings without checking whether cosine is more appropriate (very common mistake — most embedding models are trained/evaluated with cosine similarity in mind).
- Jaccard on raw word sets ignores frequency and order entirely; can wrongly rate two documents with wildly different length and content overlap as similar/dissimilar.
- Edit distance is O(m·n) — expensive at scale; for large-scale fuzzy matching, use approximate methods (BK-trees, MinHash/LSH) instead of brute-force pairwise Levenshtein.

### Interview Questions

1. **Q: Derive/explain the TF-IDF formula and explain intuitively why the IDF term uses a logarithm.**
   A: `TF-IDF(t,d) = TF(t,d) * IDF(t)`, where `IDF(t) = log(N / (1+df(t)))`. TF captures local importance (how often the term appears in this document); IDF captures global rarity (down-weighting terms that appear in most documents, since they're less discriminative). The log dampens the effect of `df(t)` so IDF grows slowly as document frequency shrinks — without the log, rare terms in huge corpora would receive disproportionately explosive weight; the log keeps the score on a more usable, roughly additive scale.

2. **Q: Why can BoW/TF-IDF not capture semantic similarity between synonyms like "car" and "automobile"?**
   A: Both are sparse, count-based representations where each word is an independent, orthogonal dimension — there's no mechanism relating two different vocabulary entries. Dense embeddings (Word2Vec/GloVe) instead place semantically similar words close together in vector space because they were trained to predict similar contexts, so "car" and "automobile" end up with similar vectors despite being different tokens.

3. **Q: Compare CBOW and Skip-gram — architecture and when to prefer each.**
   A: CBOW predicts the center word from the averaged embeddings of context words; Skip-gram predicts each context word from the center word. CBOW trains faster (one prediction per window) and works well on frequent words since it smooths over context; Skip-gram treats each (center, context) pair as an independent example, giving rare words more effective training signal, so Skip-gram is generally preferred when good representations for rare words matter, at higher computational cost.

4. **Q: Explain negative sampling in Word2Vec and why it's necessary.**
   A: The original formulation requires a softmax over the entire vocabulary to predict context words, which is computationally infeasible for large vocabularies (cost scales with V per update). Negative sampling reframes training as binary classification: distinguish true (center, context) pairs from k randomly sampled "negative" (center, random word) pairs, using a sigmoid loss. This reduces per-step cost from O(V) to O(k), making large-scale training tractable while still learning useful embeddings.

5. **Q: Why does Word2Vec's negative sampling use unigram frequency raised to the 3/4 power rather than raw frequency or uniform sampling?**
   A: Raw frequency sampling would oversample extremely common words (like "the") as negatives, wasting most negative-sample capacity on uninformative contrasts. Uniform sampling ignores frequency entirely, undersampling common words as useful negatives. Raising frequency to the 0.75 power smooths the distribution — it still favors more frequent words as negatives (which are more informative contrasts) but reduces the extreme skew, empirically found to give the best embedding quality.

6. **Q: How does GloVe's training objective fundamentally differ from Word2Vec's?**
   A: GloVe factorizes a precomputed global word-word co-occurrence matrix via weighted least squares, so the dot product of two vectors approximates the log of their co-occurrence probability ratio, directly using corpus-wide statistics. Word2Vec is predictive and local — it slides a context window through the corpus and updates via gradient steps on local (center, context) pairs, never explicitly building a global co-occurrence matrix.

7. **Q: How does FastText solve the OOV problem that plagues Word2Vec and GloVe?**
   A: FastText represents a word as the sum of embeddings of its character n-grams (plus the whole word), rather than one atomic vector. An unseen word can still be embedded by decomposing it into n-grams that likely appeared in other training words, producing a reasonable vector by composition, whereas Word2Vec/GloVe have literally no vector for a word absent from their training vocabulary.

8. **Q: What is the key architectural limitation of static embeddings that ELMo addresses, and how?**
   A: Static embeddings assign one fixed vector per word type regardless of sentence context, so polysemous words (bank/bank) get identical representations in every context. ELMo trains a deep bidirectional LSTM language model and computes each token's embedding as a learned weighted combination of hidden states across all LSTM layers for that specific sentence, producing a different vector for the same word depending on its surrounding context.

9. **Q: Why is ELMo's bidirectionality called "shallow" compared to BERT's?**
   A: ELMo trains a forward LSTM (predicting left-to-right) and a backward LSTM (predicting right-to-left) largely independently, then concatenates/combines their hidden states. Each direction's LSTM never sees the other direction's context *during* its own computation. BERT instead uses self-attention where every token attends jointly to both left and right context simultaneously within the same computation, yielding deeper, jointly-conditioned bidirectional representations.

10. **Q: When would you choose cosine similarity over Euclidean distance for comparing text embeddings?**
    A: When the direction of the embedding vector encodes meaning but magnitude is confounded by factors like word/document frequency or length, cosine similarity (which normalizes by vector norm) gives a cleaner semantic comparison. Since most embedding models (Word2Vec, sentence embeddings, transformer output vectors) are typically evaluated/used with cosine similarity in mind, it's the conventional and usually correct default choice.

11. **Q: Give an example where Jaccard similarity would fail to reflect true semantic similarity.**
    A: Comparing "The movie was not good" and "The movie was good" — as token sets, these overlap heavily (differing by just "not"), giving high Jaccard similarity, but the sentences have opposite sentiment. Jaccard only measures lexical set overlap, with no notion of word order, negation, or semantics.

12. **Q: How is edit distance used practically in an NLP pipeline, and what's its computational cost?**
    A: It's used for spelling correction (finding the nearest known word to a typo), deduplication/fuzzy matching of entity names or records, and OCR error correction. Standard dynamic-programming Levenshtein distance is O(m·n) time and space for strings of length m and n, which is expensive at scale — production systems use approximate techniques like BK-trees or MinHash/LSH for candidate generation before falling back to exact edit distance.

13. **Q: What's the main practical tradeoff of using high-order n-grams (e.g., trigrams+) as features in a classical text classifier?**
    A: They capture more local word-order information than unigrams but the feature space grows combinatorially with vocabulary size and n, most higher-order n-grams appear only once or a handful of times in the training corpus (severe sparsity), increasing overfitting risk and memory/compute cost without necessarily improving generalization — often requiring feature selection, hashing, or minimum-frequency pruning to be usable.

14. **Q: Why do embedding-based methods generally outperform BoW/TF-IDF on tasks requiring semantic generalization, but TF-IDF often remains competitive for topic-level document classification?**
    A: Semantic generalization tasks (paraphrase detection, semantic search) benefit from embeddings capturing meaning beyond exact word overlap. But topic classification (e.g., "sports" vs "finance" news) is often driven by presence of highly discriminative, topic-specific keywords, which TF-IDF captures directly and interpretably, and its simplicity/speed and lack of training data requirements often make it a strong, hard-to-beat baseline for such tasks.

15. **Q: What information is fundamentally lost when converting a sentence to a Bag-of-Words vector, and what real failure mode does that cause?**
    A: Word order and syntactic structure are completely discarded — "dog bites man" and "man bites dog" produce identical BoW vectors. This causes failure on any task where meaning depends on order/structure: negation scope, semantic roles (who did what to whom), and phrase-level idioms.

---

## Classical NLP Tasks and Models

### Language Modeling: N-gram LMs, Perplexity, Smoothing

A **language model (LM)** assigns a probability to a sequence of words, `P(w_1, w_2, ..., w_n)`, factored via the chain rule:

```
P(w_1...w_n) = P(w_1) * P(w_2|w_1) * P(w_3|w_1,w_2) * ... * P(w_n|w_1...w_{n-1})
```

An **n-gram LM** approximates this with the **Markov assumption**: each word depends only on the previous `n-1` words, not the full history.

```
Bigram (n=2): P(w_i | w_1...w_{i-1}) ≈ P(w_i | w_{i-1})
Trigram (n=3): P(w_i | w_1...w_{i-1}) ≈ P(w_i | w_{i-2}, w_{i-1})
```

Probabilities are estimated via **maximum likelihood** from corpus counts:

```
P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
```

#### Perplexity

**Perplexity** is the standard intrinsic evaluation metric for a language model — the exponentiated average negative log-likelihood per token, interpretable as "the model's average branching factor" (how many roughly-equally-likely word choices it's confused between at each step):

```
PP(W) = exp( -(1/N) * Σ_{i=1}^{N} log P(w_i | w_1...w_{i-1}) )
```

Lower perplexity = better model (more confident, accurate probability assignment to the actual test sequence). A perplexity of 100 roughly means the model is, on average, as uncertain as if choosing uniformly among 100 words at each step. Perplexity is comparable only across models sharing the same vocabulary/tokenization — comparing perplexities across different tokenizers is invalid.

```python
import math

def perplexity(log_probs):
    # log_probs: list of log P(w_i | context) for each token in test sequence, natural log
    n = len(log_probs)
    avg_neg_log_prob = -sum(log_probs) / n
    return math.exp(avg_neg_log_prob)

# toy example: model assigns these probabilities to the actual observed words
probs = [0.2, 0.1, 0.5, 0.3]
log_probs = [math.log(p) for p in probs]
print(perplexity(log_probs))
```

#### Smoothing (handling zero-count n-grams)

MLE assigns **zero probability** to any n-gram never seen in training, which is catastrophic — a single unseen bigram makes the whole sequence probability zero. Smoothing redistributes probability mass from seen to unseen events.

**Laplace (Add-one) smoothing**: add 1 to every count before normalizing.

```
P_Laplace(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)
```

Simple but overcorrects badly — it gives far too much probability mass to unseen events when `V` (vocabulary size) is large, severely underestimating probabilities of frequent n-grams.

**Kneser-Ney smoothing** (the gold standard for n-gram LMs) improves on simpler discounting methods (like Good-Turing or basic backoff) via two key ideas:

1. **Absolute discounting**: subtract a fixed discount `d` (typically 0.1–0.9, estimated from held-out data) from each nonzero count, rather than proportional/add-one schemes, and redistribute the "saved" probability mass to unseen n-grams.
2. **Continuation probability for the lower-order model**: when backing off to a lower-order n-gram (e.g., trigram→bigram→unigram) for a context with insufficient data, Kneser-Ney does *not* use the lower-order model's raw frequency. Instead it uses how many *distinct contexts* the word has appeared as a novel continuation in — i.e., a word like "Francisco" is very frequent but almost always follows "San," so its raw unigram frequency overstates how "generally likely" it is in *new* contexts. Kneser-Ney's continuation probability captures "how many different words does this word appear after" rather than "how often does this word appear," correctly demoting words like "Francisco" that are frequent but contextually narrow.

```
P_KN(w_i | w_{i-1}) = max(count(w_{i-1}, w_i) - d, 0) / count(w_{i-1})
                       + λ(w_{i-1}) * P_continuation(w_i)

where P_continuation(w) ∝ |{w' : count(w', w) > 0}|  (number of distinct preceding words)
```

```python
from collections import defaultdict, Counter

def laplace_bigram_prob(w_prev, w, bigram_counts, unigram_counts, vocab_size):
    return (bigram_counts[(w_prev, w)] + 1) / (unigram_counts[w_prev] + vocab_size)
```

| Smoothing | Idea | Weakness |
|---|---|---|
| Laplace (add-1) | Add 1 to all counts | Overcorrects; wastes too much mass on unseen events with large V |
| Add-k (k<1) | Add smaller constant | Still ad hoc; k needs tuning |
| Good-Turing | Reestimate counts from frequency-of-frequencies | Complex; unstable for high counts |
| Kneser-Ney | Absolute discounting + continuation probability | State of the art for n-gram LMs, but n-gram LMs are still fundamentally limited by fixed context window and data sparsity |

**Note on modern relevance**: n-gram LMs have been superseded by neural LMs (RNN/transformer) for most production use, but perplexity as a metric, the Markov/chain-rule decomposition, and smoothing intuitions (handling rare events) remain foundational vocabulary expected in interviews, and n-gram LMs are still used for lightweight tasks (spelling correction candidate scoring, quick baselines, speech recognition language model rescoring in resource-constrained settings).

### Sequence Labeling: POS Tagging, NER — HMM/CRF, BIO Scheme

**Sequence labeling** assigns one label per token in a sequence, where labels are interdependent (the label of token `i` is informed by neighboring labels), unlike independent per-token classification.

#### POS Tagging

Assigns each word its part of speech (`NOUN`, `VERB`, `ADJ`, `DET`, ...). Example: `"Time/NOUN flies/VERB like/ADP an/DET arrow/NOUN"`.

#### Named Entity Recognition (NER)

Identifies and classifies spans of text into predefined categories (`PERSON`, `ORG`, `LOC`, `DATE`, ...). Example: `"[Barack Obama]_PERSON visited [Paris]_LOC in [2015]_DATE"`.

#### BIO Tagging Scheme

Since entities can span multiple tokens, NER is framed as per-token classification using the **BIO** (Begin-Inside-Outside) scheme (variants: BIOES adds End/Single):

| Tag | Meaning |
|---|---|
| `B-X` | Beginning of an entity of type X |
| `I-X` | Inside (continuation) of an entity of type X |
| `O` | Outside any entity |

```
Tokens: Barack   Obama    visited  Paris   in      2015
Tags:   B-PER    I-PER    O        B-LOC   O       B-DATE
```

This converts span extraction into a per-token sequence-labeling classification problem solvable by any sequence model.

#### HMM (Hidden Markov Model) for Sequence Labeling

An HMM models the *joint* probability of a label sequence and the observed word sequence, assuming:
- **Markov assumption on labels**: `P(tag_i | tag_1...tag_{i-1}) ≈ P(tag_i | tag_{i-1})` (transition probabilities).
- **Output independence**: each word depends only on its own tag: `P(word_i | tag_i)` (emission probabilities).

```
P(tags, words) = Π_i P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Trained via counting (MLE) from labeled data; decoding (finding the most likely tag sequence for a new sentence) uses the **Viterbi algorithm** (dynamic programming, O(N·T²) for sequence length N and T tags), avoiding the exponential cost of checking every possible tag sequence.

**Limitation**: HMMs are *generative* models factoring `P(words, tags)`, and the strong independence assumptions (each word depends only on its own tag; no rich overlapping features) limit expressiveness — you can't easily add features like "is capitalized," "previous word ends in -ing," etc.

#### CRF (Conditional Random Field) for Sequence Labeling

A **linear-chain CRF** is a *discriminative* model that directly models `P(tags | words)` (not the joint), and crucially allows arbitrary, overlapping, hand-engineered **feature functions** over the whole observation sequence at each position, not just the current word:

```
P(tags | words) = (1/Z(words)) * exp( Σ_i Σ_k λ_k * f_k(tag_{i-1}, tag_i, words, i) )
```

where `f_k` are feature functions (e.g., "current word is capitalized AND previous tag is B-PER," "word ends in -tion," "word is in a gazetteer list") and `λ_k` are learned weights; `Z(words)` is a normalization constant (partition function) summing over all possible tag sequences, computed efficiently via the forward-backward algorithm; decoding still uses Viterbi.

**HMM vs CRF:**

| | HMM | CRF |
|---|---|---|
| Model type | Generative (`P(words, tags)`) | Discriminative (`P(tags｜words)`) |
| Features | Fixed (word identity → tag) | Arbitrary, overlapping, engineered features |
| Training | Simple MLE counting | Gradient-based (maximize conditional log-likelihood) |
| Typical performance | Lower (limited feature use) | Higher (rich features), was long the SOTA for NER pre-neural |
| Decoding | Viterbi | Viterbi |

Before transformers, the strongest classical NER systems were **BiLSTM-CRF**: a bidirectional LSTM produces contextual features per token, and a CRF layer on top models label transition dependencies (e.g., enforcing that `I-PER` cannot follow `O` directly), combining neural feature learning with structured output consistency.

```python
# Conceptual sklearn-crfsuite feature function example for NER
def word_features(sent, i):
    word = sent[i]
    return {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word[-3:]': word[-3:],
        'prev_word': sent[i-1] if i > 0 else '<START>',
    }
```

### Syntactic Parsing: Dependency and Constituency Parsing

Parsing recovers the grammatical structure of a sentence — structure that some NLP tasks (semantic role labeling, relation extraction, grammar checking, some feature-engineering pipelines) still consume directly, even though most modern systems no longer build an explicit parse tree before running a transformer end-to-end.

#### Constituency Parsing

**Constituency (phrase-structure) parsing** recursively groups words into nested phrases (constituents) — `NP` (noun phrase), `VP` (verb phrase), `PP` (prepositional phrase) — using a context-free grammar (CFG), producing a tree whose leaves are words and whose internal nodes are phrase labels.

```
"The cat sat on the mat"

              S
        ______|______
       |             VP
       NP        _____|_____
   ____|____    |           PP
  DET      N   V       _____|_____
  |        |   |      P           NP
 The      cat sat     on      ____|____
                              DET      N
                              |        |
                             the      mat
```

Classical parsers (e.g., PCFG chart parsers using the **CKY algorithm**, a dynamic-programming algorithm analogous to Viterbi but for tree structures, O(n³·|grammar|)) find the highest-probability parse under a probabilistic context-free grammar learned from a treebank (e.g., the Penn Treebank). Modern neural constituency parsers (e.g., chart parsers with a neural scoring function, or sequence-to-sequence parsers that linearize the tree) outperform PCFG-only parsers, but the underlying phrase-structure formalism is unchanged.

#### Dependency Parsing

**Dependency parsing** instead represents grammar as a tree of directed, labeled binary relations between words — each word (except the root) has exactly one **head** it depends on, and the relation is labeled with its grammatical role (`nsubj`, `dobj`, `amod`, `prep`, ...). There is no intermediate phrase-node layer; every node in the tree is a word.

```
Arcs: sat -> cat  (nsubj)      "sat" is the ROOT
      sat -> on   (prep)
      on  -> mat  (pobj)
      mat -> the  (det)
      cat -> The  (det)
```

Dependency trees are generally considered more directly useful for downstream tasks (relation extraction, information extraction) because grammatical-relation labels map fairly directly onto semantic roles ("who did what to whom"), without needing to traverse a phrase-structure tree to extract that information.

**Parsing algorithms:**
- **Transition-based (shift-reduce) parsing**: process the sentence left to right while maintaining a stack and buffer; at each step a classifier (historically an SVM, now typically a small neural network, e.g., the classic "arc-standard" parser of Chen & Manning, 2014) predicts an action — `SHIFT` (move next buffer word to stack), `LEFT-ARC`/`RIGHT-ARC` (pop the stack, attach a dependency, label it). Greedy and fast (linear time), but locally greedy, like greedy decoding elsewhere in NLP.
- **Graph-based parsing**: score every possible head-dependent arc, then find the highest-scoring valid tree via a maximum spanning tree algorithm (e.g., Chu-Liu/Edmonds' algorithm for non-projective trees, or Eisner's algorithm for projective trees). Globally optimal given the arc scores, at higher computational cost than transition-based parsing.

```python
import spacy

nlp = spacy.load("en_core_web_sm")   # ships a neural transition-based dependency parser
doc = nlp("The cat sat on the mat")
for token in doc:
    print(token.text, token.dep_, token.head.text)
# The       det      cat
# cat       nsubj    sat
# sat       ROOT     sat
# on        prep     sat
# the       det      mat
# mat       pobj     on
```

**Projective vs non-projective**: a dependency tree is *projective* if its arcs can be drawn above the sentence without crossing lines (equivalent to every subtree spanning a contiguous substring). English is mostly projective; languages with more flexible word order (e.g., Czech, Dutch) frequently require non-projective parsing, which is why algorithm choice (Eisner's — projective only — vs Chu-Liu/Edmonds — handles non-projective) matters cross-lingually.

**Pitfalls:**
- Classic PP-attachment ambiguity ("I saw the man with a telescope" — does "with a telescope" attach to "saw" or "the man"?) is a canonical example of genuine structural ambiguity that pure syntax cannot always resolve without semantic/world knowledge.
- Parsing accuracy is usually reported per-arc (UAS — unlabeled attachment score — and LAS — labeled attachment score), not per-sentence exact match, since getting every single arc right in a long sentence is rare.
- Dependency/constituency parses are largely absent from modern end-to-end transformer pipelines (self-attention implicitly captures some syntactic structure without an explicit parse), but explicit parses remain useful for feature engineering, grammar/style-checking tools, and structured information extraction where interpretability/auditability of the intermediate representation matters.

### Word Sense Disambiguation (WSD)

**Word Sense Disambiguation** is the task of determining which sense of a polysemous word is intended in a given context — e.g., does "bank" in "I deposited money at the bank" mean a financial institution or a riverbank? This is the flip side of the polysemy limitation raised earlier for static embeddings (Word2Vec/GloVe assign one vector per word regardless of sense).

**Sense inventories**: WSD typically disambiguates against a fixed sense inventory such as **WordNet synsets** (a synset is a set of synonymous senses; "bank" has multiple synsets, one per sense).

**Approaches:**
1. **Knowledge-based — Lesk algorithm**: compute the word-overlap between the context sentence and each candidate sense's dictionary gloss (definition) in WordNet; pick the sense whose gloss overlaps most with the context. Simple, requires no training data, but brittle — glosses are short, so overlap is often sparse.
   ```
   sense* = argmax_s |context_words ∩ gloss_words(s)|
   ```
2. **Supervised classification**: treat WSD as per-instance classification — features include surrounding words, POS tags, and syntactic context; trained on sense-tagged corpora (e.g., SemCor). Accurate but requires expensive sense-annotated training data per word, and struggles with senses/words unseen in training (the "knowledge acquisition bottleneck").
3. **Contextual embeddings (modern default)**: a word's contextual embedding from BERT/GPT-style models is already implicitly sense-disambiguated — "bank" in a financial-context sentence and a river-context sentence gets *different* vectors purely as a side effect of self-attention over context, with no dedicated WSD component. Explicit WSD as a standalone task/model has consequently become far less central in modern NLP pipelines, but the underlying evaluation task (e.g., nearest-neighbor sense matching using contextual embeddings against sense-tagged examples) is still used to probe whether a model's representations genuinely separate senses.

```python
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

sentence = "I deposited money at the bank"
sense = lesk(word_tokenize(sentence), "bank", "n")
print(sense, sense.definition())
# Synset('bank.n.01') / financial-institution sense (result depends on NLTK/WordNet version)
```

**Pitfalls:**
- Sense granularity is a modeling choice — WordNet's sense distinctions are often finer than what any downstream task actually needs (e.g., distinguishing "bank" as a slope of a river vs. an edge of a river as separate senses is rarely useful), leading many practical systems to coarsen the sense inventory.
- WSD evaluation datasets (SensEval/SemEval WSD tasks) are small relative to modern pretraining corpora, so supervised WSD models often don't generalize well beyond their training distribution.
- Confusing WSD with simple polysemy-blind embedding lookup is a common conceptual error — static embeddings *cannot* do WSD at all (one vector per word type), which is a frequently probed interview distinction.

### Text Classification Pipelines End-to-End

A typical classical pipeline for sentiment analysis / spam detection:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# 1. Load & clean data
df = pd.DataFrame({
    "text": ["free money now click here", "hey, are we still meeting today?", "win a free prize!!!", "let's grab lunch tomorrow"],
    "label": [1, 0, 1, 0]  # 1 = spam, 0 = ham
})

# 2. Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Build pipeline: TF-IDF -> classifier (fit only on train to avoid leakage)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
])

# 4. Fit and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

**End-to-end considerations:**
- **Data leakage**: fit the vectorizer (vocabulary, IDF statistics) only on training data, never on the full dataset before splitting.
- **Class imbalance**: spam/sentiment datasets are often imbalanced; use `class_weight="balanced"`, resampling (SMOTE for text is tricky — usually oversample/undersample at the document level), or threshold tuning; always report precision/recall/F1 per class, not just accuracy.
- **Feature engineering beyond TF-IDF**: message length, punctuation counts, presence of URLs/phone numbers (spam), capitalization ratio.
- **Calibration**: if downstream decisions use predicted probabilities (not just labels), check calibration (Platt scaling / isotonic regression) since raw classifier scores aren't always well-calibrated probabilities.
- **Baseline first**: always start with TF-IDF + Logistic Regression/Linear SVM before reaching for a fine-tuned transformer — it's fast, interpretable (inspect top weighted features), and often surprisingly competitive.

### Topic Modeling: LDA and NMF

Topic models discover latent thematic structure in a document collection *unsupervised* — no labels needed.

#### LDA (Latent Dirichlet Allocation)

LDA is a generative probabilistic model assuming each document is a mixture of topics, and each topic is a distribution over words. The generative story (how LDA assumes documents were "created"):

1. For each topic `k`, draw a word distribution `φ_k ~ Dirichlet(β)` (a distribution over the vocabulary).
2. For each document `d`, draw a topic distribution `θ_d ~ Dirichlet(α)` (a distribution over topics).
3. For each word position in document `d`:
   a. Draw a topic assignment `z ~ Categorical(θ_d)`.
   b. Draw the observed word `w ~ Categorical(φ_z)`.

The Dirichlet distribution is used as the prior because it's the conjugate prior for the categorical/multinomial distribution, making inference tractable; the hyperparameters `α` (document-topic concentration) and `β` (topic-word concentration) control how sparse/concentrated the mixtures are — low `α` means each document is dominated by few topics, low `β` means each topic is dominated by few words.

Inference (since we only observe words, not topics) is done via **Variational Bayes** or **Gibbs Sampling**, estimating the posterior distributions over `θ` and `φ` that best explain the observed corpus.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = ["cat dog pet animal", "stock market economy finance", "dog cat puppy kitten", "finance bank investment stock"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=2, random_state=42, max_iter=50)
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
    print(f"Topic {topic_idx}: {top_words}")
```

**Choosing K (number of topics)**: use coherence scores (e.g., `C_v` or UMass coherence), perplexity on held-out documents, or human judgment via topic inspection; there's no single "correct" K, it's a modeling choice balancing granularity and interpretability.

#### NMF (Non-negative Matrix Factorization)

NMF is a purely linear-algebra (not probabilistic/generative) approach: given the non-negative document-term matrix `X` (e.g., TF-IDF matrix, `documents × vocabulary`), factorize it into two low-rank non-negative matrices:

```
X ≈ W · H
X: (n_docs × n_words),  W: (n_docs × k topics),  H: (k topics × n_words)
```

subject to all entries of `W` and `H` being non-negative, solved by iterative optimization (multiplicative update rules or coordinate descent) minimizing reconstruction error (Frobenius norm or KL divergence). Non-negativity is what makes the factors interpretable as "topics" (additive combinations, no cancellation via negative weights) — this is directly analogous to `θ` (document-topic weights) and `φ` (topic-word weights) in LDA, but estimated by matrix factorization rather than probabilistic inference.

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
nmf = NMF(n_components=2, random_state=42, max_iter=500)
nmf.fit(X_tfidf)
feature_names = tfidf.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
    print(f"Topic {topic_idx}: {top_words}")
```

**LDA vs NMF:**

| | LDA | NMF |
|---|---|---|
| Nature | Probabilistic generative model | Linear algebra / matrix factorization |
| Input | Raw counts (multinomial assumption) | Usually TF-IDF (though counts work too) |
| Output | Probability distributions (topics sum to 1) | Non-negative weights (not necessarily normalized probabilities) |
| Interpretability | Strong theoretical grounding, probabilistic topic mixtures | Often sharper, more distinct topics empirically on TF-IDF |
| Speed | Slower (sampling/variational inference) | Faster (deterministic optimization) |
| Determinism | Stochastic (depends on sampling/init) | Deterministic given fixed init (mostly) |

**Pitfalls:**
- Topic models are sensitive to preprocessing: failing to remove stopwords/very frequent or very rare words leads to garbage "topics" dominated by function words.
- Number of topics K is a hyperparameter with no ground truth — over-specifying K produces redundant/fragmented topics, under-specifying merges distinct themes.
- LDA assumes bag-of-words (no word order) — cannot capture phrase-level or contextual meaning; modern alternatives use embeddings (e.g., BERTopic, clustering contextual embeddings) for richer topic discovery.

### Interview Questions

1. **Q: Derive perplexity from cross-entropy and explain what a perplexity of 50 intuitively means.**
   A: Perplexity is `exp(H)` where `H = -(1/N) Σ log P(w_i | context)` is the average per-token cross-entropy (negative log-likelihood) of the model on the test sequence. A perplexity of 50 means the model is, on average, as uncertain about the next word as if it had to choose uniformly among 50 equally likely candidates at each position — lower is better, and perplexity is only comparable between models using the same vocabulary/tokenization.

2. **Q: Why does raw MLE estimation fail for n-gram language models, and what problem does smoothing solve?**
   A: MLE assigns zero probability to any n-gram unseen in training data, and since sentence probability is a product of n-gram probabilities, a single unseen n-gram zeroes out the entire sequence probability — even for otherwise fluent sentences. Smoothing redistributes some probability mass from seen to unseen events so that no n-gram ever gets exactly zero probability, making the model robust to novel (but valid) word combinations at test time.

3. **Q: Why is Kneser-Ney smoothing generally superior to Laplace (add-one) smoothing?**
   A: Laplace smoothing adds a large uniform amount of probability mass to every possible n-gram (proportional to vocabulary size V), massively overcorrecting and underestimating the probability of frequent n-grams when V is large. Kneser-Ney uses absolute discounting (subtracting a small fixed amount, tuned from held-out data) and, critically, backs off to lower-order models using continuation probability — how many distinct contexts a word appears in — rather than raw frequency, correctly demoting words that are frequent but only in narrow contexts (like "Francisco" after "San").

4. **Q: What is the Markov assumption in n-gram language modeling, and what's its main limitation?**
   A: It assumes the probability of the next word depends only on the previous `n-1` words, not the entire history, which makes probability estimation tractable via simple counting. Its limitation is an inability to capture long-range dependencies (e.g., subject-verb agreement or coreference spanning many words back), which is a core reason neural (RNN/transformer) language models, with effectively unbounded or much longer context, outperform n-gram LMs.

5. **Q: Explain the BIO tagging scheme and why NER is framed this way instead of directly predicting spans.**
   A: BIO assigns each token a tag: `B-X` for the beginning of an entity of type X, `I-X` for tokens inside a continuing entity, and `O` for tokens outside any entity. This converts NER — a variable-length span-extraction problem — into a fixed per-token classification problem, which is directly solvable with standard sequence labeling models (HMM, CRF, BiLSTM, transformer token classifiers) without needing to search over all possible spans.

6. **Q: What are the two core independence assumptions of an HMM used for POS tagging, and how does a CRF relax them?**
   A: HMMs assume (1) the Markov property on tags — each tag depends only on the immediately preceding tag — and (2) output independence — each word depends only on its own tag, given the tag. CRFs relax the feature limitation: rather than being restricted to word-identity emissions, a CRF is discriminative and can use arbitrary, overlapping feature functions of the entire observed sequence (capitalization, suffixes, gazetteer membership, neighboring words) at each position, generally yielding higher accuracy.

7. **Q: Why is the Viterbi algorithm needed for HMM/CRF decoding instead of just picking each token's most likely tag independently?**
   A: Naively picking the highest-probability tag per token independently ignores the sequential dependencies between adjacent tags (e.g., "I-PER" cannot validly follow "O"), and can produce inconsistent tag sequences. Viterbi is a dynamic programming algorithm that finds the single most probable *entire* tag sequence jointly, in O(N·T²) time (N = sequence length, T = number of tags), avoiding the exponential cost of enumerating all T^N possible sequences.

8. **Q: In a spam classification pipeline using TF-IDF + Logistic Regression, where exactly would data leakage occur if done incorrectly, and how do you prevent it?**
   A: Leakage occurs if the TF-IDF vectorizer (which learns vocabulary and document-frequency/IDF statistics) is fit on the entire dataset (including test data) before the train/test split, letting statistics from test documents influence feature weighting seen during training. Prevention: always split data first, then fit the vectorizer only on the training set, and use `.transform()` (not `.fit_transform()`) on the test set.

9. **Q: Why is accuracy often a poor metric for spam/sentiment classifiers, and what should be reported instead?**
   A: These datasets are frequently class-imbalanced (e.g., mostly ham, little spam), so a trivial always-predict-majority-class model achieves high accuracy while being useless. Precision, recall, and F1 per class (and possibly a confusion matrix, or ROC-AUC/PR-AUC for probabilistic outputs) give a much more honest picture of performance on the minority class that usually matters most.

10. **Q: Explain the generative story of LDA — how does it imagine a document was created?**
    A: For each topic, a word distribution is drawn from a Dirichlet prior; for each document, a topic-mixture distribution is drawn from another Dirichlet prior. Then, for every word position in the document, a topic is sampled from the document's topic mixture, and a word is sampled from that topic's word distribution. LDA inference works backward from the observed words to estimate the most plausible topic-word and document-topic distributions that could have generated the corpus.

11. **Q: Why use a Dirichlet distribution specifically as the prior in LDA rather than some other distribution?**
    A: The Dirichlet distribution is the conjugate prior to the categorical/multinomial distribution (which governs drawing a topic or a word), which makes the posterior computation tractable (or at least far more efficient to approximate via variational methods or Gibbs sampling) than it would be with a non-conjugate prior.

12. **Q: What is the fundamental methodological difference between LDA and NMF for topic modeling?**
    A: LDA is a probabilistic generative model estimating latent probability distributions (document-topic and topic-word) via Bayesian inference (variational or sampling methods) over a multinomial word-generation process. NMF is a deterministic linear algebra technique that directly factorizes the document-term matrix into two non-negative low-rank matrices via numerical optimization (e.g., minimizing reconstruction error), with no explicit probabilistic generative story.

13. **Q: How would you choose the number of topics K for an LDA model in practice?**
    A: There's no single ground-truth answer; practitioners combine quantitative signals — topic coherence scores (e.g., UMass or C_v coherence, which measure how semantically related the top words in each topic are) and held-out perplexity — with qualitative human inspection of the resulting topics for interpretability and business relevance, often sweeping across several K values and comparing.

14. **Q: Why can BiLSTM-CRF outperform a plain BiLSTM (without CRF) for sequence labeling tasks like NER?**
    A: A plain BiLSTM with a softmax output classifies each token's tag independently given its contextual hidden state, so it has no explicit mechanism to enforce valid tag transitions (e.g., preventing `I-PER` from following `O`). Adding a CRF layer on top models the transition structure between adjacent labels explicitly, jointly decoding the most probable *consistent* tag sequence via Viterbi, which typically improves both accuracy and structural validity of predicted entity spans.

15. **Q: What's a major limitation shared by both LDA and NMF as topic modeling techniques, and how have modern approaches addressed it?**
    A: Both operate on bag-of-words representations, completely discarding word order and any notion of contextual/semantic meaning beyond co-occurrence statistics, so they cannot distinguish polysemous usage or capture nuanced semantic relationships between topically related but lexically different words. Modern approaches (e.g., BERTopic, clustering of contextual sentence/document embeddings) replace the BoW input with dense contextual embeddings before clustering/dimensionality reduction, capturing richer semantic structure.

16. **Q: What's the fundamental structural difference between a constituency parse and a dependency parse of the same sentence?**
    A: A constituency parse builds a tree of nested phrase-structure nodes (NP, VP, PP) over a context-free grammar, where only the leaves are actual words. A dependency parse has no intermediate phrase layer — every node is a word, connected to exactly one head word via a labeled grammatical relation (nsubj, dobj, ...), directly encoding "who depends on whom" without needing to traverse phrase nodes to extract that information.

17. **Q: Compare transition-based and graph-based dependency parsing.**
    A: Transition-based parsing processes the sentence left to right with a stack/buffer, using a classifier to greedily predict shift/left-arc/right-arc actions — fast (linear time) but locally greedy, so early mistakes can propagate. Graph-based parsing scores every candidate head-dependent arc and finds the globally highest-scoring valid tree via a maximum spanning tree algorithm (Chu-Liu/Edmonds for non-projective trees, Eisner's algorithm for projective trees) — more accurate in principle but more expensive.

18. **Q: What does it mean for a dependency parse to be "non-projective," and why does it matter cross-lingually?**
    A: A dependency tree is projective if its arcs can be drawn above the sentence without crossing (every subtree spans a contiguous substring); non-projective trees have crossing arcs, arising from more flexible word order. English is largely projective, but languages like Czech or Dutch frequently require non-projective parses, so parsers built only on projective algorithms (e.g., Eisner's) fail on those languages, requiring the more general (but costlier) Chu-Liu/Edmonds algorithm.

19. **Q: What is the classic PP-attachment ambiguity, and why does it illustrate a limit of pure syntactic parsing?**
    A: In a sentence like "I saw the man with a telescope," the prepositional phrase "with a telescope" can syntactically attach either to the verb ("saw ... with a telescope," meaning I used the telescope) or to the noun ("the man with a telescope," meaning the man had the telescope) — both are grammatically valid parses. Resolving which is intended requires semantic or world knowledge beyond what syntax alone can determine, illustrating that a correct parse doesn't guarantee a correct interpretation.

20. **Q: Explain the Lesk algorithm for word sense disambiguation and its main weakness.**
    A: The Lesk algorithm picks the WordNet sense whose dictionary gloss (definition) has the highest word overlap with the surrounding context sentence — a simple, training-free knowledge-based heuristic. Its main weakness is sparsity: dictionary glosses are short, so genuine overlap is often minimal or zero even for the correct sense, making the algorithm brittle compared to supervised or contextual-embedding-based approaches.

21. **Q: Why has dedicated word sense disambiguation become less central in modern NLP pipelines that use transformer models?**
    A: A contextual embedding from BERT/GPT-style self-attention already produces a different vector for the same word depending on its surrounding context, implicitly separating word senses as a side effect of how the representation is computed — with no dedicated WSD component or sense inventory needed. This is in sharp contrast to static embeddings (Word2Vec/GloVe), which assign one fixed vector per word regardless of sense and thus cannot disambiguate at all.

22. **Q: Why can static word embeddings never perform true word sense disambiguation, even in principle?**
    A: Static embeddings (Word2Vec, GloVe, FastText's whole-word component) assign exactly one vector per word type, computed once during training and looked up identically regardless of the sentence it appears in — there is no mechanism for the same word to produce different vectors in different contexts. WSD fundamentally requires context-dependent representations, which only contextual models (ELMo, BERT, GPT) can provide.

---

## Sequence-to-Sequence and Neural NLP

### RNN/LSTM-based NLP Models

**Recurrent Neural Networks (RNNs)** process sequences one token at a time, maintaining a hidden state `h_t` that summarizes all previous tokens:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

This recurrence lets RNNs handle variable-length sequences and, in principle, arbitrarily long context. In practice, **vanilla RNNs suffer from vanishing/exploding gradients**: because the same weight matrix is applied repeatedly through backpropagation-through-time, gradients shrink (or blow up) exponentially with sequence length, making it hard to learn dependencies spanning more than a few dozen tokens.

**LSTM (Long Short-Term Memory)** solves this with a gating mechanism that controls information flow explicitly, maintaining both a hidden state `h_t` and a cell state `C_t`:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        # input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        # output gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)     # candidate cell state
C_t = f_t * C_{t-1} + i_t * C̃_t            # updated cell state (additive!)
h_t = o_t * tanh(C_t)                      # updated hidden state
```

The key insight: the cell state update is **additive** (`f_t * C_{t-1} + i_t * C̃_t`) rather than a repeated multiplicative transformation, so gradients can flow through many time steps largely unimpeded when the forget gate `f_t` stays close to 1, mitigating vanishing gradients. **GRU (Gated Recurrent Unit)** is a simplified variant merging the forget/input gates into a single "update gate" and eliminating the separate cell state, achieving similar performance with fewer parameters.

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)               # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(embedded)    # h_n: (2, batch, hidden_dim) for bidirectional
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # concat fwd + bwd final states
        return self.fc(final_hidden)
```

### Encoder-Decoder for Translation/Summarization

The **encoder-decoder (seq2seq)** architecture (Sutskever et al., 2014) handles tasks where input and output are both sequences of *different, variable* lengths (translation, summarization, dialogue generation):

- **Encoder**: an RNN/LSTM processes the entire input sequence and compresses it into a fixed-size **context vector** (typically the final hidden state).
- **Decoder**: another RNN/LSTM, initialized with the encoder's context vector, generates the output sequence one token at a time, feeding each generated token back as input for the next step (autoregressive generation), until an `<eos>` token is produced.

```
Encoder: h_1, h_2, ..., h_T = RNN_enc(x_1, x_2, ..., x_T)
Context vector: c = h_T  (or some function of all h_i)
Decoder: s_0 = c;  s_t = RNN_dec(s_{t-1}, y_{t-1});  y_t = softmax(W * s_t)
```

**The bottleneck problem**: compressing an entire (possibly long) source sentence into one fixed-size vector `c` is a severe information bottleneck — quality degrades sharply as source sequence length increases, since early-sentence information gets diluted by the time the encoder reaches the end. This directly motivated attention (next subtopic).

### Attention in Seq2Seq: Bahdanau and Luong Attention

Instead of forcing the decoder to rely on a single fixed context vector, **attention** lets the decoder, at each output step, look back at *all* encoder hidden states and compute a weighted combination — dynamically deciding which parts of the source sentence are most relevant for generating the current output token.

#### Bahdanau Attention (Additive, 2015)

```
e_{t,i} = v^T * tanh(W_1 * s_{t-1} + W_2 * h_i)      # alignment score (additive/MLP-based)
α_{t,i} = softmax_i(e_{t,i})                          # attention weights (sum to 1 over i)
c_t = Σ_i α_{t,i} * h_i                               # context vector, weighted sum of encoder states
s_t = RNN_dec(s_{t-1}, [y_{t-1}; c_t])                # decoder uses previous hidden state + new context
```

Bahdanau attention uses the decoder's *previous* hidden state `s_{t-1}` (before updating) to compute alignment scores via a small feedforward network (additive combination), and feeds the resulting context vector as *additional input* to the decoder's recurrence.

#### Luong Attention (Multiplicative, 2015)

```
e_{t,i} = s_t^T * W * h_i      # (general/multiplicative form; also has "dot" and "concat" variants)
α_{t,i} = softmax_i(e_{t,i})
c_t = Σ_i α_{t,i} * h_i
s̃_t = tanh(W_c * [s_t; c_t])  # context combined AFTER the decoder's own hidden state update
```

Luong attention uses the decoder's *current* (already updated) hidden state `s_t` to compute alignment scores, typically via a simpler dot-product or bilinear form, and combines context *after* the RNN step rather than feeding it in as input.

| | Bahdanau (additive) | Luong (multiplicative) |
|---|---|---|
| Score function | Feedforward network: `v^T tanh(W1*s + W2*h)` | Dot/bilinear: `s^T W h` (or simple dot product) |
| Uses decoder state | Previous state `s_{t-1}` | Current state `s_t` |
| Context vector used | Fed into decoder RNN input | Combined with output after RNN step |
| Computational cost | Higher (extra feedforward layer) | Lower (simpler score functions) |

**Significance**: attention directly foreshadows the transformer's self-attention mechanism — "attend to all positions, weighted by relevance" is the core idea later generalized into the fully attention-based Transformer architecture (no recurrence at all), which is why understanding Bahdanau/Luong attention is considered essential grounding before studying transformers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_dim), encoder_outputs: (batch, seq_len, hidden_dim)
        decoder_hidden_exp = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden_dim)
        scores = self.v(torch.tanh(self.W1(decoder_hidden_exp) + self.W2(encoder_outputs)))  # (batch, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)  # (batch, hidden_dim)
        return context, attn_weights.squeeze(-1)
```

### Beam Search vs Greedy Decoding

At generation/inference time, the decoder must choose which token to output at each step, and generation is inherently a search problem over an exponentially large space of possible output sequences.

**Greedy decoding**: at each step, pick the single highest-probability token. Fast (O(1) hypotheses maintained) but myopic — a locally optimal choice at step `t` can lock in a globally suboptimal full sequence (e.g., picking a common word early on that forces an awkward continuation).

**Beam search**: maintain the top `k` (beam width) partial sequences ("beams") at each step, ranked by cumulative log-probability, and expand each into all possible next tokens, then prune back down to the top `k` overall. This explores a wider portion of the search space than greedy (k=1 is exactly greedy decoding) without the full exponential cost of exhaustive search.

```
score(sequence) = Σ_t log P(y_t | y_1...y_{t-1}, x)   (sum of log-probs, to avoid numeric underflow
                                                          from multiplying many small probabilities)
```

```python
import torch
import torch.nn.functional as F

def beam_search_step(model_logits_fn, start_token, eos_token, beam_width=3, max_len=20):
    # beams: list of (sequence, cumulative_log_prob)
    beams = [([start_token], 0.0)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos_token:
                completed.append((seq, score))
                continue
            logits = model_logits_fn(seq)              # (vocab_size,)
            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                candidates.append((seq + [tid], score + lp))
        if not candidates:
            break
        # keep top beam_width overall
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    completed.extend(beams)
    completed.sort(key=lambda x: x[1], reverse=True)
    return completed[0]
```

**Length bias and length normalization**: because each additional token multiplies in another (≤1) probability, raw cumulative log-probability systematically favors *shorter* sequences. Production systems apply **length normalization** (dividing by sequence length, often with an exponent `α` < 1, e.g., Google NMT's `score / length^α`) to counteract this bias toward truncated outputs.

| | Greedy | Beam Search |
|---|---|---|
| Hypotheses tracked | 1 | k (beam width) |
| Optimality | Locally greedy, often suboptimal globally | Better approximation to global optimum, still not exhaustive |
| Speed | Fastest | Slower, scales with k |
| Diversity of output | None (deterministic) | Can still produce generic/repetitive text; needs length normalization, coverage penalties, or sampling-based methods (top-k, nucleus/top-p sampling) for more diverse/creative generation |
| Common failure mode | Repetition loops, myopic word choice | Can favor short, "safe," generic outputs without length normalization |

**Pitfalls:**
- Beam search is not guaranteed to find the truly optimal sequence — it's a heuristic pruning of the search space, and larger beam width does not monotonically improve output *quality* for open-ended generation tasks (it can make outputs more generic/bland, a well-documented finding in NMT/summarization literature).
- For open-ended, creative generation (dialogue, story generation), sampling-based decoding (temperature sampling, top-k, nucleus/top-p sampling) is generally preferred over beam search, which tends to produce repetitive, low-diversity text in those settings.
- Forgetting to apply length normalization in beam search systematically biases toward premature `<eos>` generation and truncated outputs.

### Interview Questions

1. **Q: Why do vanilla RNNs struggle with long sequences, and how does LSTM's design specifically address this?**
   A: Vanilla RNNs backpropagate gradients through repeated multiplication by the same weight matrix at every time step, causing gradients to vanish (or explode) exponentially with sequence length, so long-range dependencies can't be learned effectively. LSTMs introduce gates (forget/input/output) and an additive cell-state update (`C_t = f_t * C_{t-1} + i_t * C̃_t`) rather than a purely multiplicative transformation, allowing gradients to flow across many time steps largely unimpeded when the forget gate stays near 1.

2. **Q: What is the "bottleneck problem" in basic encoder-decoder seq2seq models, and what motivated attention?**
   A: The encoder compresses the entire variable-length input sequence into a single fixed-size context vector (typically the final hidden state), which becomes an increasingly severe information bottleneck as input length grows — early tokens' information gets diluted/overwritten by the time encoding finishes. Attention was introduced to let the decoder access *all* encoder hidden states directly at every decoding step, rather than relying on one compressed vector, mitigating the bottleneck.

3. **Q: Walk through the mathematical steps of Bahdanau attention at a single decoding step.**
   A: Compute an alignment score for each encoder position `i` using a feedforward network over the decoder's previous hidden state and that encoder state: `e_{t,i} = v^T tanh(W1*s_{t-1} + W2*h_i)`. Normalize scores via softmax across all `i` to get attention weights `α_{t,i}` summing to 1. Compute the context vector as the weighted sum `c_t = Σ_i α_{t,i} * h_i`. Feed this context vector (concatenated with the previous output token) into the decoder RNN to produce the next hidden state and output.

4. **Q: What's the key structural difference between Bahdanau and Luong attention?**
   A: Bahdanau computes alignment scores using the decoder's *previous* hidden state via an additive feedforward network, and injects the resulting context vector as input to the decoder's recurrent step. Luong computes scores using the decoder's *current* (already-updated) hidden state via a simpler multiplicative/dot-product function, and combines the context vector with the decoder output *after* the recurrence step, generally at lower computational cost.

5. **Q: Why does greedy decoding sometimes produce clearly worse output than beam search, even though it picks the "best" token at every step?**
   A: Greedy decoding is locally optimal but not globally optimal — the highest-probability token at step `t` can commit the sequence to a path where all subsequent continuations are poor (e.g., grammatically painting itself into a corner), whereas a slightly lower-probability token at step `t` might lead to a much higher-probability, more coherent full sequence. Beam search hedges against this by tracking multiple candidate sequences simultaneously.

6. **Q: Why is beam search not equivalent to finding the truly optimal output sequence?**
   A: Beam search only keeps the top-k highest-scoring partial hypotheses at each step and discards the rest, so a hypothesis that looks suboptimal early on (and gets pruned) but would have led to the true best full sequence is permanently lost — it's a heuristic, incomplete search, not exhaustive search over the exponential space of all possible sequences.

7. **Q: Why does raw beam search score favor shorter sequences, and how is this typically corrected?**
   A: The score is a sum of log-probabilities (each ≤ 0), so every additional generated token can only decrease (or at best not increase) the cumulative score; shorter sequences accumulate fewer negative terms and thus score higher by default, biasing the model toward premature termination. This is corrected via length normalization — dividing the score by sequence length (often raised to a tunable exponent) before comparing candidates.

8. **Q: When would you prefer sampling-based decoding (top-k / nucleus sampling) over beam search?**
   A: For open-ended generative tasks like dialogue, story, or creative text generation, where diversity and naturalness matter more than a single "correct" answer; beam search tends to converge on generic, repetitive, high-probability-but-bland text in these settings, while sampling methods introduce controlled randomness that better matches the diversity of natural human-generated text.

9. **Q: In an encoder-decoder translation model, what's the difference in role between the encoder's final hidden state and the attention context vector at decoding step t?**
   A: The encoder's final hidden state (used in vanilla, no-attention seq2seq) is one fixed summary of the *entire* source sentence, used identically at every decoding step. The attention context vector `c_t` is *recomputed fresh at every decoding step* as a weighted combination over *all* encoder hidden states, with weights that change dynamically depending on what the decoder currently needs (e.g., focusing on different source words when generating different target words).

10. **Q: What problem do attention weights visually reveal in a trained translation model, and why is this useful for debugging?**
    A: Visualizing the attention weight matrix (source positions vs. target positions) typically reveals roughly monotonic, locally-focused alignments for related languages (e.g., English-French), or clearly crossed alignments for languages with different word order (e.g., subject-object-verb vs subject-verb-object). This gives an interpretable window into what the model is "looking at" when generating each word, useful for diagnosing systematic translation errors (e.g., attending to the wrong source phrase for a given output word).

11. **Q: Why is GRU sometimes preferred over LSTM in production systems despite similar performance?**
    A: GRU merges the forget and input gates into a single update gate and removes the separate cell state, resulting in fewer parameters and a simpler computation graph than LSTM, which can mean faster training/inference and lower memory footprint, with often comparable empirical performance on many tasks — though LSTM can still outperform on some tasks requiring finer-grained memory control.

12. **Q: How does bidirectionality in a BiLSTM encoder help sequence labeling or encoding tasks?**
    A: A unidirectional LSTM's hidden state at position `i` only encodes information from tokens `1...i` (left context); many tasks need right context too (e.g., disambiguating a word's tag based on what follows it). A BiLSTM runs a forward LSTM and a separate backward LSTM and concatenates their hidden states at each position, giving every position access to both full left and right context.

---

## Transformer-based NLP

*(Deep architectural internals — multi-head self-attention mechanics, positional encodings, layer norm placement, scaling laws, RLHF/instruction tuning — belong in the Generative AI / LLM syllabus file. This section covers how BERT/GPT-family models are used and fine-tuned for classic NLP tasks.)*

### BERT: Masked Language Modeling, NSP, Fine-tuning

**BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018)** is a transformer *encoder*-only model pretrained with two self-supervised objectives on large unlabeled corpora:

#### Masked Language Modeling (MLM)

Randomly mask ~15% of input tokens and train the model to predict the original token using **both left and right context simultaneously** (true bidirectionality, unlike ELMo's shallow concatenation of independent directions). Of the masked positions: 80% are replaced with `[MASK]`, 10% with a random token, and 10% left unchanged — this mix prevents the model from *only* learning to handle the literal `[MASK]` token (which never appears at fine-tuning/inference time) and forces it to build genuinely robust contextual representations for every token, since it can never be sure whether the token it's looking at is "trustworthy."

```
Input:  "The [MASK] sat on the mat"
Target: predict "cat" at the masked position, using full bidirectional context
```

#### Next Sentence Prediction (NSP)

Given two segments A and B, predict whether B genuinely follows A in the original text (50% of training pairs are true consecutive pairs, 50% are random pairs from elsewhere in the corpus). Intended to teach sentence-pair relationship understanding useful for tasks like NLI or QA. (Later work, e.g., RoBERTa, found NSP contributes little and often removed it in favor of just longer MLM training on contiguous text, sometimes replacing NSP with alternative objectives like sentence-order prediction.)

#### Fine-tuning BERT for Downstream Tasks

BERT is pretrained once, then **fine-tuned** (all or most weights updated with a small task-specific head) on labeled downstream data:

- **Sentence/sequence classification** (sentiment, topic): add a classification head on top of the `[CLS]` token's final hidden state.
- **Token classification** (NER, POS): add a per-token classification head on top of every token's final hidden state.
- **Span extraction / extractive QA**: add two linear heads predicting start-position and end-position logits over all tokens (see QA section below).
- **Sentence-pair tasks** (NLI, paraphrase detection): feed both sentences separated by `[SEP]`, classify from `[CLS]`.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

texts = ["This movie was fantastic!", "Absolutely terrible experience."]
labels = [1, 0]  # 1 = positive, 0 = negative

encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
outputs = model(**encoded, labels=torch.tensor(labels))
print(outputs.loss, outputs.logits)
# In practice: wrap into a Dataset + Trainer (or a manual training loop) for full fine-tuning.
```

**Fine-tuning practical tips:**
- Use a small learning rate (typically 2e-5 to 5e-5) with warmup — large LRs destroy pretrained knowledge (catastrophic forgetting).
- Few epochs (2–4) are usually sufficient; BERT overfits quickly on small datasets.
- Freezing lower layers and only fine-tuning upper layers + head can help with very small datasets or to save compute, at some accuracy cost.
- Always use the exact tokenizer that matches the pretrained checkpoint — mismatched vocab/tokenization silently corrupts inputs.

### GPT-family: Autoregressive LM Objective vs BERT's Bidirectionality

**GPT (Generative Pretrained Transformer)** models are transformer *decoder*-only architectures trained with a single, simple objective: **causal (autoregressive) language modeling** — predict the next token given only the *preceding* tokens (left-to-right, no access to future tokens):

```
Objective: maximize Σ_t log P(w_t | w_1, ..., w_{t-1})
```

This is enforced architecturally via a **causal attention mask** that prevents each position from attending to any future position, ensuring the model can be used for genuine left-to-right generation at inference time without any information leakage from the future.

**Core BERT vs GPT distinction:**

| | BERT | GPT |
|---|---|---|
| Architecture | Transformer *encoder* | Transformer *decoder* |
| Pretraining objective | Masked LM (+ optionally NSP) | Causal (next-token) LM |
| Context per token | Full bidirectional (left + right) | Left-only (causal mask) |
| Natural use case | Understanding tasks: classification, NER, extractive QA (needs full-sequence context) | Generation tasks: text completion, open-ended QA, summarization, dialogue |
| Adaptation to downstream task | Typically fine-tuned with task-specific head | Fine-tuned, or increasingly used via prompting/in-context learning with no architecture change |
| Can generate text autoregressively? | Not natively (bidirectional attention breaks causal generation) | Yes, natively |

**Why the objective choice matters**: bidirectional context (BERT) gives richer representations for *understanding* a fixed piece of text (since every token can use both past and future information), making it strong for classification/labeling/extraction. Causal, left-to-right modeling (GPT) is exactly the setup needed for coherent autoregressive *generation*, since at generation time the model genuinely never has access to "future" tokens (they don't exist yet) — training and inference conditions match perfectly, whereas trying to generate autoregressively from a bidirectionally-trained model would create a train/inference mismatch.

### Sentence Embeddings (Sentence-BERT) and Semantic Search

Naively averaging BERT's token embeddings (or using the `[CLS]` token from a vanilla, non-fine-tuned BERT) produces surprisingly poor sentence-level semantic representations for similarity tasks — vanilla BERT was never trained with a sentence-similarity objective, so its embedding space isn't well-organized for cosine-similarity comparison, and computing pairwise BERT similarity for every pair in a large collection (cross-encoder style, feeding both sentences together) is also computationally infeasible at scale (O(n²) full transformer passes for n documents).

**Sentence-BERT (SBERT, Reimers & Gurevych, 2019)** fine-tunes BERT with a **siamese/twin network structure**: both sentences are passed *independently* through the same BERT encoder, mean/max-pooled into fixed-size sentence vectors, and trained with objectives like:
- **Triplet loss / contrastive loss** on labeled similar/dissimilar sentence pairs.
- **Classification objective on NLI data** (entailment/contradiction/neutral) using the concatenated `(u, v, |u-v|)` sentence vector representation, which incidentally shapes the embedding space to be well-suited for cosine similarity.
- Modern variants increasingly use large-scale **contrastive learning** (e.g., in-batch negatives, hard negative mining) directly optimizing cosine similarity to be high for semantically similar pairs and low for dissimilar ones.

Because both sentences are encoded *independently* into fixed vectors, once precomputed, comparing millions of sentence pairs is just a fast vector dot-product/cosine-similarity operation — enabling **semantic search at scale**: embed a large document corpus once (offline), store vectors in an index (e.g., FAISS, HNSW), and at query time embed only the query and retrieve nearest neighbors by cosine similarity, in sublinear time with approximate nearest-neighbor search.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = ["The cat sits on the mat.", "Stock markets rallied today.", "A feline rests on a rug."]
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)

query = "Where is the cat?"
query_embedding = model.encode(query, normalize_embeddings=True)

# cosine similarity (dot product of normalized vectors = cosine similarity)
similarities = corpus_embeddings @ query_embedding
best_match_idx = np.argmax(similarities)
print(corpus[best_match_idx], similarities[best_match_idx])
```

**Cross-encoder vs bi-encoder (SBERT):**

| | Cross-encoder | Bi-encoder (SBERT) |
|---|---|---|
| Input | Both texts jointly, e.g. `[CLS] text_A [SEP] text_B [SEP]` | Each text encoded independently |
| Interaction | Full cross-attention between both texts | None until final similarity computation |
| Accuracy | Higher (rich cross-attention) | Slightly lower |
| Scalability | O(n²) — must jointly encode every pair; infeasible for large-scale retrieval | O(n) encoding + fast vector search; scales to millions of documents |
| Typical use | Reranking a small candidate set (e.g., top-100 from a bi-encoder retriever) | First-stage large-scale retrieval / semantic search |

A common production pattern (used in RAG pipelines and search systems) is **retrieve-then-rerank**: use a fast bi-encoder to retrieve a manageable candidate set from a huge corpus, then apply a more accurate but expensive cross-encoder to rerank just those candidates.

### Task-specific Fine-tuning: Classification Head, Span Extraction, Generation

| Task type | Head architecture | Output |
|---|---|---|
| Sequence classification | Linear layer on `[CLS]`/pooled representation | Single label (or multi-label) over the whole input |
| Token classification (NER, POS) | Linear layer applied to every token's final hidden state | One label per token |
| Extractive QA (span extraction) | Two linear layers over all tokens, predicting start-logit and end-logit independently | Start/end token indices defining the answer span within the context |
| Sequence generation (summarization, translation) | Full decoder (or encoder-decoder) generating tokens autoregressively; loss is token-level cross-entropy over the whole target sequence | Generated text sequence |

**Extractive QA head mechanics**: for each token `i` in the context, the model computes a start-score `s_i` and end-score `e_i` (typically via `Linear(hidden_states) → 2 logits per token`, split into start/end); the predicted answer span is `argmax` over valid `(start ≤ end)` combinations of `s_i + e_j`. Training minimizes cross-entropy loss against the ground-truth start and end token indices independently, then sums the two losses.

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

question = "What did the cat sit on?"
context = "The cat sat on the mat in the living room."
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)
answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
print(tokenizer.decode(answer_tokens))  # "the mat"
```

**Practical fine-tuning tips across all head types:**
- Choose `max_length`/truncation strategy carefully for QA — long contexts often need a **sliding window** approach (overlapping chunks) since the answer might be truncated otherwise.
- For token classification, remember subword tokenization splits words into multiple tokens — labels must be aligned to the first subword of each word (commonly, subsequent subwords get a special "ignore" label, e.g., `-100` in PyTorch cross-entropy, so they don't contribute to loss).
- For generation heads, decoding strategy (greedy/beam/sampling) at inference is a separate choice from training (which uses teacher forcing — feeding the ground-truth previous token, not the model's own generated token, during training).

### Multilingual and Cross-Lingual NLP

Most of this file's examples are implicitly English-centric; production NLP systems increasingly need to work across languages, either by training one model to handle many languages at once, or by transferring a model trained in one (usually high-resource) language to another with little or no labeled data.

#### Multilingual Embeddings

Static cross-lingual embeddings align *separately trained* monolingual embedding spaces (e.g., English Word2Vec and French Word2Vec) into a **shared vector space**, so that a word and its translation end up close together, even though the two embedding spaces were never trained jointly.

- **Supervised alignment**: given a small bilingual dictionary (a few thousand word-translation pairs), learn a linear (often orthogonality-constrained) transformation matrix `W` mapping the source embedding space onto the target space, solved via the **Procrustes problem** (`min_W ||WX - Y||_F` subject to `W` orthogonal, solved in closed form via SVD).
- **Unsupervised alignment** (e.g., **MUSE**, Conneau et al., 2018): learns the same mapping *without* any bilingual dictionary, using adversarial training (a discriminator tries to distinguish mapped source embeddings from real target embeddings) followed by a refinement step (iterative Procrustes alignment using the mapping's own high-confidence mutual nearest-neighbor pairs as a pseudo-dictionary).
- **Sentence-level cross-lingual embeddings** (e.g., **LASER**, **LaBSE**): encode whole sentences from many languages into one shared space directly (trained with parallel/translation-pair objectives), enabling cross-lingual semantic search and **bitext mining** (finding translation pairs in unaligned multilingual corpora).

#### Multilingual Transformers

- **mBERT (multilingual BERT)**: a single BERT model pretrained with the standard MLM objective jointly on Wikipedia text from ~104 languages, sharing one large subword vocabulary across all of them. There is no explicit cross-lingual supervision (no parallel/translation data) — cross-lingual alignment emerges purely from shared subwords (e.g., named entities, numbers, cognates) and joint training pressure.
- **XLM / XLM-R**: XLM adds an explicit **Translation Language Modeling (TLM)** objective on top of MLM — trained on concatenated parallel sentence pairs so the model must use cross-lingual context to fill in masks, producing stronger alignment than mBERT's implicit approach. **XLM-RoBERTa (XLM-R)** instead scales up: RoBERTa-style MLM-only pretraining, but on a much larger, cleaner CommonCrawl-derived corpus across 100 languages, and empirically outperforms both mBERT and XLM on most cross-lingual benchmarks — showing that *scale and data quality* can substitute for an explicit cross-lingual training signal.

#### Zero-Shot Cross-Lingual Transfer

Because these models share one representation space across languages, a common and highly practical pattern is: **fine-tune on labeled task data in one language only** (typically English, where labels are abundant), then **directly apply the fine-tuned model to a different language at inference time**, with zero labeled examples in that target language.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Fine-tuned only on English CoNLL-2003 NER data
model_name = "xlm-roberta-base"  # example base; assume a fine-tuned NER checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner = pipeline("ner", model=model, tokenizer=tokenizer)

# Zero-shot: run on German text despite fine-tuning only on English-labeled data
print(ner("Angela Merkel besuchte Berlin im Jahr 2015."))
```

This works because the shared subword vocabulary and jointly-trained representation space place semantically/syntactically analogous tokens from different languages near each other, so a classification head trained on English hidden states generalizes reasonably to hidden states produced from other languages by the same encoder. Performance degrades with **linguistic distance** from the fine-tuning language — transfer from English to German or Spanish (similar script, similar word order) is much stronger than transfer to, say, Japanese or Finnish.

**Pitfalls and known limitations:**
- **Curse of multilinguality**: a fixed model capacity shared across 100+ languages trades off *per-language* quality against *breadth* — a multilingual model typically underperforms a comparably-sized monolingual model on any single high-resource language, and low-resource languages in the mix get comparatively little effective capacity.
- **Tokenizer fairness**: a shared subword vocabulary trained on a corpus dominated by high-resource languages fragments low-resource-language words into far more subword pieces than it does for high-resource languages, increasing sequence length, compute cost, and (in LLM-API contexts) token-billed cost disproportionately for underrepresented languages — the same tokens-per-word issue flagged earlier for BPE, now with an equity dimension across languages.
- **Zero-shot transfer is not free**: it works best for typologically similar languages and structurally simple tasks; tasks requiring language-specific syntactic or cultural knowledge (e.g., idiom-heavy sentiment, language-specific NER entity conventions) often still need at least some target-language labeled data or few-shot adaptation.

### Interview Questions

1. **Q: Explain BERT's Masked Language Modeling objective and why the 80/10/10 masking strategy (mask/random/unchanged) is used instead of always using `[MASK]`.**
   A: MLM randomly masks ~15% of tokens and trains the model to predict them using full bidirectional context. If every masked position were always replaced with the literal `[MASK]` token, the model would learn representations conditioned on seeing `[MASK]`, but at fine-tuning/inference time `[MASK]` never appears in real input, creating a train-test mismatch. By sometimes keeping the original token or substituting a random token, the model is forced to build genuinely robust contextual representations for *every* token, since it can't assume any given token is "safe" or unmasked.

2. **Q: Why is BERT called "bidirectional" while ELMo is only "shallowly bidirectional"?**
   A: BERT's self-attention lets every token attend to both left and right context *jointly, within the same computation* at every layer — the representation at each position is built from genuinely combined bidirectional information. ELMo trains a forward LSTM and a backward LSTM largely independently and only concatenates/combines their separately-computed hidden states afterward, so neither direction's computation is ever informed by the other during its own forward pass.

3. **Q: Why can't BERT be used directly for autoregressive text generation the way GPT can?**
   A: BERT is trained with full bidirectional self-attention (every token sees the whole sequence, including "future" tokens relative to any generation order), which is fundamentally incompatible with left-to-right generation, where future tokens don't exist yet at inference time. GPT is trained with a causal attention mask that only allows attending to past tokens, exactly matching the conditions present during actual autoregressive generation.

4. **Q: What's the practical benefit of Next Sentence Prediction in original BERT, and why did later work (e.g., RoBERTa) drop it?**
   A: NSP was intended to teach the model to understand relationships between sentence pairs, useful for tasks like NLI. RoBERTa's ablations found NSP contributed little to downstream performance and that simply training MLM longer on contiguous spans of text (without the NSP objective) performed as well or better, suggesting the original NSP task was either too easy or not well aligned with useful representation learning.

5. **Q: Describe how you would fine-tune BERT for a token classification task like NER, including a key subword-alignment pitfall.**
   A: Add a linear classification head applied to every token's final hidden state, and train with per-token cross-entropy against BIO labels. The pitfall: subword tokenization splits some words into multiple tokens, so ground-truth word-level labels must be mapped only to the *first* subword of each word (subsequent subword pieces are typically assigned an ignore label, e.g. -100, excluded from the loss), otherwise labels get misaligned or duplicated incorrectly across subword pieces.

6. **Q: Why is naive averaging of BERT token embeddings a poor sentence representation for semantic similarity tasks?**
   A: Vanilla, non-fine-tuned BERT was never trained with an explicit sentence-level similarity objective, so its embedding space isn't organized such that cosine similarity between averaged token vectors reliably reflects semantic similarity — empirically, averaged BERT embeddings often underperform even simpler baselines like averaged GloVe vectors on similarity benchmarks.

7. **Q: What is the siamese network structure in Sentence-BERT, and why does it enable scalable semantic search?**
   A: SBERT passes two sentences independently (not jointly) through the same shared-weight BERT encoder, pooling each into a fixed-size vector, then trains with objectives (triplet/contrastive loss, or NLI classification on the combined representation) that shape the embedding space for meaningful cosine similarity. Because each sentence is encoded independently, a large corpus can be embedded once, offline, and stored in a vector index; queries only need one new encoding plus a fast vector similarity search, unlike a cross-encoder which must jointly process every candidate pair at query time.

8. **Q: Compare a cross-encoder and a bi-encoder for a semantic search system, and describe a common production pattern combining both.**
   A: A cross-encoder jointly encodes the query and each candidate document with full cross-attention, giving higher accuracy but costing one full transformer forward pass per candidate — infeasible for large corpora. A bi-encoder (like SBERT) encodes documents independently and offline, enabling fast approximate nearest-neighbor retrieval at query time, at somewhat lower accuracy. Production systems typically use retrieve-then-rerank: a bi-encoder retrieves a candidate shortlist (e.g., top 100) from millions of documents, then a cross-encoder reranks just that shortlist for higher final precision.

9. **Q: How does an extractive QA model architecture (like for SQuAD) differ from a text classification head on BERT?**
   A: A classification head applies a single linear layer to the pooled `[CLS]` representation to produce one label for the whole input. An extractive QA head instead applies a linear layer to *every* token's final hidden state, producing a start-logit and end-logit per token; the predicted answer is the highest-scoring valid `(start, end)` span, so the output is a substring of the input context rather than a fixed label from a predefined set.

10. **Q: Why does fine-tuning BERT typically use a much smaller learning rate than training a model from scratch?**
    A: BERT arrives pretrained with rich, generally useful language representations learned from massive corpora; a large learning rate during fine-tuning would apply large gradient updates that can catastrophically overwrite/forget this pretrained knowledge in just a few steps. Small learning rates (2e-5–5e-5) with warmup allow gentle adaptation to the downstream task while largely preserving the useful pretrained representations.

11. **Q: What does "teacher forcing" mean during sequence generation model training, and how does it differ from inference-time decoding?**
    A: During training, at each decoding step, the model is given the *ground-truth* previous target token as input (regardless of what the model itself would have predicted), which stabilizes and speeds up training by preventing early errors from compounding across the sequence. At inference time, there's no ground truth available, so the model must feed back its *own* previously generated tokens — this train/inference discrepancy (sometimes called "exposure bias") can degrade generation quality, especially for longer sequences.

12. **Q: In what scenario would you choose a decoder-only (GPT-style) model over an encoder-only (BERT-style) model for a downstream task, and vice versa?**
    A: Choose decoder-only/GPT-style for open-ended generation tasks (summarization, dialogue, free-form text completion) where the model must produce novel, potentially long output sequences, especially when leveraging prompting/in-context learning without task-specific fine-tuning. Choose encoder-only/BERT-style for understanding tasks with a fixed, typically shorter output space — classification, NER, extractive QA — where bidirectional context over the full input improves representation quality and a simple task-specific head suffices.

13. **Q: Why might a sliding-window strategy be necessary for extractive QA over long documents?**
    A: Transformer models have a fixed maximum input length (e.g., 512 tokens for base BERT); if the context document exceeds this, naive truncation might cut off the exact portion of the context containing the true answer. A sliding window splits the long context into overlapping chunks, running QA inference on each chunk and aggregating/selecting the best-scoring answer span across all chunks, ensuring no candidate answer region is missed.

14. **Q: What's the key difference in how mBERT and XLM-R achieve cross-lingual alignment?**
    A: mBERT relies on implicit alignment — a single BERT model trained with only the standard MLM objective jointly on ~104 languages' Wikipedia text, sharing a subword vocabulary, with cross-lingual signal emerging solely from shared subwords and joint training pressure (no parallel data). XLM-R instead scales up: still MLM-only, but pretrained on a much larger, cleaner CommonCrawl corpus across 100 languages, empirically outperforming both mBERT and XLM (which added an explicit translation-language-modeling objective on parallel data) — demonstrating that scale/data quality can substitute for explicit cross-lingual supervision.

15. **Q: Explain zero-shot cross-lingual transfer and why it works at all.**
    A: A multilingual encoder is fine-tuned on labeled task data in only one language (typically English) and then applied directly to a different language at inference time with zero labeled examples in that language. This works because the shared representation space (built from a shared subword vocabulary and jointly-trained weights) places semantically/syntactically analogous tokens from different languages near each other, so a classification head trained on English hidden states transfers reasonably to hidden states the same encoder produces for other languages — performance depends heavily on linguistic distance from the fine-tuning language.

16. **Q: What is the "curse of multilinguality"?**
    A: Because a multilingual model has fixed parameter capacity shared across all languages it's trained on, packing in more languages trades off against per-language quality — a multilingual model typically underperforms a comparably-sized monolingual model on any single high-resource language, and low-resource languages in the mix receive comparatively little effective model capacity.

17. **Q: How does MUSE learn cross-lingual word embeddings without a bilingual dictionary?**
    A: MUSE uses adversarial training — a generator learns a linear mapping from the source language's embedding space into the target's, while a discriminator tries to distinguish mapped source embeddings from real target embeddings; once the mapping is roughly aligned, a refinement step (iterative Procrustes alignment using the mapping's own high-confidence mutual nearest-neighbor pairs as a pseudo-dictionary) sharpens it further, entirely without human-provided bilingual supervision.

---

## Applied NLP Tasks

### Named Entity Recognition and Relation Extraction

**NER** (covered in depth above) identifies entity spans and types. **Relation Extraction (RE)** goes a step further: given two identified entities in a sentence, classify the semantic relationship between them (e.g., `"Elon Musk founded SpaceX"` → `(Elon Musk, founded, SpaceX)`).

**Approaches to RE:**
1. **Pattern/rule-based**: hand-crafted lexico-syntactic patterns (e.g., `"<PERSON> founded <ORG>"`); high precision, low recall, brittle.
2. **Feature-based classifiers**: extract features (dependency path between entities, POS tags, entity types) and train a classifier (SVM, logistic regression) over candidate entity pairs.
3. **Neural approaches**: encode the sentence with a transformer, mark the two entity spans (e.g., special entity markers inserted around each mention), and classify the relation from the resulting contextual representation — often fine-tuning BERT-style models with entity-marker tokens for this purpose.
4. **Joint entity + relation extraction**: single end-to-end model that jointly predicts entity spans and relations simultaneously, avoiding compounding errors from a two-stage pipeline (NER errors propagate into RE if done sequentially).

```python
# Conceptual entity-marker approach for RE fine-tuning
sentence = "[E1]Elon Musk[/E1] founded [E2]SpaceX[/E2] in 2002."
# Feed marked sentence through BERT, pool representations at E1/E2 marker positions,
# classify relation type (e.g., "founder_of") from concatenated marker representations.
```

**Pitfalls**: RE datasets are often extremely imbalanced (most entity pairs in a sentence have "no relation"); distant supervision (using knowledge base facts to auto-label sentences mentioning both entities) introduces label noise since co-occurrence doesn't guarantee the sentence actually expresses that relation.

### Coreference Resolution

**Coreference resolution** identifies all mentions in a text that refer to the same real-world entity, and clusters them together — e.g., in `"Marie Curie won the Nobel Prize. She was the first woman to do so."`, `"Marie Curie"` and `"She"` must be linked into a single coreference chain.

**Terminology:**
- **Anaphora**: a mention (usually a pronoun) referring *back* to an earlier mention (the common case, "Marie Curie ... She").
- **Cataphora**: a mention referring *forward* to a later one (rarer, e.g., "Before **she** spoke, **Marie Curie** paused.").
- **Mention**: any noun phrase or pronoun that could refer to an entity (proper names, definite noun phrases, pronouns).

**Approaches:**
1. **Rule-based**: heuristics using gender/number agreement, syntactic salience, and recency (e.g., Hobbs' algorithm walks up the syntactic tree searching for an antecedent matching the pronoun's gender/number; centering theory ranks candidate antecedents by discourse salience). Interpretable but brittle on genuinely ambiguous cases.
2. **Mention-pair / mention-ranking classifiers**: extract features (distance between mentions, gender/number agreement, syntactic role) for candidate `(mention, antecedent)` pairs and classify/rank whether they co-refer; requires a separate mention-detection stage first.
3. **End-to-end neural coreference** (Lee et al., 2017, and successors): jointly considers *all* spans in the document as potential mentions and *all* pairs of spans as potential coreference links in one differentiable model (using contextual embeddings, typically from a BERT-style encoder), scoring each span's "mention-ness" and each span-pair's "antecedent-ness" together — removing the hard dependency on a separate, error-compounding mention-detection stage.

**The Winograd Schema Challenge** is the canonical hard-case benchmark, designed so that resolving the pronoun requires **commonsense world knowledge**, not just syntax/gender/number agreement:
```
"The trophy doesn't fit in the suitcase because it's too big."   → "it" = the trophy
"The trophy doesn't fit in the suitcase because it's too small." → "it" = the suitcase
```
Both sentences are syntactically identical; only world knowledge about size relationships and containment disambiguates the pronoun, which is why this remains a stress test for whether a model has "real" understanding versus surface pattern matching.

**Evaluation metrics**: coreference is scored by comparing predicted clusters to gold clusters using **MUC**, **B³**, and **CEAF**, each capturing a different notion of cluster-overlap correctness; the standard reported number is the **CoNLL F1**, the average of MUC, B³, and CEAF F1-scores (as used in the CoNLL-2012 shared task).

```python
# Conceptual usage of a neural coreference resolver (e.g., via spaCy + a coref extension, or AllenNLP)
# doc = coref_model("Marie Curie won the Nobel Prize. She was the first woman to do so.")
# doc.coref_clusters -> [["Marie Curie", "She"]]
```

**Pitfalls:**
- Simple gender/number-agreement heuristics fail on any case requiring semantic/world knowledge (Winograd-style), and on singular "they," ambiguous group references, or nested possessives.
- Coreference errors compound into downstream tasks that depend on it — e.g., relation extraction or summarization can attribute an action to the wrong entity if an upstream coreference link is wrong.
- Cross-document coreference (linking mentions of the same entity across *multiple* documents, needed for building knowledge graphs from a corpus) is substantially harder than within-document coreference and is a distinct, less mature subtask.

### Text Summarization: Extractive vs Abstractive

**Extractive summarization** selects and concatenates existing sentences (or spans) directly from the source document, without generating new text. Approaches range from classical (TextRank — a graph-based algorithm treating sentences as nodes, edges weighted by inter-sentence similarity, ranked via PageRank-style centrality) to neural (binary classifier scoring each sentence's "should be included" probability using contextual embeddings).

**Abstractive summarization** generates new sentences that may paraphrase, compress, or synthesize information not verbatim present in the source — requires a full sequence-to-sequence generative model (encoder-decoder transformer, e.g., BART, T5, PEGASUS), trained to produce fluent, novel summary text conditioned on the source document.

| | Extractive | Abstractive |
|---|---|---|
| Output | Verbatim source sentences/spans | Newly generated text |
| Faithfulness/factuality risk | Low (text is copied verbatim, can't misstate facts not already there) | Higher (can hallucinate facts not supported by source) |
| Fluency/coherence | Can be disjointed (sentences from different parts stitched together) | Generally more fluent, coherent, human-like |
| Complexity | Simpler (ranking/selection problem) | Harder (full generation problem) |
| Typical models | TextRank, LexRank, BERT-based sentence scorer | BART, T5, PEGASUS, GPT-style models |

**Hybrid approaches**: extract-then-abstract pipelines first select salient sentences/content, then rewrite/compress them abstractively, balancing factual grounding with fluency.

**Evaluation** typically uses ROUGE (see below), though ROUGE has known limitations for abstractive summaries (it rewards lexical overlap, not necessarily factual correctness or true semantic quality) — modern evaluation increasingly supplements ROUGE with factual-consistency checkers and human evaluation.

### Machine Translation Evaluation: BLEU, ROUGE, METEOR

#### BLEU (Bilingual Evaluation Understudy)

BLEU measures **n-gram precision** of a generated (candidate) translation against one or more human reference translations, with a **brevity penalty** to discourage gaming the metric by generating very short, high-precision-but-incomplete output.

```
BLEU = BP * exp( Σ_{n=1}^{N} w_n * log(p_n) )

p_n = (count of candidate n-grams that appear in reference, clipped by max reference count)
      / (total candidate n-grams of length n)

BP (brevity penalty) = 1                                if c > r
                      = exp(1 - r/c)                     if c <= r
      where c = candidate length, r = reference length
```

`p_n` is computed for n = 1 to 4 (typically), each n-gram's count in the candidate is *clipped* to not exceed its count in the reference (preventing a model from repeating one correct word many times to inflate precision artificially). The geometric mean (via the log-sum-exp) of these n-gram precisions, scaled by the brevity penalty, gives the final BLEU score, usually reported as 0–100.

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["the", "cat", "is", "on", "the", "mat"]]
candidate = ["the", "cat", "sat", "on", "the", "mat"]

score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
print(score)
```

**BLEU limitations**: purely lexical n-gram overlap — penalizes valid paraphrases/synonyms that don't lexically match the reference, doesn't account for grammaticality or meaning beyond n-gram overlap, correlates poorly with human judgment at the sentence level (better at corpus level), and is sensitive to the choice/number of reference translations available.

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE is BLEU's recall-oriented cousin, primarily used for **summarization** evaluation. The most common variants:

- **ROUGE-N**: n-gram *recall* between candidate and reference (`overlapping n-grams / total n-grams in reference`) — measures how much of the reference content is captured, complementing BLEU's precision focus.
- **ROUGE-L**: based on the **Longest Common Subsequence (LCS)** between candidate and reference, capturing in-order (not necessarily contiguous) overlap, which better rewards sentence-level structural similarity than fixed n-gram matching.
- **ROUGE-S / ROUGE-SU**: skip-bigram co-occurrence statistics, allowing gaps between matched word pairs.

```
ROUGE-N (recall) = Σ (count of matching n-grams) / Σ (count of n-grams in reference summaries)
```

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(
    "The cat sat on the mat",
    "The cat is sitting on the mat"
)
print(scores)
# {'rouge1': Score(precision=..., recall=..., fmeasure=...), 'rouge2': ..., 'rougeL': ...}
```

**BLEU vs ROUGE**: BLEU is precision-oriented (penalizes adding unsupported content — appropriate for translation, where you want to avoid generating extra/wrong content) while ROUGE is recall-oriented (rewards capturing reference content — appropriate for summarization, where missing key information is the primary failure mode). Both are commonly reported with an F-measure balancing precision and recall regardless of original orientation.

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR improves on BLEU's purely surface-level n-gram matching by incorporating **stemming, synonym matching (via WordNet), and paraphrase tables**, matching words even when not lexically identical, and computes a harmonic mean of unigram precision and recall (weighted more toward recall than BLEU), with an explicit **fragmentation penalty** that penalizes matches that are scattered/out-of-order relative to the reference (rewarding contiguous, well-ordered matches).

```
METEOR = F_mean * (1 - Penalty)

F_mean = (10 * P * R) / (R + 9*P)     [weighted harmonic mean favoring recall]
Penalty = γ * (chunks / matches)^θ     [fragmentation penalty; more/smaller chunks = more scattered = higher penalty]
```

METEOR correlates better with human judgment at the *sentence* level than BLEU (BLEU is more reliable at corpus level), at the cost of requiring language-specific resources (WordNet-like synonym databases, stemmers), limiting its use for low-resource languages.

| Metric | Orientation | Handles synonyms/paraphrase? | Best suited for | Correlation with human judgment |
|---|---|---|---|---|
| BLEU | Precision (+ brevity penalty) | No (exact n-gram match only) | Machine translation (corpus-level) | Moderate at corpus level, weak at sentence level |
| ROUGE | Recall-oriented (F-measure variants exist) | No (ROUGE-L allows gaps but still lexical) | Summarization | Moderate |
| METEOR | Balanced (recall-weighted harmonic mean) | Yes (stemming, WordNet synonyms, paraphrases) | MT, especially sentence-level evaluation | Better than BLEU at sentence level |

**Shared limitation across all three**: they are fundamentally *lexical/surface-form* metrics (even METEOR's synonym matching is limited to what's in its lexical resources) and do not directly measure semantic equivalence, factual correctness, or fluency the way a human or a learned neural metric (e.g., BERTScore, COMET) would — a perfectly valid paraphrase with low lexical overlap can score poorly despite being an excellent translation/summary. Modern MT/summarization evaluation increasingly supplements these with embedding-based metrics (BERTScore) and human evaluation.

### Question Answering: Extractive vs Generative

**Extractive QA** (SQuAD-style): the answer is guaranteed to be a contiguous span within a given context passage; the model predicts start/end token positions (see BERT span-extraction head above). Cannot answer questions whose answer requires synthesis, reasoning across multiple non-adjacent spans, or isn't literally present in the text. Simpler to evaluate (exact match / F1 against gold span) and inherently more factually grounded (answer is verbatim from a trusted source).

**Generative QA**: the model generates a free-form answer using a decoder (seq2seq or decoder-only LLM), not constrained to be a literal substring of any given context. Can answer questions requiring synthesis/reasoning/summarizing across multiple facts, or with no explicit context at all (parametric/"closed-book" QA drawing on knowledge learned during pretraining) — but riskier for factual accuracy/hallucination since nothing forces grounding in a verifiable source. Modern **Retrieval-Augmented Generation (RAG)** systems combine both worlds: retrieve relevant passages (semantic search, as covered above), then generate an answer conditioned on retrieved context, aiming for the fluency/synthesis ability of generative QA with better grounding than pure closed-book generation.

| | Extractive QA | Generative QA |
|---|---|---|
| Answer form | Exact span from given context | Freely generated text |
| Requires context passage? | Yes, always | Optional (can be closed-book) |
| Can synthesize/summarize across facts? | No | Yes |
| Hallucination risk | Very low (answer literally copied from source) | Higher (can generate unsupported claims) |
| Evaluation | Exact match, span-F1 | ROUGE/BLEU (weak), human eval, factuality checkers |

### Sentiment/Emotion Analysis and Intent Classification for Chatbots

**Sentiment analysis** classifies text polarity — commonly binary (positive/negative), ternary (+neutral), or fine-grained (1–5 star scale, or continuous valence score). **Emotion analysis** goes further, classifying into discrete emotion categories (joy, anger, sadness, fear, surprise, disgust — often based on psychological models like Ekman's basic emotions) or continuous valence-arousal dimensions.

**Aspect-based sentiment analysis (ABSA)** identifies sentiment *per aspect/entity* mentioned in text rather than one overall label — e.g., for `"The food was great but service was slow"`: `food → positive`, `service → negative`. This is more actionable for businesses than a single overall sentiment score and typically requires jointly identifying aspect terms (a span-extraction-like subtask) and classifying sentiment conditioned on each aspect.

**Intent classification** (chatbot/virtual assistant NLU) maps a user utterance to one of a predefined set of intents (`book_flight`, `check_balance`, `cancel_order`, ...), typically alongside **slot filling** (a sequence labeling task, structurally identical to NER, extracting task-relevant entities like dates, locations, amounts from the utterance). Together, intent classification + slot filling form the core of a task-oriented dialogue system's Natural Language Understanding (NLU) module.

```
Utterance: "Book a flight to Paris on Friday"
Intent: book_flight
Slots:  O    O  O    O  B-LOC  O  B-DATE
              (flight)   (Paris)  (Friday)
```

```python
# Conceptual joint intent + slot model (common architecture: shared BERT encoder, two heads)
# Head 1 (sequence classification from [CLS]): intent label
# Head 2 (token classification, one label per token): BIO slot tags
```

**Practical considerations for chatbot NLU:**
- **Out-of-scope/unknown intent detection**: production systems must detect utterances that don't match any known intent (rather than forcing a wrong classification) — typically via a confidence threshold, a dedicated "none/fallback" class, or explicit out-of-distribution detection techniques.
- **Class imbalance across intents**: some intents (e.g., "greeting") vastly outnumber rare-but-important ones (e.g., "cancel_subscription") in real traffic logs; requires careful sampling/weighting during training and monitoring in production.
- **Ambiguous/multi-intent utterances**: "Book a flight and also check my rewards balance" may require multi-label intent classification or a more sophisticated dialogue management layer, not just single-label classification.
- **Data drift**: user phrasing evolves (new slang, product names); production NLU pipelines need ongoing monitoring and periodic retraining/expansion of the intent taxonomy.

**Pitfalls across applied tasks generally:**
- Deploying a sentiment classifier trained on product reviews directly on social media text (or vice versa) without domain adaptation — sentiment expression, sarcasm patterns, and vocabulary shift significantly across domains.
- Treating BLEU/ROUGE scores as ground truth for generation quality without any human evaluation — these are proxies with known blind spots (fluent but unfaithful, or awkward but faithful, outputs can score similarly).
- Ignoring sarcasm, negation scope, and mixed sentiment within a single sentence, all classic failure modes of bag-of-words-flavored classical sentiment pipelines.

### Interview Questions

1. **Q: Why is relation extraction typically harder and more error-prone than NER, and how does pipeline error propagation manifest concretely?**
   A: RE depends on first correctly identifying entity spans and types (NER) before classifying the relation between them; if NER mislabels or misses an entity, the RE stage either can't evaluate that pair at all or evaluates it with wrong entity-type information, compounding errors through the pipeline. This motivates joint entity-and-relation extraction models that predict both simultaneously, avoiding hard dependency on a perfect upstream NER stage.

2. **Q: Explain how TextRank performs extractive summarization.**
   A: TextRank builds a graph where each sentence is a node, and edges are weighted by a similarity measure between sentence pairs (e.g., overlap-based or embedding cosine similarity). It then runs a PageRank-style iterative algorithm to compute a centrality score for each sentence, treating highly-connected-to-other-important-sentences nodes as more salient, and selects the top-scoring sentences (subject to length constraints) as the summary.

3. **Q: Why does abstractive summarization carry higher hallucination/factuality risk than extractive summarization?**
   A: Extractive summaries only ever contain text copied verbatim from the source, so they cannot introduce claims that weren't already stated (though they can lose context or juxtapose sentences misleadingly). Abstractive summarization uses a generative decoder that can produce fluent, plausible-sounding sentences not actually supported by the source content, since nothing in the generation process constrains output strictly to source facts.

4. **Q: Walk through the BLEU score formula and explain the purpose of the brevity penalty and n-gram clipping.**
   A: BLEU computes `BP * exp(Σ w_n log p_n)` where `p_n` is the clipped n-gram precision for order n (n=1..4 typically) and BP is the brevity penalty. Clipping caps how many times a candidate's repeated n-gram can count toward precision (bounded by the reference's count of that n-gram), preventing gaming the score by repeating one high-precision n-gram many times. The brevity penalty (`exp(1 - r/c)` when candidate length c ≤ reference length r) penalizes overly short candidates, since precision alone would reward a very short but 100%-correct fragment unfairly.

5. **Q: Why is ROUGE typically preferred over BLEU for summarization evaluation, and vice versa for translation?**
   A: Summarization's main failure mode is omitting important reference content, so recall-oriented ROUGE (measuring how much of the reference is covered) is a better fit. Translation's main failure mode is producing extra/incorrect content not licensed by the source, so precision-oriented BLEU (measuring how much of the candidate is actually correct/supported) is the more natural fit. Both metrics have F-measure variants, but their conventional orientations reflect these differing typical failure modes.

6. **Q: What specific improvement does METEOR make over BLEU, and what's its cost?**
   A: METEOR incorporates stemming and WordNet-based synonym/paraphrase matching, so it credits semantically equivalent but lexically different words (unlike BLEU's exact n-gram matching), and it applies a fragmentation penalty rewarding contiguous, well-ordered matches. The cost is a dependency on language-specific lexical resources (WordNet, stemmers, paraphrase tables), which limits straightforward application to low-resource languages lacking such resources.

7. **Q: What's a fundamental blind spot shared by BLEU, ROUGE, and METEOR as evaluation metrics?**
   A: All three are surface-form/lexical-overlap metrics (even METEOR's synonym matching only covers what's captured by its lexical resources) — none directly measures semantic equivalence, logical/factual correctness, or fluency the way a human evaluator or a learned semantic metric (e.g., BERTScore) would. A valid paraphrase with low lexical overlap to the reference can score poorly despite being an excellent translation/summary.

8. **Q: When would you choose extractive QA over generative QA for a production system, and why?**
   A: When factual grounding and auditability are critical (e.g., legal, medical, financial domains) and the answer is genuinely present verbatim in a trusted source document, extractive QA is preferable since the answer is guaranteed to be a literal span from that source, eliminating hallucination risk inherent to free-form generation, even though it can't handle questions requiring synthesis across multiple facts or absent literal answers.

9. **Q: How does Retrieval-Augmented Generation (RAG) try to combine the strengths of extractive and generative QA?**
   A: RAG first retrieves relevant passages from a trusted corpus (via semantic search, e.g., a bi-encoder + vector index) and then conditions a generative model on those retrieved passages to produce a synthesized, fluent answer. This aims to get generative QA's ability to synthesize/summarize across multiple retrieved facts while substantially grounding the output in retrieved, verifiable source content, reducing (though not eliminating) hallucination risk relative to pure closed-book generation.

10. **Q: Explain aspect-based sentiment analysis and why it's more useful than overall document sentiment for many business applications.**
    A: ABSA identifies sentiment separately for each aspect/entity mentioned in a text (e.g., "food" positive, "service" negative within one review), rather than collapsing the entire text into a single overall sentiment label. This is far more actionable for businesses since a single review can contain genuinely mixed sentiment across different aspects, and an overall label would obscure exactly which aspect needs attention (e.g., knowing service specifically was bad, not just "this review was mixed").

11. **Q: Describe the standard NLU architecture for a task-oriented chatbot combining intent classification and slot filling.**
    A: A shared encoder (e.g., BERT) processes the user utterance; one head (typically applied to the pooled `[CLS]` representation) performs sequence classification to predict the overall intent (e.g., `book_flight`); a second head (applied per-token) performs BIO-scheme sequence labeling to extract task-relevant slot entities (e.g., destination, date) from the utterance, jointly trained with a combined loss.

12. **Q: Why is out-of-scope/unknown-intent detection critical for production chatbot NLU, and how is it typically handled?**
    A: Standard intent classifiers assume every input belongs to one of the predefined intent classes, but real users frequently ask things outside the system's designed scope; forcing a confident (but wrong) classification onto an out-of-scope utterance leads to broken, frustrating conversations. It's typically handled via a confidence threshold on the softmax output (routing low-confidence predictions to a fallback/human-handoff path), a dedicated "none of the above" training class with representative negative examples, or explicit out-of-distribution detection methods.

13. **Q: Why can a sentiment classifier trained on Amazon product reviews perform poorly on Twitter/social media text, even for similar polarity concepts?**
    A: Domains differ significantly in vocabulary (slang, abbreviations, hashtags, emoji), text length and structure, sarcasm/irony prevalence, and even what "positive/negative" typically refers to contextually; a classifier's learned decision boundary is calibrated to the statistical patterns of its training domain and doesn't automatically transfer, a phenomenon called domain shift — addressed via domain adaptation, fine-tuning on in-domain labeled data, or at minimum evaluating on in-domain validation data before deployment.

14. **Q: What's a key risk of relying solely on ROUGE scores to select the "best" summarization model during development, without any human evaluation?**
    A: ROUGE rewards lexical overlap with reference summaries but doesn't measure factual correctness, coherence, or genuine informativeness; a model can achieve a high ROUGE score by copying phrasing similar to references while still introducing factual errors (for abstractive models) or by producing disjointed, low-readability extracted sentences (for extractive models) — human evaluation (or factuality-specific automatic metrics) is necessary to catch these failure modes that ROUGE is blind to.

15. **Q: What is the difference between anaphora and cataphora in coreference resolution?**
    A: Anaphora is a mention (usually a pronoun) referring back to an earlier-mentioned entity ("Marie Curie... She"), which is the overwhelmingly common case in natural text. Cataphora is the reverse — a mention referring forward to an entity introduced later in the text ("Before she spoke, Marie Curie paused") — rarer, and harder for left-to-right processing models to resolve since the antecedent hasn't appeared yet.

16. **Q: Why does the Winograd Schema Challenge specifically target commonsense reasoning rather than syntax?**
    A: Its sentence pairs are syntactically identical and only differ in one word (e.g., "too big" vs "too small"), yet that single word flips which entity the pronoun "it" refers to. Since gender/number agreement and syntactic heuristics give zero signal here (both candidate antecedents match), correctly resolving the pronoun requires genuine world knowledge (about relative sizes and containment), making it a stress test for whether a model has real understanding versus pattern matching on syntactic cues.

17. **Q: How does end-to-end neural coreference resolution differ from a classic mention-pair pipeline?**
    A: A mention-pair pipeline first detects candidate mentions in a separate stage, then classifies/ranks candidate antecedent pairs — errors in mention detection propagate and cap the ceiling of the whole pipeline. End-to-end neural coreference (e.g., Lee et al., 2017) jointly scores every span in the document as a potential mention and every span pair as a potential coreference link in one differentiable model built on contextual embeddings, removing the hard dependency on a separate, error-compounding mention-detection stage.

18. **Q: What does the CoNLL F1 metric for coreference resolution actually average?**
    A: It's the average of three different cluster-scoring metrics — MUC, B³, and CEAF — each of which captures a different notion of how well predicted coreference clusters overlap with gold clusters (e.g., link-based vs. mention-based overlap). Reporting the average of all three, as established by the CoNLL-2012 shared task, gives a more robust picture than relying on any single metric, since each has known biases/blind spots on its own.

---

## Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What does BPE stand for and what does it merge? | Byte Pair Encoding; iteratively merges the most frequent adjacent symbol pair to build a subword vocabulary. |
| 2 | What score does WordPiece use to choose merges instead of raw frequency? | `freq(ab) / (freq(a) * freq(b))` — a likelihood-ratio/PMI-like score. |
| 3 | What marker does SentencePiece use for whitespace? | `▁` (underscore-like symbol representing a space). |
| 4 | Stemming or lemmatization: which can produce a non-real word? | Stemming (e.g., Porter stemmer → "studi"). |
| 5 | Name one task where you should NOT remove stopwords. | Sentiment analysis / negation handling (or any pretrained-transformer pipeline). |
| 6 | What does the `[CLS]` token's final hidden state represent in BERT? | A pooled representation of the whole input sequence, used for classification. |
| 7 | TF-IDF: what does IDF penalize? | Terms that appear in many/most documents (low discriminative power). |
| 8 | CBOW predicts what from what? | The center word from the average of its surrounding context words. |
| 9 | Skip-gram predicts what from what? | Context words from the center word. |
| 10 | Why use negative sampling in Word2Vec? | Avoids the O(V) full softmax cost by turning training into cheap binary classification against k sampled negatives. |
| 11 | What does GloVe factorize? | A global word-word co-occurrence matrix. |
| 12 | How does FastText represent a word? | As the sum of embeddings of its character n-grams (plus the whole word). |
| 13 | What's the key limitation static embeddings share (Word2Vec/GloVe/FastText)? | One fixed vector per word regardless of context (can't handle polysemy). |
| 14 | What architecture does ELMo use? | A deep bidirectional multi-layer LSTM language model. |
| 15 | Formula for perplexity given average log-likelihood per token H? | `PP = exp(H)` where H is average negative log-likelihood. |
| 16 | Why does MLE fail for unseen n-grams? | It assigns zero probability, zeroing out the whole sequence probability. |
| 17 | What does Kneser-Ney's continuation probability measure? | How many distinct contexts a word has appeared in (not raw frequency). |
| 18 | What does the "B" mean in BIO tagging? | Beginning of an entity span. |
| 19 | HMM: generative or discriminative? | Generative (models joint P(words, tags)). |
| 20 | CRF: generative or discriminative? | Discriminative (models conditional P(tags\|words) directly). |
| 21 | What algorithm decodes the optimal tag sequence for HMM/CRF? | Viterbi (dynamic programming). |
| 22 | LDA: what distribution is used as the prior over topic mixtures? | Dirichlet distribution. |
| 23 | NMF constraint that makes topics interpretable? | Non-negativity of the factor matrices. |
| 24 | What gate mechanism lets LSTMs mitigate vanishing gradients? | Forget/input/output gates with an additive cell-state update. |
| 25 | What's the seq2seq "bottleneck problem"? | Compressing the whole source sequence into one fixed-size context vector loses information, especially for long inputs. |
| 26 | Bahdanau attention uses which decoder state to compute alignment scores? | The previous decoder hidden state (before the current step's update). |
| 27 | Luong attention uses which decoder state? | The current (already updated) decoder hidden state. |
| 28 | Greedy decoding beam width equivalent? | k = 1. |
| 29 | Why does raw beam search score favor short sequences? | Cumulative log-probability only decreases (or stays flat) with more tokens; length normalization corrects this. |
| 30 | BERT pretraining objectives (two, original paper)? | Masked Language Modeling and Next Sentence Prediction. |
| 31 | What attention mask does GPT use during training? | A causal mask, preventing attention to future tokens. |
| 32 | Why can't BERT generate text autoregressively out of the box? | It's trained with full bidirectional attention, incompatible with left-to-right generation constraints. |
| 33 | What is Sentence-BERT's training network structure called? | Siamese (twin) network — same encoder applied independently to each sentence. |
| 34 | Cross-encoder vs bi-encoder: which scales better for large-scale retrieval? | Bi-encoder (independent encoding + fast vector search). |
| 35 | Extractive QA head predicts what per token? | A start-logit and an end-logit. |
| 36 | BLEU is oriented toward precision or recall? | Precision (with a brevity penalty). |
| 37 | ROUGE is oriented toward precision or recall? | Recall (though F-measure variants exist). |
| 38 | What extra capability does METEOR have that BLEU lacks? | Synonym/stemming-aware matching via WordNet, plus a fragmentation penalty. |
| 39 | What's the main risk of generative (vs extractive) QA? | Hallucination — generating unsupported/incorrect content. |
| 40 | What two subtasks make up chatbot NLU? | Intent classification and slot filling. |
| 41 | Cosine similarity range and what it ignores? | -1 to 1; ignores vector magnitude, compares only direction. |
| 42 | Jaccard similarity formula? | `|A ∩ B| / |A ∪ B|`. |
| 43 | What DP algorithm computes edit distance? | Levenshtein dynamic programming, O(m·n). |
| 44 | Why must `[PAD]` tokens be masked in attention? | To prevent the model from attending to or being penalized on meaningless padding positions. |
| 45 | What is "teacher forcing"? | Feeding the ground-truth previous token (not the model's own prediction) as decoder input during training. |
| 46 | Aspect-based sentiment analysis output granularity? | Per-aspect/entity sentiment, not one overall document-level label. |
| 47 | What does RAG stand for and what problem does it address for generative QA? | Retrieval-Augmented Generation; grounds generated answers in retrieved source passages to reduce hallucination. |
| 48 | What kind of nodes does constituency parsing add that dependency parsing doesn't have? | Phrase-structure labels (NP, VP, PP, ...) — dependency parsing has no intermediate phrase nodes, only words. |
| 49 | In dependency parsing, how many heads does each non-root word have? | Exactly one. |
| 50 | What DP algorithm finds the optimal PCFG parse, analogous to Viterbi for trees? | The CKY algorithm. |
| 51 | What graph algorithm finds the optimal non-projective dependency tree? | Chu-Liu/Edmonds' maximum spanning tree algorithm. |
| 52 | What does the Lesk algorithm use to disambiguate word sense? | Word overlap between the context and each candidate sense's dictionary gloss. |
| 53 | Why can't static embeddings perform word sense disambiguation? | They assign one fixed vector per word type regardless of context. |
| 54 | Name the four EDA text augmentation operations. | Synonym replacement, random insertion, random swap, random deletion. |
| 55 | What is back-translation used for in NLP? | Generating paraphrased augmented training data by translating to a pivot language and back. |
| 56 | What does mBERT share across ~104 languages? | A single model, a shared subword vocabulary, and the standard MLM objective, jointly pretrained. |
| 57 | What is zero-shot cross-lingual transfer? | Fine-tuning a multilingual model on one language's labeled data and applying it directly to another language with no labels there. |
| 58 | What is the "curse of multilinguality"? | Fixed model capacity split across many languages, trading off per-language quality for breadth. |
| 59 | What does coreference resolution cluster together? | All mentions in a text that refer to the same real-world entity. |
| 60 | What benchmark tests commonsense-dependent pronoun resolution? | The Winograd Schema Challenge. |
