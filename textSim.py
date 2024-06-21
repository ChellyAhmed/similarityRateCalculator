from sentence_transformers import SentenceTransformer
from arabic import arabicText
from french import frenchText
from spanish import spanishText
from german import germanText
from english import englishText

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two dictionaries of sentences
sentences1 = {
    "Arabic": arabicText,
    "French": frenchText,
    "English": englishText,
    "Spanish": spanishText,
    "German": germanText
}

sentences2 = {
    "Arabic": arabicText,
    "French": frenchText,
    "English": englishText,
    "Spanish": spanishText,
    "German": germanText
}

# Remove words like "the", "a", "an", etc.  from the sentences
# this is a simple way to improve the similarity score by removing common words
# that are not very informative
def remove_stopwords(sentence):
    return ' '.join([word for word in sentence.split() if word not in ['the', 'a', 'an', 'be', 'to', 'of', 'for', 'in', 'that', ]])

sentences1 = {key: remove_stopwords(value) for key, value in sentences1.items()}
sentences2 = {key: remove_stopwords(value) for key, value in sentences2.items()}

# Compute embeddings for both dictionariesge
embeddings1 = model.encode(list(sentences1.values()))
embeddings2 = model.encode(list(sentences2.values()))

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for idx_i, sentence1 in enumerate(sentences1.values()):
    print("Comparing text with:", list(sentences1.keys())[idx_i])
    for idx_j, sentence2 in enumerate(sentences2.values()):
        print(f" - {list(sentences2.keys())[idx_j]}: {similarities[idx_i][idx_j]:.4f}")
