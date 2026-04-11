from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression

def linear_probe(clean_embeddings, clean_labels, test_embedding, test_label):

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(clean_embeddings, clean_labels)

    preds = clf.predict(test_embedding)
    acc = accuracy_score(test_label, preds)
    return acc