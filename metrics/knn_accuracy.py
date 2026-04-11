from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_accuracy(embeddings, labels, test_embeddings=None, test_labels=None, test_size=0.2):

    if test_embeddings is not None and test_labels is not None:
        X_train, y_train = embeddings, labels
        X_test, y_test = test_embeddings, test_labels
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X_train, y_train)

    score = knn.score(X_test, y_test)
    return score
