import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def algorithmPAM(X, k, max_iter = 1000, metric = 'mahalanobis', seed = 0):
    """
        Implementacja algorytmu PAM do klasteryzacji danych.

        Args:
            X (numpy.ndarray): Tablica danych o wymiarach (n_samples, n_features).
                               Zawiera dane wejściowe, które mają zostać podzielone na klastry.
            k (int): Liczba klastrów. Musi być większa od 0 i mniejsza niż liczba punktów w zbiorze danych.
            max_iter (int, opcjonalny): Maksymalna liczba iteracji algorytmu. Domyślnie 1000.
            metric (str, opcjonalny): Metryka odległości używana do obliczeń.
                                      Domyślnie 'mahalanobis'. Może to być również 'euclidean', 'cityblock', itp.
            seed (int, opcjonalny): Ziarno generatora losowego używane do inicjalizacji medoidów. Domyślnie 0.

        Returns:
            tuple:
                - medoids (numpy.ndarray): Indeksy punktów wybranych jako medoidy.
                - clusters (numpy.ndarray): Indeksy klastrów przypisanych do każdego punktu w X.
                - silhouette_scores (list of float): Lista współczynników sylwetki dla każdego punktu w zbiorze danych.

        Raises:
            ValueError: Jeśli:
                - `X` nie jest tablicą numpy.
                - `k` jest mniejsze lub równe 0.
                - `k` jest większe niż liczba punktów w zbiorze danych.

        Szczegóły:
            1. Algorytm inicjalizuje losowe punkty jako medoidy.
            2. W każdej iteracji przypisuje punkty do najbliższego medoidu, tworząc klastry.
            3. Następnie dla każdego klastra wybiera nowy medoid, który minimalizuje sumę odległości wewnątrz klastra.
            4. Proces powtarza się, aż medoidy nie będą zmieniały swoich lokalizacji lub osiągnięta zostanie maksymalna liczba iteracji.
            5. Współczynniki sylwetki są obliczane dla każdego punktu w celu oceny jakości klasteryzacji.
        """
    np.random.seed(seed)
    if not isinstance(X, np.ndarray):
        raise ValueError("X powinno być tablicą numpy")
    if k <= 0 or len(X) < k:
        raise ValueError("Wartość k jest niepoprwana")

    nPoints = len(X)

    medoids = np.random.choice(nPoints, k, replace=False)

    for i in range(max_iter):
        dist = cdist(X, X[medoids], metric=metric)
        clusters = np.argmin(dist, axis = 1)

        newMedoids = np.copy(medoids)
        for c in range(k):
            clustersP = np.where(clusters == c)[0]
            if len(clusters) == 0:
                continue
            clustersD = cdist(X[clustersP], X[clustersP], metric = metric)
            totalDist = np.sum(clustersD, axis = 1)
            bestMedoid = clustersP[np.argmin(totalDist)]
            newMedoids[c] = bestMedoid

            if np.array_equal(medoids, newMedoids):
                break
        medoids = newMedoids

    silhouette_scores = []
    for i in range(nPoints):
        # Odległość do punktów w tej samej grupie
        same_cluster = X[clusters == clusters[i]]
        a_i = np.mean(cdist([X[i]], same_cluster, metric=metric))

        # Odległość do punktów w najbliższej innej grupie
        other_clusters = [X[clusters == c] for c in range(k) if c != clusters[i]]
        b_i = np.min([np.mean(cdist([X[i]], cluster, metric=metric)) for cluster in other_clusters])

        # Obliczanie sylwetki dla punktu i
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_scores.append(s_i)

    return medoids, clusters, silhouette_scores


def test_empty_dataset():
    """Test dla pustego zbioru danych"""
    try:
        X = np.array([])
        algorithmPAM(X, k=3)
    except ValueError as e:
        print("Test passed: ", e)

def test_invalid_k():
    """Test dla niepoprawnej wartości k"""
    try:
        X = np.random.rand(10, 2)
        algorithmPAM(X, k=0)
    except ValueError as e:
        print("Test passed: ", e)

def test_multiple_clusters():
    """Test dla wielu klastrów"""
    X1 = np.random.rand(50, 2) + np.array([0, 0])
    X2 = np.random.rand(50, 2) + np.array([5, 5])
    X3 = np.random.rand(50, 2) + np.array([10, 0])
    X = np.vstack((X1, X2, X3))
    medoids, clusters, scores = algorithmPAM(X, k=3)
    assert len(medoids) == 3, "Niepoprawna liczba medoidów"
    print("Test passed: multiple clusters")

def test_silhouette_scores():
    """Test dla wartości sylwetki"""
    X1 = np.random.rand(50, 2) + np.array([0, 0])
    X2 = np.random.rand(50, 2) + np.array([5, 5])
    X3 = np.random.rand(50, 2) + np.array([10, 0])
    X = np.vstack((X1, X2, X3))
    medoids, clusters, scores = algorithmPAM(X, k=3)
    assert all(-1 <= s <= 1 for s in scores), "Wartości sylwetki powinny być w zakresie [-1, 1]"
    print("Test passed: silhouette scores")

def test_different_metrics():
    """Test dla różnych metryk"""
    X = np.random.rand(50, 2)
    metrics = ['euclidean', 'cityblock', 'cosine']
    for metric in metrics:
        medoids, clusters, scores = algorithmPAM(X, k=3, metric=metric)
        print(f"Test passed: metric {metric}")

def test_large_dataset():
    """Test dla dużego zbioru danych"""
    X = np.random.rand(10000, 10)
    medoids, clusters, scores = algorithmPAM(X, k=5, max_iter=50)
    assert len(medoids) == 5, "Niepoprawna liczba medoidów"
    print("Test passed: large dataset")

test_empty_dataset()
test_invalid_k()
test_multiple_clusters()
test_silhouette_scores()
test_different_metrics()
test_large_dataset()


def test_visualization_2d_clusters():
    """Test z wizualizacją 2D klastrów"""
    X1 = np.random.rand(50, 2) + np.array([0, 0])
    X2 = np.random.rand(50, 2) + np.array([5, 5])
    X3 = np.random.rand(50, 2) + np.array([10, 0])
    X = np.vstack((X1, X2, X3))

    medoids, clusters, _ = algorithmPAM(X, k=3, metric='euclidean', seed=42)

    plt.figure(figsize=(8, 6))
    for cluster_id in range(3):
        cluster_points = X[clusters == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id + 1}')

    plt.scatter(X[medoids, 0], X[medoids, 1], color='red', marker='x', s=150, label='Medoids')
    plt.title('Test: 2D Visualization of Clusters')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid()
    plt.show()

def test_visualization_3d_clusters():
    """Test z wizualizacją 3D klastrów"""
    X1 = np.random.rand(50, 3) + np.array([0, 0, 0])
    X2 = np.random.rand(50, 3) + np.array([5, 5, 5])
    X3 = np.random.rand(50, 3) + np.array([10, 0, 5])
    X = np.vstack((X1, X2, X3))

    medoids, clusters, _ = algorithmPAM(X, k=3, metric='euclidean', seed=42)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_id in range(3):
        cluster_points = X[clusters == cluster_id]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_id + 1}')

    ax.scatter(
        X[medoids, 0], X[medoids, 1], X[medoids, 2],
        color='red', marker='x', s=150, label='Medoids'
    )
    ax.set_title('Test: 3D Visualization of Clusters')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.legend()
    plt.show()

test_visualization_2d_clusters()
test_visualization_3d_clusters()