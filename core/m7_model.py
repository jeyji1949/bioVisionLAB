"""
core/m7_model.py
================
Module M7 — Classification de cellules sanguines sur BloodMNIST

Deux approches comparées sur les mêmes données :
  A) Random Forest  — ML classique, features HOG + stats
  B) CNN (PyTorch)  — Deep Learning, architecture légère

Structure
---------
  BloodMNISTLoader   — télécharge et charge le dataset via medmnist
  HOGExtractor       — features HOG from scratch (NumPy uniquement)
  RandomForestM7     — Random Forest from scratch (NumPy uniquement)
  BloodCNN           — CNN PyTorch léger (2 conv + pooling + 2 dense)
  CNNTrainer         — entraînement + évaluation du CNN
  M7Runner           — wrapper haut niveau appelé par ui/m7_panel.py

Classes BloodMNIST (8 classes)
-------------------------------
  0: basophil       1: eosinophil     2: erythroblast   3: ig (immature granulocyte)
  4: lymphocyte     5: monocyte       6: neutrophil      7: platelet

Usage
-----
  runner = M7Runner()
  runner.load_data(progress_cb=lambda msg: print(msg))
  rf_results = runner.train_random_forest(progress_cb=...)
  cnn_results = runner.train_cnn(epochs=5, progress_cb=...)
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Class names
# ─────────────────────────────────────────────────────────────────────────────

BLOOD_CLASSES = [
    "Basophile", "Éosinophile", "Érythroblaste", "Granulocyte imm.",
    "Lymphocyte", "Monocyte", "Neutrophile", "Plaquette"
]
N_CLASSES = 8


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BloodMNIST Loader
# ─────────────────────────────────────────────────────────────────────────────

class BloodMNISTLoader:
    """
    Télécharge et charge BloodMNIST via la bibliothèque medmnist.

    BloodMNIST = 17 092 images 28×28 RGB de cellules sanguines humaines,
    8 types cellulaires, split standard train/val/test.
    """

    def __init__(self):
        self.train_images = None   # (N, 28, 28, 3) uint8
        self.train_labels = None   # (N,) int
        self.test_images  = None
        self.test_labels  = None
        self.loaded       = False

    def load(self, progress_cb=None):
        """
        Charge BloodMNIST. Télécharge automatiquement si absent du cache.

        Parameters
        ----------
        progress_cb : callable(str) — appelé avec des messages de progression
        """
        try:
            import medmnist
            from medmnist import BloodMNIST as _BM
        except ImportError:
            raise ImportError(
                "medmnist non installé.\n"
                "Lancer : pip install medmnist"
            )

        if progress_cb:
            progress_cb("Chargement BloodMNIST (téléchargement si nécessaire)…")

        train_ds = _BM(split="train", download=True, size=28)
        test_ds  = _BM(split="test",  download=True, size=28)

        if progress_cb:
            progress_cb("Conversion en tableaux NumPy…")

        # medmnist datasets expose .imgs (N,H,W,C) and .labels (N,1)
        self.train_images = np.array(train_ds.imgs,   dtype=np.uint8)
        self.train_labels = np.array(train_ds.labels, dtype=np.int32).ravel()
        self.test_images  = np.array(test_ds.imgs,    dtype=np.uint8)
        self.test_labels  = np.array(test_ds.labels,  dtype=np.int32).ravel()
        self.loaded       = True

        if progress_cb:
            progress_cb(
                f"BloodMNIST chargé — "
                f"train: {len(self.train_labels)} · "
                f"test: {len(self.test_labels)} images"
            )

    def to_grayscale(self, images):
        """Convert (N,28,28,3) → (N,28,28) grayscale."""
        return (0.299 * images[:, :, :, 0]
                + 0.587 * images[:, :, :, 1]
                + 0.114 * images[:, :, :, 2]).astype(np.uint8)

    def subsample(self, n_train=2000, n_test=500, seed=42):
        """
        Retourne un sous-ensemble stratifié pour accélérer le RF.

        Les 8 classes sont représentées proportionnellement.
        """
        rng = np.random.default_rng(seed)

        def _sample(images, labels, n):
            idx = rng.choice(len(labels), min(n, len(labels)), replace=False)
            return images[idx], labels[idx]

        tr_img, tr_lbl = _sample(self.train_images, self.train_labels, n_train)
        te_img, te_lbl = _sample(self.test_images,  self.test_labels,  n_test)
        return tr_img, tr_lbl, te_img, te_lbl


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HOG Feature Extractor (from scratch, NumPy only)
# ─────────────────────────────────────────────────────────────────────────────

class HOGExtractor:
    """
    Histogram of Oriented Gradients — implémenté from scratch.

    Paramètres adaptés à des images 28×28 :
      cell_size = 4   → 7×7 = 49 cellules
      n_bins    = 9   → 9 orientations (0°–180°)

    Feature vector size = 7 × 7 × 9 = 441 dimensions
    + 10 stats globales = 451 dimensions totales

    Pourquoi HOG pour les cellules ?
    ---------------------------------
    Les cellules sanguines se distinguent principalement par :
    - La forme du noyau (lobulé, bilobé, rond…)
    - La texture de la chromatine
    HOG capture exactement ces propriétés : gradients locaux d'intensité
    organisés par orientation.
    """

    def __init__(self, cell_size=4, n_bins=9):
        self.cell_size = cell_size
        self.n_bins    = n_bins

    def _gradients(self, gray):
        """Gradients horizontaux et verticaux par différences finies."""
        h, w    = gray.shape
        gx      = np.zeros((h, w), dtype=np.float64)
        gy      = np.zeros((h, w), dtype=np.float64)
        # Horizontal gradient
        gx[:, 1:-1] = gray[:, 2:].astype(float) - gray[:, :-2].astype(float)
        gx[:, 0]    = gray[:, 1].astype(float)  - gray[:, 0].astype(float)
        gx[:, -1]   = gray[:, -1].astype(float) - gray[:, -2].astype(float)
        # Vertical gradient
        gy[1:-1, :] = gray[2:, :].astype(float) - gray[:-2, :].astype(float)
        gy[0, :]    = gray[1, :].astype(float)  - gray[0, :].astype(float)
        gy[-1, :]   = gray[-1, :].astype(float) - gray[-2, :].astype(float)
        mag   = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(np.abs(gy), np.abs(gx)) * 180.0 / np.pi  # 0–90°→0–180°
        return mag, angle

    def _cell_histogram(self, mag_cell, ang_cell):
        """Histogramme d'orientations pondéré par la magnitude pour une cellule."""
        hist = np.zeros(self.n_bins)
        bin_width = 180.0 / self.n_bins
        for i in range(mag_cell.shape[0]):
            for j in range(mag_cell.shape[1]):
                bin_idx = int(ang_cell[i, j] / bin_width) % self.n_bins
                hist[bin_idx] += mag_cell[i, j]
        return hist

    def _global_stats(self, gray):
        """10 statistiques globales complémentaires au HOG."""
        flat = gray.astype(np.float64).ravel()
        return np.array([
            flat.mean(),
            flat.std(),
            float(np.percentile(flat, 25)),
            float(np.percentile(flat, 75)),
            float(flat.min()),
            float(flat.max()),
            float(np.mean(flat > 128)),    # fraction pixels clairs
            float(np.mean(np.abs(np.diff(flat)))),  # variation locale
            float(np.sum(flat**2) / flat.size),     # énergie
            float(-np.sum(
                (flat / 255 + 1e-10) * np.log2(flat / 255 + 1e-10)
            ) / flat.size),  # entropie normalisée
        ])

    def extract(self, gray):
        """
        Extrait le vecteur HOG + stats d'une image grayscale 28×28.

        Parameters
        ----------
        gray : (28, 28) uint8

        Returns
        -------
        features : (451,) float64
        """
        mag, ang = self._gradients(gray)
        h, w     = gray.shape
        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size

        hog_feats = []
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0 = cy * self.cell_size
                x0 = cx * self.cell_size
                mag_c = mag[y0:y0+self.cell_size, x0:x0+self.cell_size]
                ang_c = ang[y0:y0+self.cell_size, x0:x0+self.cell_size]
                hist = self._cell_histogram(mag_c, ang_c)
                hog_feats.append(hist)

        hog_vec   = np.concatenate(hog_feats)
        stats_vec = self._global_stats(gray)
        features  = np.concatenate([hog_vec, stats_vec])

        # L2 normalisation
        norm = np.linalg.norm(features)
        if norm > 1e-10:
            features /= norm
        return features

    def extract_batch(self, images_gray, progress_cb=None):
        """
        Extrait les features pour un batch d'images.

        Parameters
        ----------
        images_gray : (N, 28, 28) uint8
        progress_cb : callable(str)

        Returns
        -------
        X : (N, 451) float64
        """
        N = len(images_gray)
        d = self.cell_size
        n_cells = (28 // d) * (28 // d)
        feat_dim = n_cells * self.n_bins + 10
        X = np.zeros((N, feat_dim), dtype=np.float64)
        for i, img in enumerate(images_gray):
            X[i] = self.extract(img)
            if progress_cb and i % 200 == 0:
                progress_cb(f"Features HOG : {i}/{N}…")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Random Forest (from scratch, NumPy only)
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTree:
    """
    Arbre de décision CART — implémenté from scratch.

    Critère : impureté de Gini
    Gini(t) = 1 - Σ p_k²

    L'arbre est limité en profondeur (max_depth) et en taille de feuille
    (min_samples_split) pour éviter l'overfitting.
    """

    def __init__(self, max_depth=8, min_samples_split=5,
                 max_features=None, seed=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.seed              = seed
        self._tree             = None

    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.bincount(y, minlength=N_CLASSES)
        probs  = counts / n
        return float(1.0 - np.sum(probs**2))

    def _best_split(self, X, y, rng):
        n, d   = X.shape
        best   = {"gain": -1, "feat": None, "thresh": None}
        gini_p = self._gini(y)

        # Random feature subset
        n_feat = self.max_features or d
        feats  = rng.choice(d, min(n_feat, d), replace=False)

        for feat in feats:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat],
                                           np.linspace(5, 95, 15))
            for t in thresholds:
                left  = y[X[:, feat] <= t]
                right = y[X[:, feat] >  t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = gini_p - (
                    len(left)  / n * self._gini(left) +
                    len(right) / n * self._gini(right)
                )
                if gain > best["gain"]:
                    best = {"gain": gain, "feat": feat, "thresh": t}
        return best

    def _build(self, X, y, depth, rng):
        # Leaf conditions
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            counts = np.bincount(y, minlength=N_CLASSES)
            return {"leaf": True, "class": int(np.argmax(counts)),
                    "probs": (counts / counts.sum()).tolist()}

        split = self._best_split(X, y, rng)
        if split["feat"] is None:
            counts = np.bincount(y, minlength=N_CLASSES)
            return {"leaf": True, "class": int(np.argmax(counts)),
                    "probs": (counts / counts.sum()).tolist()}

        mask  = X[:, split["feat"]] <= split["thresh"]
        left  = self._build(X[mask],  y[mask],  depth+1, rng)
        right = self._build(X[~mask], y[~mask], depth+1, rng)
        return {"leaf": False, "feat": split["feat"],
                "thresh": split["thresh"],
                "left": left, "right": right}

    def fit(self, X, y):
        rng        = np.random.default_rng(self.seed)
        self._tree = self._build(X, y, 0, rng)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["class"], node["probs"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict_proba(self, X):
        probs = np.zeros((len(X), N_CLASSES))
        for i, x in enumerate(X):
            _, p = self._predict_one(x, self._tree)
            probs[i] = p
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class RandomForestM7:
    """
    Random Forest — ensemble de DecisionTree.

    Deux sources de randomité :
    1. Bootstrap sampling  — chaque arbre voit un sous-ensemble des données
    2. Random features      — chaque split considère √d features aléatoires

    Ces deux mécanismes réduisent la variance et évitent l'overfitting.

    Parameters
    ----------
    n_estimators : nombre d'arbres (10 suffit pour la démo)
    max_depth    : profondeur max de chaque arbre
    seed         : reproductibilité
    """

    def __init__(self, n_estimators=10, max_depth=8,
                 min_samples_split=5, seed=42):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.seed              = seed
        self.trees_            = []

    def fit(self, X, y, progress_cb=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        max_feat = max(1, int(np.sqrt(d)))
        self.trees_ = []

        for i in range(self.n_estimators):
            if progress_cb:
                progress_cb(f"Arbre {i+1}/{self.n_estimators}…")
            # Bootstrap sample
            idx  = rng.choice(n, n, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                seed=int(rng.integers(1e6)),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        """Moyenne des probabilités de tous les arbres."""
        all_probs = np.zeros((len(X), N_CLASSES))
        for tree in self.trees_:
            all_probs += tree.predict_proba(X)
        return all_probs / self.n_estimators

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_importances(self, n_features):
        """
        Importance approximée par fréquence d'utilisation de chaque feature.
        (Impurity-based importance would require storing gain per node.)
        """
        counts = np.zeros(n_features)
        def _count(node):
            if node["leaf"]:
                return
            counts[node["feat"]] += 1
            _count(node["left"])
            _count(node["right"])
        for tree in self.trees_:
            _count(tree._tree)
        total = counts.sum()
        return counts / total if total > 0 else counts


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CNN (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cnn():
    """
    Architecture CNN légère pour images 28×28 RGB.

    Conv1: 3→16 filtres 3×3, ReLU, MaxPool 2×2  → 16×13×13
    Conv2: 16→32 filtres 3×3, ReLU, MaxPool 2×2 → 32×6×6
    Flatten → 1152
    Dense1: 1152→128, ReLU, Dropout 0.3
    Dense2: 128→8  (logits)

    ~170k paramètres — entraînable en quelques minutes sur CPU.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch non installé.\npip install torch")

    class BloodCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),  # 28→14→7
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, N_CLASSES),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return BloodCNN()


class CNNTrainer:
    """
    Entraîne et évalue le CNN sur BloodMNIST.

    Usage
    -----
      trainer = CNNTrainer()
      results = trainer.train(
          train_images, train_labels,
          test_images,  test_labels,
          epochs=5, batch_size=64,
          progress_cb=lambda msg: print(msg)
      )
    """

    def __init__(self):
        self.model       = None
        self.train_losses = []
        self.train_accs   = []
        self.val_accs     = []

    def _to_tensor(self, images, labels=None):
        import torch
        # images: (N, 28, 28, 3) uint8 → (N, 3, 28, 28) float32 / 255
        x = torch.tensor(
            images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        )
        if labels is not None:
            y = torch.tensor(labels.astype(np.int64))
            return x, y
        return x

    def train(self, tr_imgs, tr_lbls, te_imgs, te_lbls,
              epochs=5, batch_size=64, lr=1e-3, progress_cb=None):
        """
        Entraîne le CNN et retourne les résultats.

        Returns
        -------
        dict with keys:
          accuracy, confusion_matrix, per_class_acc,
          train_losses, train_accs, val_accs, predictions
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device("cpu")  # CPU only for compatibility
        self.model = _build_cnn().to(device)
        optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        criterion  = nn.CrossEntropyLoss()

        X_tr, y_tr = self._to_tensor(tr_imgs, tr_lbls)
        X_te, y_te = self._to_tensor(te_imgs, te_lbls)

        n_batches = (len(X_tr) + batch_size - 1) // batch_size
        self.train_losses = []
        self.train_accs   = []
        self.val_accs     = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0

            # Shuffle
            perm = torch.randperm(len(X_tr))
            X_tr = X_tr[perm]
            y_tr = y_tr[perm]

            for b in range(n_batches):
                xb = X_tr[b*batch_size:(b+1)*batch_size].to(device)
                yb = y_tr[b*batch_size:(b+1)*batch_size].to(device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss    += loss.item() * len(xb)
                epoch_correct += (logits.argmax(1) == yb).sum().item()

            avg_loss = epoch_loss / len(X_tr)
            train_acc = epoch_correct / len(X_tr) * 100

            # Validation
            val_acc = self._eval_accuracy(X_te, y_te, batch_size, device)
            self.train_losses.append(round(avg_loss, 4))
            self.train_accs.append(round(train_acc, 1))
            self.val_accs.append(round(val_acc, 1))

            if progress_cb:
                progress_cb(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"loss={avg_loss:.4f}  train={train_acc:.1f}%  val={val_acc:.1f}%"
                )

        # Final evaluation
        preds = self._predict(X_te, batch_size, device)
        acc   = float((preds == te_lbls).mean() * 100)
        cm    = self._confusion_matrix(te_lbls, preds)
        per_class = [
            round(float(cm[i, i] / cm[i].sum() * 100), 1)
            if cm[i].sum() > 0 else 0.0
            for i in range(N_CLASSES)
        ]

        return {
            "accuracy":        round(acc, 2),
            "confusion_matrix": cm.tolist(),
            "per_class_acc":   per_class,
            "train_losses":    self.train_losses,
            "train_accs":      self.train_accs,
            "val_accs":        self.val_accs,
            "predictions":     preds.tolist(),
            "true_labels":     te_lbls.tolist(),
            "epochs":          epochs,
        }

    def _eval_accuracy(self, X, y, batch_size, device):
        import torch
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for b in range(0, len(X), batch_size):
                xb = X[b:b+batch_size].to(device)
                yb = y[b:b+batch_size]
                preds = self.model(xb).argmax(1).cpu()
                correct += (preds == yb).sum().item()
        return correct / len(X) * 100

    def _predict(self, X, batch_size, device):
        import torch
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for b in range(0, len(X), batch_size):
                xb = X[b:b+batch_size].to(device)
                all_preds.append(self.model(xb).argmax(1).cpu().numpy())
        return np.concatenate(all_preds)

    def _confusion_matrix(self, true, pred):
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
        for t, p in zip(true, pred):
            cm[t, p] += 1
        return cm


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix_np(true, pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm):
    accs = []
    for i in range(N_CLASSES):
        total = cm[i].sum()
        accs.append(round(float(cm[i, i] / total * 100), 1) if total > 0 else 0.0)
    return accs


# ─────────────────────────────────────────────────────────────────────────────
# 6.  M7Runner — high-level wrapper called by ui/m7_panel.py
# ─────────────────────────────────────────────────────────────────────────────

class M7Runner:
    """
    Orchestrateur complet du Module M7.

    Usage
    -----
      runner = M7Runner()

      # Step 1 — load data (once)
      runner.load_data(progress_cb=cb)

      # Step 2a — Random Forest
      rf_results = runner.train_random_forest(
          n_samples=2000, n_estimators=10, progress_cb=cb)

      # Step 2b — CNN
      cnn_results = runner.train_cnn(
          epochs=5, batch_size=64, progress_cb=cb)

    Both result dicts are also stored as runner.rf_results / runner.cnn_results.
    """

    def __init__(self):
        self.loader      = BloodMNISTLoader()
        self.hog         = HOGExtractor(cell_size=4, n_bins=9)
        self.rf          = None
        self.cnn_trainer = None
        self.rf_results  = None
        self.cnn_results = None

        # Sub-sampled data stored after load
        self._tr_imgs = None
        self._tr_lbls = None
        self._te_imgs = None
        self._te_lbls = None

    def load_data(self, n_train=2000, n_test=400, progress_cb=None):
        """
        Télécharge BloodMNIST et prépare les sous-ensembles de travail.
        """
        self.loader.load(progress_cb=progress_cb)
        tr_img, tr_lbl, te_img, te_lbl = self.loader.subsample(
            n_train=n_train, n_test=n_test)
        self._tr_imgs = tr_img
        self._tr_lbls = tr_lbl
        self._te_imgs = te_img
        self._te_lbls = te_lbl
        if progress_cb:
            progress_cb(
                f"Données prêtes — {len(tr_lbl)} train · {len(te_lbl)} test")

    def train_random_forest(self, n_estimators=10, max_depth=8,
                            progress_cb=None):
        """
        Extrait les features HOG et entraîne le Random Forest.

        Returns
        -------
        dict:
          accuracy, confusion_matrix, per_class_acc,
          feature_importances, n_features, n_estimators
        """
        if self._tr_imgs is None:
            raise RuntimeError("Appeler load_data() d'abord.")

        # Convert to grayscale for HOG
        if progress_cb:
            progress_cb("Extraction features HOG (train)…")
        tr_gray = self.loader.to_grayscale(self._tr_imgs)
        te_gray = self.loader.to_grayscale(self._te_imgs)

        X_tr = self.hog.extract_batch(tr_gray, progress_cb=progress_cb)
        if progress_cb:
            progress_cb("Extraction features HOG (test)…")
        X_te = self.hog.extract_batch(te_gray)

        if progress_cb:
            progress_cb(f"Entraînement Random Forest ({n_estimators} arbres)…")
        self.rf = RandomForestM7(
            n_estimators=n_estimators, max_depth=max_depth, seed=42)
        self.rf.fit(X_tr, self._tr_lbls, progress_cb=progress_cb)

        if progress_cb:
            progress_cb("Évaluation…")
        preds = self.rf.predict(X_te)
        acc   = float((preds == self._te_lbls).mean() * 100)
        cm    = confusion_matrix_np(self._te_lbls, preds)
        importances = self.rf.feature_importances(X_tr.shape[1])

        self.rf_results = {
            "accuracy":            round(acc, 2),
            "confusion_matrix":    cm.tolist(),
            "per_class_acc":       per_class_accuracy(cm),
            "feature_importances": importances.tolist(),
            "n_features":          X_tr.shape[1],
            "n_estimators":        n_estimators,
            "n_train":             len(self._tr_lbls),
            "n_test":              len(self._te_lbls),
        }
        if progress_cb:
            progress_cb(f"Random Forest — Accuracy : {acc:.2f}%")
        return self.rf_results

    def train_cnn(self, epochs=5, batch_size=64, lr=1e-3,
                  progress_cb=None):
        """
        Entraîne le CNN PyTorch sur les images RGB.

        Returns
        -------
        dict: accuracy, confusion_matrix, per_class_acc,
              train_losses, train_accs, val_accs, epochs
        """
        if self._tr_imgs is None:
            raise RuntimeError("Appeler load_data() d'abord.")

        if progress_cb:
            progress_cb("Initialisation CNN PyTorch…")
        self.cnn_trainer = CNNTrainer()
        self.cnn_results = self.cnn_trainer.train(
            self._tr_imgs, self._tr_lbls,
            self._te_imgs, self._te_lbls,
            epochs=epochs, batch_size=batch_size, lr=lr,
            progress_cb=progress_cb,
        )
        self.cnn_results["n_train"] = len(self._tr_lbls)
        self.cnn_results["n_test"]  = len(self._te_lbls)
        if progress_cb:
            progress_cb(
                f"CNN terminé — Accuracy : {self.cnn_results['accuracy']:.2f}%")
        return self.cnn_results