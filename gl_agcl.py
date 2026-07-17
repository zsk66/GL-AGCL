# -*- coding: utf-8 -*-
"""
GL-AGCL: Global-Local Anchor Graph Collaborative Learning
=========================================================
Reference implementation for the paper

    "Global-Local Anchor Graph Collaborative Learning for Scalable
     Incomplete Multi-View Clustering"

This is an implementation of the
GL-AGCL method and its block-coordinate-descent (BCD) solver. Every subproblem
(P, A, Y, L, G) is solved in closed form; the sparse-simplex projection uses the
linear-time root-finding described in the paper.

Model (per view v):
    min   sum_v { ||P^(v) X^(v) - A^(v) (G + L^(v)) I^(v)||_F^2
                  + gamma ||P^(v)||_F^2 }
        + beta sum_{u != v} ||L^(u) - L^(v)||_F^2
        + beta_graph tr(G L_S G^T)                  (optional KNN smoothness)
    s.t.  P^(v) Q^(v) P^(v)T = I,   A^(v)T A^(v) = I,
          Y^(v) = G + L^(v),  Y^(v)T 1 = 1,  Y^(v) >= 0,
          ||Y^(v)_{:,i}||_0 <= kappa.

"""

import argparse
import warnings

import numpy as np
from scipy.linalg import svd, eigh
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


# ============================================================================
# Clustering metrics
# ============================================================================
def clustering_accuracy(y_true, y_pred):
    """Clustering ACC via optimal (Hungarian) label matching."""
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def purity_score(y_true, y_pred):
    """Clustering purity."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = y_true.size
    n_clusters = int(y_pred.max() + 1)
    total_correct = 0
    for k in range(n_clusters):
        cluster_mask = (y_pred == k)
        if cluster_mask.sum() == 0:
            continue
        cluster_labels = y_true[cluster_mask]
        most_frequent = np.bincount(cluster_labels.astype(int)).argmax()
        total_correct += (cluster_labels == most_frequent).sum()
    return total_correct / n_samples


# ============================================================================
# GL-AGCL model
# ============================================================================
class GLAGCL:
    """Global-Local Anchor Graph Collaborative Learning.

    Parameters
    ----------
    n_clusters : int          final number of clusters k.
    n_anchors : int           number of anchors s.
    latent_dim : int or None  latent feature dimension d (defaults to n_clusters).
    gamma : float             conjugate-projection ridge weight gamma_v.
    beta : float              diversity-penalty weight between local graphs.
    theta_init : float        initial quadratic-penalty weight theta.
    rho : float               multiplicative growth of theta per iteration (>1 to grow).
    max_iter : int            max BCD iterations.
    tol : float               relative-objective stopping tolerance.
    kappa : int or None       cardinality (nonzeros per column of Y); None -> n_clusters.
    beta_graph : float        KNN Laplacian smoothness weight on G.
    k_nn : int                neighbors for the KNN graph.
    use_graph : bool          enable the KNN smoothness term.
    n_kmeans_runs : int       n_init for the final KMeans.
    normalize_data : bool     column-wise L2 normalization of inputs.
    random_state : int or None
    verbose : bool
    """

    def __init__(self, n_clusters=10, n_anchors=100, latent_dim=None,
                 gamma=1.0, beta=1.0, theta_init=0.1, rho=1.0,
                 max_iter=50, tol=1e-6, kappa=None,
                 beta_graph=1.0, k_nn=3, use_graph=True,
                 n_kmeans_runs=20, normalize_data=True,
                 auto_latent_dim=True, random_state=None, verbose=True):
        self.k = n_clusters
        self.s = n_anchors
        self.gamma = gamma
        self.beta = beta
        self.theta_init = theta_init
        self.theta = theta_init
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        # kappa: cardinality of each Y column (paper's kappa). Falls back to k.
        self.kappa = kappa if kappa is not None else n_clusters
        self.beta_graph = beta_graph
        self.k_nn = k_nn
        self.use_graph = use_graph
        self.n_kmeans_runs = n_kmeans_runs
        self.normalize_data = normalize_data
        self.auto_latent_dim = auto_latent_dim
        self.random_state = random_state
        self.verbose = verbose

        if auto_latent_dim or latent_dim is None:
            self.d = n_clusters
        else:
            self.d = latent_dim

    # ------------------------------------------------------------------ fit
    def fit(self, X_views, I_views=None, y_true=None):
        """Run BCD. Returns the best (lowest-NCut) cluster labels.

        X_views : list of (d_v, n_v) arrays  (features x available-samples).
        I_views : list of (n, n_v) 0/1 indicator matrices; None -> complete data.
        y_true  : optional ground-truth labels for progress reporting.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.V = len(X_views)
        self.n = X_views[0].shape[1] if I_views is None else I_views[0].shape[0]

        if I_views is None:
            I_views = [np.eye(self.n) for _ in range(self.V)]

        # Column-wise L2 normalization.
        if self.normalize_data:
            X_normed = []
            for v in range(self.V):
                X_v = X_views[v].copy().astype(np.float64)
                col_norms = np.sqrt(np.maximum(1e-14, np.sum(X_v ** 2, axis=0, keepdims=True)))
                X_normed.append(X_v / col_norms)
            self.X_views = X_normed
        else:
            self.X_views = [X_v.copy().astype(np.float64) for X_v in X_views]

        self.I_views = I_views

        # Visibility masks (which of the n complete samples each view observes).
        self.vis_masks, self.vis_float = [], []
        for v in range(self.V):
            vis = (I_views[v].sum(axis=1) > 0)
            self.vis_masks.append(vis)
            self.vis_float.append(vis.astype(np.float64))

        if self.verbose:
            print("  [Config] V=%d n=%d latent_dim=%d n_anchors=%d kappa=%d kmeans=%d"
                  % (self.V, self.n, self.d, self.s, self.kappa, self.n_kmeans_runs))
            if self.use_graph:
                print("  [Graph] beta_graph=%.3g k_nn=%d" % (self.beta_graph, self.k_nn))

        if self.use_graph:
            self._build_knn_graphs()

        self._initialize_variables()
        self._precompute_G_solver()
        self.theta = self.theta_init

        self.history = {"obj": [], "ACC": [], "NMI": [], "ncut": [], "best_iter": []}
        self._best_labels = None
        self._best_ncut = np.inf
        self._best_iter = -1

        for iteration in range(self.max_iter):
            obj_old = self._compute_objective()

            # P-step is guarded: reject if it fails to decrease the objective.
            P_old = [P.copy() for P in self.P_views]
            PXI_old = [pxi.copy() for pxi in self.PXI]
            self._update_P()
            if self._compute_objective() > obj_old:
                self.P_views = P_old
                self.PXI = PXI_old

            self._update_A()
            self._update_Y()
            self._update_L()
            self._update_G()

            if self.rho > 1.0:
                self.theta = min(self.theta * self.rho, 10.0)
                self._precompute_G_solver()

            obj_new = self._compute_objective()
            rel_change = abs(obj_new - obj_old) / (abs(obj_old) + 1e-10)

            # Cluster on the consensus anchor graph S = G^T G; keep best NCut.
            sample_every = 5 if self.n >= 500 else 1
            do_cluster = ((iteration + 1) % sample_every == 0
                          or iteration == 0 or iteration == self.max_iter - 1)
            if do_cluster:
                S = self.G.T @ self.G
                S = np.maximum((S + S.T) / 2, 0)
                iter_labels = self._spectral_clustering(S)
                ncut_val = self._normalized_cut(S, iter_labels)
            else:
                iter_labels = (self._best_labels if self._best_labels is not None
                               else np.zeros(self.n, dtype=np.int64))
                ncut_val = self._best_ncut

            if ncut_val < self._best_ncut:
                self._best_ncut = ncut_val
                self._best_labels = iter_labels.copy()
                self._best_iter = iteration + 1

            if y_true is not None:
                iter_acc = clustering_accuracy(y_true, iter_labels)
                iter_nmi = normalized_mutual_info_score(y_true, iter_labels)
                best_acc = clustering_accuracy(y_true, self._best_labels)
            else:
                iter_acc = iter_nmi = best_acc = 0.0

            self.history["obj"].append(obj_new)
            self.history["ACC"].append(iter_acc)
            self.history["NMI"].append(iter_nmi)
            self.history["ncut"].append(ncut_val)
            self.history["best_iter"].append(self._best_iter)

            if self.verbose and y_true is not None:
                marker = " *" if self._best_iter == iteration + 1 else ""
                print("  Iter %3d/%d: obj=%.2f ACC=%.2f%% NMI=%.2f%% ncut=%.4f [best@%d %.2f%%]%s"
                      % (iteration + 1, self.max_iter, obj_new, iter_acc * 100,
                         iter_nmi * 100, ncut_val, self._best_iter, best_acc * 100, marker))

            if rel_change < self.tol and iteration > 10:
                if self.verbose:
                    print("  Converged at iter %d" % (iteration + 1))
                break

        if self.verbose:
            print("  >>> Best iter %d (ncut=%.4f)" % (self._best_iter, self._best_ncut))
        return self._best_labels


    # ---------------------------------------------- KNN graph Laplacian (sparse)
    def _build_knn_graphs(self):
        """Sum of per-view KNN Laplacians lifted to the n-sample space (sparse)."""
        S_sum = sp.csr_matrix((self.n, self.n))
        for v in range(self.V):
            X_v_T = self.X_views[v].T  # n_v x d_v
            n_v = X_v_T.shape[0]
            actual_k = min(self.k_nn, n_v - 1)
            if actual_k < 1:
                continue
            W = kneighbors_graph(X_v_T, n_neighbors=actual_k,
                                 mode="connectivity", include_self=False)
            W = (W + W.T) * 0.5
            G_v = sp.csr_matrix(self.I_views[v])   # n x n_v
            S_sum = S_sum + G_v @ W @ G_v.T        # n x n (sparse)

        S_sum = S_sum.tocsr()
        D_vec = np.asarray(S_sum.sum(axis=1)).ravel()
        self.L_graph_sum = (sp.diags(D_vec) - S_sum).tocsr()
        if self.verbose:
            print("  [Graph] L_S nnz=%d (%.2f%%)"
                  % (self.L_graph_sum.nnz,
                     100.0 * self.L_graph_sum.nnz / (self.n * self.n + 1e-9)))

    # ---------------------------------------------- G-update solver (sparse LU)
    def _precompute_G_solver(self):
        """Factor M = 2 diag(vis_count) + theta*V*I + 2 beta_graph L_S once per theta."""
        vis_count = np.zeros(self.n)
        for v in range(self.V):
            vis_count += self.vis_float[v]
        M = sp.diags(2.0 * vis_count + self.theta * self.V)
        if self.use_graph:
            M = M + 2.0 * self.beta_graph * self.L_graph_sum
        self._M_sp = M.tocsc()
        self._M_lu = splu(self._M_sp)

    # ---------------------------------------------- initialization
    def _initialize_variables(self):
        """Thin-SVD based init that evaluates Q^{-1/2} implicitly (never forms d_v x d_v)."""
        self.QXI_views = []      # Q^{-1/2} X I^T   (d_v x n)
        self._XIv_views = []     #        X I^T     (d_v x n)
        self._IXv_views = []     #  (X I^T)^T       (n x d_v)
        self._Ux_views = []      # left singulars of X_v
        self._coef2_views = []   # implicit Q^{-1/2} coefficients
        self._inv_sqrt_gamma = 1.0 / np.sqrt(self.gamma + 1e-14)

        self.P_views = []
        for v in range(self.V):
            X_v = self.X_views[v]
            d_v, n_v = X_v.shape
            vis_idx = np.where(self.vis_masks[v])[0]

            U_x, sigma, Vt_x = np.linalg.svd(X_v, full_matrices=False)
            denom_eig = np.sqrt(sigma ** 2 / n_v + self.gamma + 1e-14)
            coef = sigma / denom_eig
            coef2 = 1.0 / denom_eig - self._inv_sqrt_gamma
            self._Ux_views.append(U_x)
            self._coef2_views.append(coef2)

            QX_v = (U_x * coef[np.newaxis, :]) @ Vt_x        # Q^{-1/2} X_v
            QXI = np.zeros((d_v, self.n))
            QXI[:, vis_idx] = QX_v
            self.QXI_views.append(np.nan_to_num(QXI))

            XIv = np.zeros((d_v, self.n))
            XIv[:, vis_idx] = X_v
            self._XIv_views.append(XIv)
            self._IXv_views.append(XIv.T)

            d_use = min(self.d, d_v)
            P_v = U_x[:, :d_use].T
            if d_use < self.d:
                P_v = np.vstack([P_v, np.zeros((self.d - d_use, d_v))])
            self.P_views.append(P_v)

        # Initialize G via KMeans on view-0 features (better than random).
        X0 = self.X_views[0].T
        if X0.shape[0] >= self.s:
            km = KMeans(n_clusters=self.s, n_init=3, max_iter=100,
                        random_state=self.random_state)
            km.fit(X0)
            from scipy.spatial.distance import cdist
            dists = cdist(X0, km.cluster_centers_, metric="euclidean")
            G_init = np.zeros((self.s, self.n))
            kk = min(self.kappa, self.s)
            vis_idx_v0 = np.where(self.vis_masks[0])[0]
            if vis_idx_v0.size > 0:
                nearest = np.argpartition(dists, kk - 1, axis=1)[:, :kk]
                d_near = np.take_along_axis(dists, nearest, axis=1)
                w = 1.0 / (d_near + 1e-10)
                w /= w.sum(axis=1, keepdims=True)
                rows = nearest.reshape(-1)
                cols_rep = np.repeat(vis_idx_v0, kk)
                G_init[rows, cols_rep] = w.reshape(-1)
            for i in np.where(~self.vis_masks[0])[0]:
                idx = np.random.choice(self.s, kk, replace=False)
                vals = np.random.rand(kk); vals /= vals.sum()
                G_init[idx, i] = vals
            self.G = G_init
        else:
            self.G = np.zeros((self.s, self.n))
            kk = min(self.kappa, self.s)
            for i in range(self.n):
                idx = np.random.choice(self.s, kk, replace=False)
                vals = np.random.rand(kk); vals /= vals.sum()
                self.G[idx, i] = vals

        self.L_views = [np.zeros((self.s, self.n)) for _ in range(self.V)]
        self.Y_views = [self.G.copy() for _ in range(self.V)]

        # A^(v) init: orthonormal factor of the least-squares map P X I^T -> G.
        self.A_views = []
        for v in range(self.V):
            try:
                PXI = self.P_views[v] @ self._XIv_views[v]
                GGT = self.G @ self.G.T + 1e-6 * np.eye(self.s)
                A_init = PXI @ self.G.T @ np.linalg.inv(GGT)
                U, _, Vt = svd(A_init, full_matrices=False)
                A_v = U @ Vt
            except Exception:
                A_v = np.random.randn(self.d, self.s) * 0.01
                U, _, Vt = svd(A_v, full_matrices=False)
                A_v = U @ Vt
            self.A_views.append(A_v)

        self.PXI = [self.P_views[v] @ self._XIv_views[v] for v in range(self.V)]


    # ---------------------------------------------- (i) update P^(v)  (Procrustes)
    def _update_P(self):
        for v in range(self.V):
            Z_v = self.G + self.L_views[v]                    # s x n
            M_v = self.QXI_views[v] @ Z_v.T @ self.A_views[v].T   # d_v x d
            M_v = np.nan_to_num(M_v)
            U, _, Vt = np.linalg.svd(M_v.T, full_matrices=False)
            W = U @ Vt                                        # d x d_v (= V U^T)
            # P_v = W Q^{-1/2}, applied implicitly via cached SVD of X_v.
            WUx = W @ self._Ux_views[v]
            self.P_views[v] = (self._inv_sqrt_gamma * W
                               + (WUx * self._coef2_views[v][np.newaxis, :]) @ self._Ux_views[v].T)
            self.PXI[v] = self.P_views[v] @ self._XIv_views[v]

    # ---------------------------------------------- (ii) update A^(v)  (Procrustes)
    def _update_A(self):
        for v in range(self.V):
            Z_v = self.G + self.L_views[v]
            N_v = Z_v @ self._IXv_views[v] @ self.P_views[v].T   # s x d
            U, _, Vt = svd(N_v, full_matrices=False)
            self.A_views[v] = Vt.T @ U.T                          # d x s

    # ---------------------------------------------- (iii) update Y^(v)  (sparse simplex)
    def _update_Y(self):
        for v in range(self.V):
            Z_v = self.G + self.L_views[v]
            self.Y_views[v] = self._sparse_simplex_batch(Z_v, self.kappa)

    def _sparse_simplex_batch(self, Z, kappa):
        """Project every column of Z onto {y>=0, 1^T y=1, ||y||_0 <= kappa}."""
        s, n = Z.shape
        if kappa >= s:
            return self._simplex_batch(Z)
        top_idx = np.argpartition(-Z, kappa, axis=0)[:kappa, :]   # kappa x n
        top_vals = np.take_along_axis(Z, top_idx, axis=0)
        proj = self._simplex_batch(top_vals)                      # kappa x n
        Y = np.zeros_like(Z)
        np.put_along_axis(Y, top_idx, proj, axis=0)
        return Y

    @staticmethod
    def _simplex_batch(A):
        """Vectorized Euclidean projection of each column onto the probability simplex."""
        s, n = A.shape
        sorted_A = np.sort(A, axis=0)[::-1]
        cumsum = np.cumsum(sorted_A, axis=0)
        idx = np.arange(1, s + 1)[:, np.newaxis]
        cond = sorted_A - (cumsum - 1.0) / idx > 0
        rho = np.maximum(cond.sum(axis=0) - 1, 0)
        theta = (cumsum[rho, np.arange(n)] - 1.0) / (rho + 1)
        Y = np.maximum(A - theta[np.newaxis, :], 0)
        col_sums = Y.sum(axis=0)
        nz = col_sums > 0
        Y[:, nz] /= col_sums[np.newaxis, nz]
        Y[:, ~nz] = 1.0 / s
        return Y

    # ---------------------------------------------- (iv) update L^(v)  (closed form)
    def _update_L(self):
        for v in range(self.V):
            AG = self.A_views[v] @ self.G                     # d x n
            b1 = self.A_views[v].T @ (self.PXI[v] - AG)        # s x n
            b1[:, ~self.vis_masks[v]] = 0.0                    # invisible -> no fit term
            c1 = self.Y_views[v] - self.G                      # s x n
            sum_L_other = np.zeros((self.s, self.n))
            for u in range(self.V):
                if u != v:
                    sum_L_other += self.L_views[u]
            denom = (2.0 * self.vis_float[v] + self.theta
                     + 2.0 * self.beta * (self.V - 1))         # n
            numer = 2.0 * b1 + self.theta * c1 + 2.0 * self.beta * sum_L_other
            self.L_views[v] = numer / denom[np.newaxis, :]

    # ---------------------------------------------- (v) update G  (global sparse solve)
    def _update_G(self):
        RHS = np.zeros((self.s, self.n))
        for v in range(self.V):
            RHS += self.theta * (self.Y_views[v] - self.L_views[v])
            AL = self.A_views[v] @ self.L_views[v]
            b2 = self.A_views[v].T @ (self.PXI[v] - AL)
            b2[:, ~self.vis_masks[v]] = 0.0
            RHS += 2.0 * b2
        # Solve M G^T = RHS^T using the prefactored sparse LU.
        self.G = self._M_lu.solve(RHS.T).T

    # ---------------------------------------------- penalized objective
    def _compute_objective(self):
        obj = 0.0
        for v in range(self.V):
            Z_v = self.G + self.L_views[v]
            recon = self.PXI[v] - self.A_views[v] @ Z_v
            recon[:, ~self.vis_masks[v]] = 0.0
            obj += np.sum(recon ** 2)
            obj += self.gamma * np.sum(self.P_views[v] ** 2)
            for u in range(v + 1, self.V):
                obj += self.beta * np.sum((self.L_views[u] - self.L_views[v]) ** 2)
            penalty = self.Y_views[v] - self.G - self.L_views[v]
            obj += (self.theta / 2.0) * np.sum(penalty ** 2)
        if self.use_graph:
            GLS = self.L_graph_sum.dot(self.G.T).T
            obj += self.beta_graph * float(np.sum(GLS * self.G))
        return obj

    # ---------------------------------------------- spectral clustering + NCut
    def _spectral_clustering(self, S):
        d_vec = S.sum(axis=1)
        d_inv_sqrt = 1.0 / (np.sqrt(d_vec) + 1e-10)
        L_norm = S * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
        n = L_norm.shape[0]
        k_use = min(self.k, n)
        try:
            _, eigvecs = eigh(L_norm, subset_by_index=[n - k_use, n - 1])
            U = eigvecs[:, ::-1]
        except TypeError:
            eigvals, eigvecs = eigh(L_norm)
            U = eigvecs[:, np.argsort(eigvals)[::-1][:k_use]]
        U = normalize(U, axis=1, norm="l2")
        km = KMeans(n_clusters=self.k, n_init=self.n_kmeans_runs,
                    random_state=self.random_state if self.random_state is not None else 0,
                    max_iter=300)
        return km.fit_predict(U)

    @staticmethod
    def _normalized_cut(S, labels):
        d = S.sum(axis=1)
        ncut = 0.0
        for k in np.unique(labels):
            mask = (labels == k)
            vol_k = d[mask].sum()
            if vol_k < 1e-10:
                continue
            ncut += S[np.ix_(mask, ~mask)].sum() / vol_k
        return ncut


# ============================================================================
# Data utilities for incomplete multi-view clustering
# ============================================================================
def generate_missing_mask(n, V, mr, seed=0):
    """Boolean (n, V) view-availability mask at missing ratio `mr`.
    Guarantees every sample keeps at least one observed view."""
    rng = np.random.RandomState(seed)
    mask = np.ones((n, V), dtype=bool)
    for v in range(V):
        idx = rng.choice(n, int(n * mr), replace=False)
        mask[idx, v] = False
    for i in range(n):
        if not mask[i].any():
            mask[i, rng.randint(V)] = True
    return mask


def build_views(X_list, mask):
    """From full per-view feature matrices X_list[v] of shape (n, d_v) and a
    (n, V) availability mask, build the model inputs:
      X_views[v] : (d_v, n_v)     features of the observed samples,
      I_views[v] : (n, n_v)       0/1 indicator mapping observed cols to samples.
    """
    n, V = mask.shape
    X_views, I_views = [], []
    for v in range(V):
        avail = np.where(mask[:, v])[0]
        X_v = X_list[v][avail].T.astype(np.float64)   # d_v x n_v
        I_v = np.zeros((n, len(avail)))
        I_v[avail, np.arange(len(avail))] = 1.0
        X_views.append(X_v)
        I_views.append(I_v)
    return X_views, I_views


def load_mat_dataset(path):
    """Load a multi-view .mat dataset. Returns (X_list, y) with X_list[v] of
    shape (n, d_v). Handles common key layouts (X/x, Y/y/gt)."""
    from scipy.io import loadmat
    data = loadmat(path)
    key = "X" if "X" in data else ("x" if "x" in data else None)
    if key is None:
        raise ValueError("Cannot find feature key 'X'/'x' in %s" % path)
    X_cell = data[key]
    X_list = [X_cell[0, v] for v in range(X_cell.shape[1])]
    # Ensure (n, d_v): views are stored as (n, d_v) already in these datasets.
    ylabel = next((k for k in ("Y", "y", "gt") if k in data), None)
    if ylabel is None:
        raise ValueError("Cannot find label key in %s" % path)
    y = np.asarray(data[ylabel]).flatten()
    return X_list, y


def _synthetic_dataset(n=300, V=3, k=4, seed=0):
    """Small synthetic multi-view dataset for a dependency-free smoke test."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, k, size=n)
    dims = [30, 40, 20][:V] + [25] * max(0, V - 3)
    X_list = []
    for v in range(V):
        d_v = dims[v]
        centers = rng.randn(k, d_v) * 3.0
        Xv = centers[y] + rng.randn(n, d_v)
        X_list.append(Xv.astype(np.float64))
    return X_list, y


# ============================================================================
# CLI / demo
# ============================================================================
def run(X_list, y, mr=0.3, seed=42, **model_kwargs):
    """Build an incomplete instance at ratio `mr`, fit GL-AGCL, and report metrics."""
    n = X_list[0].shape[0]
    V = len(X_list)
    k = len(np.unique(y))
    mask = generate_missing_mask(n, V, mr, seed=seed)
    X_views, I_views = build_views(X_list, mask)
    model = GLAGCL(n_clusters=k, random_state=seed, **model_kwargs)
    labels = model.fit(X_views, I_views, y_true=y)
    acc = clustering_accuracy(y, labels) * 100
    nmi = normalized_mutual_info_score(y, labels) * 100
    pur = purity_score(y, labels) * 100
    print("\n=== GL-AGCL result (MR=%.1f) ===" % mr)
    print("ACC=%.2f%%  NMI=%.2f%%  Purity=%.2f%%" % (acc, nmi, pur))
    return model, dict(acc=acc, nmi=nmi, purity=pur)


def main():
    ap = argparse.ArgumentParser(description="GL-AGCL for incomplete multi-view clustering.")
    ap.add_argument("--mat", type=str, default=None,
                    help="Path to a multi-view .mat dataset (keys X/x and Y/y/gt).")
    ap.add_argument("--mr", type=float, default=0.3, help="Missing ratio in [0,1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_anchors", type=int, default=100)
    ap.add_argument("--latent_dim", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beta_graph", type=float, default=0.5)
    ap.add_argument("--k_nn", type=int, default=3)
    ap.add_argument("--kappa", type=int, default=None,
                    help="Cardinality (nonzeros per Y column); default = #clusters.")
    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--no_graph", action="store_true", help="Disable KNN smoothness term.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.mat:
        X_list, y = load_mat_dataset(args.mat)
        print("Loaded %s: %d samples, %d views, %d classes"
              % (args.mat, X_list[0].shape[0], len(X_list), len(np.unique(y))))
    else:
        print("No --mat given; running a synthetic smoke test.")
        X_list, y = _synthetic_dataset()

    run(X_list, y, mr=args.mr, seed=args.seed,
        n_anchors=args.n_anchors, latent_dim=args.latent_dim,
        auto_latent_dim=(args.latent_dim is None),
        gamma=args.gamma, beta=args.beta, beta_graph=args.beta_graph,
        k_nn=args.k_nn, kappa=args.kappa, max_iter=args.max_iter,
        use_graph=(not args.no_graph), verbose=(not args.quiet))


if __name__ == "__main__":
    main()
