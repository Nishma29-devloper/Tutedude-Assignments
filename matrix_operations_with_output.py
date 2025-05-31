
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------
# 1. Matrix and Vector Operations
# -------------------------------
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([1, 2, 3])

AxB = np.dot(A, B)
trace_A = np.trace(A)
eigvals_A, eigvecs_A = np.linalg.eig(A)

print("1. Matrix and Vector Operations")
print("A * B =", AxB)
print("Trace of A =", trace_A)
print("Eigenvalues of A =", eigvals_A)
print("Eigenvectors of A =\n", eigvecs_A)

# -------------------------------
# 2. Invertibility of Matrices
# -------------------------------
A_updated = np.array([[2, 0, 1],
                      [1, 3, 2],
                      [0, 1, 1]])
B = np.array([1, 2, 3])

det_A_updated = np.linalg.det(A_updated)
if det_A_updated != 0:
    inv_A_updated = np.linalg.inv(A_updated)
    X = np.linalg.solve(A_updated, B)
else:
    inv_A_updated = None
    X = None

print("\n2. Invertibility of Matrices")
print("Determinant of A:", det_A_updated)
print("Inverse of A:\n", inv_A_updated)
print("Solution X for AX = B:", X)

# -------------------------------
# 3. Practical Matrix Operations
# -------------------------------
np.random.seed(0)
C = np.random.randint(1, 21, size=(4, 4))

rank_C = np.linalg.matrix_rank(C)
submatrix_C = C[:2, -2:]
frobenius_norm_C = np.linalg.norm(C, 'fro')

C_3x3 = C[:3, :3]
if A_updated.shape[1] == C_3x3.shape[0]:
    product_AC = np.dot(A_updated, C_3x3)
else:
    product_AC = None

print("\n3. Practical Matrix Operations")
print("Matrix C:\n", C)
print("Rank of C:", rank_C)
print("Submatrix of C:\n", submatrix_C)
print("Frobenius norm of C:", round(frobenius_norm_C, 2))
print("C 3x3:\n", C_3x3)
print("A * C_3x3:\n", product_AC)

# -------------------------------
# 4. Data Science Context
# -------------------------------
D = np.array([[3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10],
              [1, 3, 5, 7, 9],
              [4, 6, 8, 10, 12],
              [5, 7, 9, 11, 13]])

scaler = StandardScaler()
D_standardized = scaler.fit_transform(D)

cov_matrix = np.cov(D_standardized, rowvar=False)
eigvals_D, eigvecs_D = np.linalg.eig(cov_matrix)

pca = PCA(n_components=2)
D_pca = pca.fit_transform(D_standardized)

print("\n4. Data Science Context")
print("Standardized D:\n", D_standardized)
print("Covariance matrix:\n", cov_matrix)
print("Eigenvalues:\n", eigvals_D)
print("Eigenvectors:\n", eigvecs_D)
print("D reduced to 2 principal components:\n", D_pca)
