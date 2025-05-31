{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bcfc1d6c-caa2-4f0c-91f1-fef6b113c22e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Matrix-Vector Product (A x B):\n",
      " [14 32 50]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Define matrix A and vector B\n",
    "A = np.array([[1, 2, 3], \n",
    "              [4, 5, 6], \n",
    "              [7, 8, 9]])\n",
    "B = np.array([1, 2, 3])\n",
    "\n",
    "# 1. Matrix-vector multiplication\n",
    "product = A @ B  # or np.dot(A, B)\n",
    "print(\"Matrix-Vector Product (A x B):\\n\", product)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "074dd4a4-32a7-438c-9421-c26b9775f51d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Trace of Matrix A: 15\n"
     ]
    }
   ],
   "source": [
    "# 2. Trace of matrix A\n",
    "trace_A = np.trace(A)\n",
    "print(\"Trace of Matrix A:\", trace_A)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e9155423-0b5c-4837-8549-50ff0ff0dd7e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Eigenvalues of A:\n",
      " [ 1.61168440e+01 -1.11684397e+00 -3.38433605e-16]\n",
      "Eigenvectors of A:\n",
      " [[-0.23197069 -0.78583024  0.40824829]\n",
      " [-0.52532209 -0.08675134 -0.81649658]\n",
      " [-0.8186735   0.61232756  0.40824829]]\n"
     ]
    }
   ],
   "source": [
    "# 3. Eigenvalues and eigenvectors of A\n",
    "eigenvalues, eigenvectors = np.linalg.eig(A)\n",
    "print(\"Eigenvalues of A:\\n\", eigenvalues)\n",
    "print(\"Eigenvectors of A:\\n\", eigenvectors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "546dc711-c784-4727-a758-0d5ba8342871",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Determinant of A: -9.51619735392994e-16\n",
      "Inverse of A:\n",
      " [[ 3.15251974e+15 -6.30503948e+15  3.15251974e+15]\n",
      " [-6.30503948e+15  1.26100790e+16 -6.30503948e+15]\n",
      " [ 3.15251974e+15 -6.30503948e+15  3.15251974e+15]]\n",
      "Solution X of the system A * X = B:\n",
      " [-0.23333333  0.46666667  0.1       ]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# --- Step 1: Define your updated matrix A here ---\n",
    "A = np.array([[1, 2, 3],     # <-- Replace with your actual updated matrix\n",
    "              [4, 5, 6],\n",
    "              [7, 8, 9]])\n",
    "\n",
    "# --- Step 2: Define vector B ---\n",
    "B = np.array([1, 2, 3])  # Given in the question\n",
    "\n",
    "# --- Step 3: Check invertibility ---\n",
    "det_A = np.linalg.det(A)\n",
    "print(\"Determinant of A:\", det_A)\n",
    "if det_A != 0:\n",
    "    # A is invertible\n",
    "    inverse_A = np.linalg.inv(A)\n",
    "    print(\"Inverse of A:\\n\", inverse_A)\n",
    "    \n",
    "    # Solve A * X = B\n",
    "    X = np.linalg.solve(A, B)\n",
    "    print(\"Solution X of the system A * X = B:\\n\", X)\n",
    "else:\n",
    "    print(\"Matrix A is not invertible. Determinant is zero.\")\n",
    "    inverse_A = None\n",
    "    X = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "67d202f8-6692-4612-915b-d7ca683e05b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[13, 16,  1,  4],\n",
       "        [ 4,  8, 10, 20],\n",
       "        [19,  5,  7, 13],\n",
       "        [ 2,  7,  8, 15]]),\n",
       " 4,\n",
       " array([[ 1,  4],\n",
       "        [10, 20]]),\n",
       " 44.36214602563767,\n",
       " array([[13, 16,  1],\n",
       "        [ 4,  8, 10],\n",
       "        [19,  5,  7]]),\n",
       " array([[45, 37,  9],\n",
       "        [63, 50, 45],\n",
       "        [23, 13, 17]]))"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Step 1: Create a 4x4 matrix C with random integers between 1 and 20\n",
    "np.random.seed(0)  # For reproducibility\n",
    "C = np.random.randint(1, 21, size=(4, 4))\n",
    "\n",
    "# Task 1: Matrix operations on C\n",
    "rank_C = np.linalg.matrix_rank(C)\n",
    "submatrix_C = C[:2, -2:]  # First 2 rows, last 2 columns\n",
    "frobenius_norm_C = np.linalg.norm(C, 'fro')\n",
    "\n",
    "# Step 2: Define updated 3x3 matrix A (let's use a valid one this time)\n",
    "A = np.array([[2, 0, 1],\n",
    "              [1, 3, 2],\n",
    "              [0, 1, 1]])\n",
    "\n",
    "# Try using top-left 3x3 block of C for multiplication\n",
    "C_3x3 = C[:3, :3]\n",
    "\n",
    "# Check if multiplication is valid: A (3x3) * C_3x3 (3x3) => Result is 3x3\n",
    "if A.shape[1] == C_3x3.shape[0]:\n",
    "    product_AC = np.dot(A, C_3x3)\n",
    "else:\n",
    "    product_AC = None\n",
    "\n",
    "C, rank_C, submatrix_C, frobenius_norm_C, C_3x3, product_AC\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "65529550-77da-43c1-a628-fa9d93bd8bac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],\n",
       "        [-0.70710678, -0.70710678, -0.70710678, -0.70710678, -0.70710678],\n",
       "        [-1.41421356, -1.41421356, -1.41421356, -1.41421356, -1.41421356],\n",
       "        [ 0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678],\n",
       "        [ 1.41421356,  1.41421356,  1.41421356,  1.41421356,  1.41421356]]),\n",
       " array([[1.25, 1.25, 1.25, 1.25, 1.25],\n",
       "        [1.25, 1.25, 1.25, 1.25, 1.25],\n",
       "        [1.25, 1.25, 1.25, 1.25, 1.25],\n",
       "        [1.25, 1.25, 1.25, 1.25, 1.25],\n",
       "        [1.25, 1.25, 1.25, 1.25, 1.25]]),\n",
       " array([ 6.25000000e+00, -3.69778549e-32, -4.19340966e-17,  0.00000000e+00,\n",
       "        -2.28523622e-64]),\n",
       " array([[ 4.47213595e-01, -2.01000734e-16, -3.25055279e-01,\n",
       "         -4.98525175e-49,  8.98516042e-34],\n",
       "        [ 4.47213595e-01,  8.66025404e-01,  8.88074128e-01,\n",
       "          5.96701066e-33, -1.29791556e-17],\n",
       "        [ 4.47213595e-01, -2.88675135e-01, -1.87672950e-01,\n",
       "         -5.16969603e-16,  8.16496581e-01],\n",
       "        [ 4.47213595e-01, -2.88675135e-01, -1.87672950e-01,\n",
       "         -7.07106781e-01, -4.08248290e-01],\n",
       "        [ 4.47213595e-01, -2.88675135e-01, -1.87672950e-01,\n",
       "          7.07106781e-01, -4.08248290e-01]]),\n",
       " array([[ 1.11022302e-15, -3.24923313e-16],\n",
       "        [-1.58113883e+00, -2.94443836e-32],\n",
       "        [-3.16227766e+00, -2.28150329e-32],\n",
       "        [ 1.58113883e+00,  1.14075164e-32],\n",
       "        [ 3.16227766e+00,  2.28150329e-32]]))"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Step 1: Define the dataset D\n",
    "D = np.array([[3, 5, 7, 9, 11],\n",
    "              [2, 4, 6, 8, 10],\n",
    "              [1, 3, 5, 7, 9],\n",
    "              [4, 6, 8, 10, 12],\n",
    "              [5, 7, 9, 11, 13]])\n",
    "\n",
    "# Task 1: Standardize D column-wise\n",
    "scaler = StandardScaler()\n",
    "D_standardized = scaler.fit_transform(D)\n",
    "\n",
    "# Task 2: Compute covariance matrix of standardized D\n",
    "cov_matrix = np.cov(D_standardized, rowvar=False)\n",
    "\n",
    "# Task 3: PCA\n",
    "# - Find eigenvalues and eigenvectors of the covariance matrix\n",
    "eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)\n",
    "\n",
    "# - Reduce D to 2 principal components\n",
    "pca = PCA(n_components=2)\n",
    "D_pca = pca.fit_transform(D_standardized)\n",
    "\n",
    "D_standardized, cov_matrix, eigenvalues, eigenvectors, D_pca\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5020b744-e935-4a5f-8e2d-95f0c63630a2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c3e2d4e-991e-4756-b18e-8f89c1f3cc0a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3fd74c03-ffe8-4344-b5c1-0c07c7be484c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a719661c-7437-4ff7-9759-df8dee2d89a7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
