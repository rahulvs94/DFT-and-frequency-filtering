# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import math
import numpy as np

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        # Using direct function
        # matrix1 = np.fft.fft2(matrix)

        # Without using direct function
        rows, columns = np.shape(matrix)
        matrix1 = np.zeros((rows, columns), dtype=complex)
        for u in range(rows):
            for v in range(columns):
                a = []
                for i in range(rows):
                    for j in range(columns):
                        omega = np.exp(-2 * math.pi * 1J * (((u * i) / rows) + ((v * j) / columns)))
                        a.append(matrix[i, j] * omega)
                matrix1[u, v] = sum(a)

        return matrix1

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        # Using direct function
        # matrix1 = np.fft.ifft2(matrix)

        # Without using direct function
        rows, columns = np.shape(matrix)
        matrix1 = np.zeros((rows, columns), dtype=complex)
        for u in range(rows):
            for v in range(columns):
                a = []
                for i in range(rows):
                    for j in range(columns):
                        omega = np.exp(2 * math.pi * 1J * (((u * i) / rows) + ((v * j) / columns)))
                        a.append(matrix[i, j] * omega)
                matrix1[u, v] = sum(a)
                # Normalized value of inverse dft
                # matrix1[u, v] = (1 / (rows * columns)) * sum(a)

        return matrix1


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        rows, columns = np.shape(matrix)
        matrix1 = np.zeros((rows, columns), dtype=float)
        for u in range(rows):
            for v in range(columns):
                a = []
                for i in range(rows):
                    for j in range(columns):
                        temp = math.cos(2 * math.pi * (((u * i) / rows) + ((v * j) / columns)))
                        a.append(matrix[i, j] * temp)
                matrix1[u, v] = sum(a)

        return matrix1


    def discrete_sine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        rows, columns = np.shape(matrix)
        matrix1 = np.zeros((rows, columns), dtype=float)
        for u in range(rows):
            for v in range(columns):
                a = []
                for i in range(rows):
                    for j in range(columns):
                        temp = math.sin(2 * math.pi * (((u * i) / rows) + ((v * j) / columns)))
                        a.append(matrix[i, j] * temp)
                matrix1[u, v] = sum(a)

        return matrix1


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        # method 1
        # matrix1 = np.abs(matrix)

        # method 2
        rows, columns = np.shape(matrix)
        matrix1 = np.zeros((rows, columns), dtype=float)
        for i in range(rows):
            for j in range(columns):
                # matrix1[i, j] = abs(matrix[i, j])
                matrix1[i, j] = math.sqrt(matrix[i, j].real ** 2 + matrix[i, j].imag ** 2)

        return matrix1
