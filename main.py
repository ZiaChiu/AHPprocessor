import os

import numpy
import numpy as np
import pandas as pd
from fractions import Fraction
import os


def get_csv(directory="."):
    csv_files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            if ".csv" in filename:
                csv_files.append(filename)
    return csv_files


class StrToArray:

    def __init__(self, ar: numpy.ndarray):
        self.ar = ar
        self.ary = []
        self.ty_list = []

    def fraction_array(self):
        for i in self.ar:
            lst = []
            ty_list_1 = []
            for j in i:

                if j is str:

                    j = j.replace(" ", "")
                    if "/" in j:
                        num = [i for i in j.split("/")]
                        j = Fraction(int(num[0]), int(num[1]))
                        # j = float(int(num[0]) / int(num[1]))
                    else:
                        j = Fraction(int(j), 1)

                elif j is int:

                    j = Fraction(int(j), 1)

                ty_list_1.append(type(j))

                lst.append(j)
            self.ary.append(lst)
            self.ty_list.append(ty_list_1)
        self.ary = numpy.array(self.ary)

    def float_array(self):
        for i in self.ar:
            lst = []
            ty_list_1 = []
            for j in i:

                if isinstance(j, str):

                    j = j.replace(" ", "")
                    if "/" in j:
                        num = [i for i in j.split("/")]
                        # j = Fraction(int(num[0]), int(num[1]))
                        j = float(int(num[0]) / int(num[1]))

                    else:
                        j = int(j)

                elif isinstance(j, int):

                    j = int(j)

                elif j is not str:
                    print()

                ty_list_1.append(type(j))

                lst.append(j)
            self.ary.append(lst)
            self.ty_list.append(ty_list_1)

    def get_array(self):
        return numpy.array(self.ary)

    def get_data(self):
        return self.ar

    def get_data_type_list(self):
        return self.ty_list

class AHP:
    def __init__(self,filename,array:numpy.ndarray):
        self.priority_vector = None
        self.filename = filename
        self.arr = array
        self.normalized_matrix = None
        self.max_eigenvalue =None

    def get_normalized_matrix(self):
        return self.normalized_matrix
    def get_max_eigenvalue(self):
        return self.max_eigenvalue

    def get_priority_vector(self):
        return self.priority_vector


    def caculator(self):
        print(f"filename: {self.filename}")

        # # Define the matrix as a numpy array
        matrix = np.array(self.arr)

        # Sum the columns
        column_sums = matrix.sum(axis=0)

        # Normalize the matrix by dividing each element by its column sum
        self.normalized_matrix = matrix / column_sums

        # Calculate the priority vector by taking the average of each row in the normalized matrix
        self.priority_vector = self.normalized_matrix.mean(axis=1)

        # Display results
        print("Normalized Matrix:\n", self.normalized_matrix)
        print("\nPriority Vector (Weights):\n", self.priority_vector)
        print("------------------------------------------------------------")

        # Calculate the eigenvalues and eigenvectors of the matrix
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        # Find the maximum eigenvalue
        self.max_eigenvalue = np.max(np.real(eigenvalues))

        # Calculate the size of the matrix (n)
        n = matrix.shape[0]

        # Define the Random Index (ri) for matrices up to order 10
        ri_values = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

        # Calculate the Consistency Index (ci)
        self.ci = (self.max_eigenvalue - n) / (n - 1)

        # Determine the Random Index (ri) based on the size of the matrix
        self.ri = ri_values[n - 1] if n - 1 < len(ri_values) else 1.49
        # default to the last known ri value if n is out of range

        # Calculate the Consistency Ratio (cr)
        self.cr = self.ci / self.ri

        # Output the results
        print(f"Maximum Eigenvalue: {self.max_eigenvalue}")
        print(f"Consistency Index (CI): {self.ci}")
        print(f"Random Index (RI): {self.ri}")
        print(f"Consistency Ratio (CR): {self.cr}")

        # Checking the consistency
        if self.cr < 0.1:
            print("The matrix is consistent.")
        else:
            print("The matrix is not consistent. Review comparisons.")

        print("---------------------------------------------------------------------")



# Load the CSV file into a DataFrame
files = get_csv()

for f in files:
    df = pd.read_csv(f, index_col=0)
    p_numpy = df.to_numpy()
    arr = StrToArray(p_numpy)
    arr.float_array()
    arr1 = arr.get_array()
    AHP(f,arr1).caculator()