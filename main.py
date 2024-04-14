import numpy
import numpy as np
import pandas as pd
from fractions import Fraction

# Load the CSV file into a DataFrame
file = ["CRFM.csv", "ECT.csv", "ELW.csv", "RIS.csv", "SIRM.csv"]

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


def ahp(ar: numpy.ndarray,filename:str):
    print(f"filename: {filename}")

    # # Define the matrix as a numpy array
    matrix = np.array(ar)

    # Sum the columns
    column_sums = matrix.sum(axis=0)

    # Normalize the matrix by dividing each element by its column sum
    normalized_matrix = matrix / column_sums

    # Calculate the priority vector by taking the average of each row in the normalized matrix
    priority_vector = normalized_matrix.mean(axis=1)

    # Display results
    # print("Normalized Matrix:\n", normalized_matrix)
    # print("\nPriority Vector (Weights):\n", priority_vector)
    print("------------------------------------------------------------")

    # Calculate the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the maximum eigenvalue
    max_eigenvalue = np.max(np.real(eigenvalues))
    print(f"Max Eigenvalue:{max_eigenvalue}\n")

    # Calculate the size of the matrix (n)
    n = matrix.shape[0]
    print(f"n:{n}")

    # Define the Random Index (ri) for matrices up to order 10
    ri_values = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

    # Calculate the Consistency Index (ci)
    ci = (max_eigenvalue - n) / (n - 1)

    # Determine the Random Index (ri) based on the size of the matrix
    ri = ri_values[n - 1] if n - 1 < len(ri_values) else 1.49
    # default to the last known ri value if n is out of range

    # Calculate the Consistency Ratio (cr)
    cr = ci / ri


    # Output the results
    print(f"Maximum Eigenvalue: {max_eigenvalue}")
    print(f"Consistency Index (CI): {ci}")
    print(f"Random Index (RI): {ri}")
    print(f"Consistency Ratio (CR): {cr}")

    # Checking the consistency
    if cr < 0.1:
        print("The matrix is consistent.")
    else:
        print("The matrix is not consistent. Review comparisons.")

    print("---------------------------------------------------------------------")


# for f in file:
#     df = pd.read_csv(f, index_col=0)
#     p_numpy = df.to_numpy()
#     arr = StrToArray(p_numpy)
#     arr.float_array()
#     arr1 = arr.get_array()
#     ahp(arr1,f)

ok