'''
         b1       b2     ...      bn
a1       A11      A12             A1n
a2       A21      A22             A2n
...
an       An1      An2            Ann



A11(a1) + A21(a2) + ... + An1(an) >= z
A12(a1) + A22(a2) + ... + An2(an) >= z
...
A1n(a1) + A2n(a2) + ... + Ann(an) >= z
a1 + ... + an = 1
a1, a2, ..., an >= 0



11(a1) + 12(a2) + 13(an) >= z
21(a1) + 22(a2) + 23(an) >= z
32(a1) + 32(a2) + 33(an) >= z
a1 + a2 + a3 = 1
a1, a2, a3 >= 0
maximize z

'''

import numpy as np

'''
این بخش از کد از درس ترم پیش دکتر ضرابی  زاده برداشته شده است
دلیل استفاده از این کد نیز ساپورت نشدن کتابخانه
scipy
 در تست های کوئرا است.
'''
import numpy as np
from math import inf


class LPSolver(object):
    EPS = 1e-9
    NEG_INF = -inf

    def __init__(self, A, b, c):
        self.m = len(b)
        self.n = len(c)
        self.N = [0] * (self.n + 1)
        self.B = [0] * self.m
        self.D = [[0 for i in range(self.n + 2)] for j in range(self.m + 2)]
        self.D = np.array(self.D, dtype=np.float64)
        for i in range(self.m):
            for j in range(self.n):
                self.D[i][j] = A[i][j]
        for i in range(self.m):
            self.B[i] = self.n + i
            self.D[i][self.n] = -1
            self.D[i][self.n + 1] = b[i]
        for j in range(self.n):
            self.N[j] = j
            self.D[self.m][j] = -c[j]
        self.N[self.n] = -1
        self.D[self.m + 1][self.n] = 1

    def Pivot(self, r, s):
        D = self.D
        B = self.B
        N = self.N
        inv = 1.0 / D[r][s]
        dec_mat = np.matmul(D[:, s:s + 1], D[r:r + 1, :]) * inv
        dec_mat[r, :] = 0
        dec_mat[:, s] = 0
        self.D -= dec_mat
        self.D[r, :s] *= inv
        self.D[r, s + 1:] *= inv
        self.D[:r, s] *= -inv
        self.D[r + 1:, s] *= -inv
        self.D[r][s] = inv
        B[r], N[s] = N[s], B[r]

    def Simplex(self, phase):
        m = self.m
        n = self.n
        D = self.D
        B = self.B
        N = self.N
        x = m + 1 if phase == 1 else m
        while True:
            s = -1
            for j in range(n + 1):
                if phase == 2 and N[j] == -1:
                    continue
                if s == -1 or D[x][j] < D[x][s] or D[x][j] == D[x][s] and N[j] < N[s]:
                    s = j
            if D[x][s] > -self.EPS:
                return True
            r = -1
            for i in range(m):
                if D[i][s] < self.EPS:
                    continue
                if r == -1 or D[i][n + 1] / D[i][s] < D[r][n + 1] / D[r][s] or (D[i][n + 1] / D[i][s]) == (
                        D[r][n + 1] / D[r][s]) and B[i] < B[r]:
                    r = i
            if r == -1:
                return False
            self.Pivot(r, s)

    def Solve(self):
        m = self.m
        n = self.n
        D = self.D
        B = self.B
        N = self.N
        r = 0
        for i in range(1, m):
            if D[i][n + 1] < D[r][n + 1]:
                r = i
        if D[r][n + 1] < -self.EPS:
            self.Pivot(r, n)
            if not self.Simplex(1) or D[m + 1][n + 1] < -self.EPS:
                return self.NEG_INF, None
            for i in range(m):
                if B[i] == -1:
                    s = -1
                    for j in range(n + 1):
                        if s == -1 or D[i][j] < D[i][s] or D[i][j] == D[i][s] and N[j] < N[s]:
                            s = j
                    self.Pivot(i, s)
        if not self.Simplex(2):
            return self.NEG_INF, None
        x = [0] * self.n
        for i in range(m):
            if B[i] < n:
                x[B[i]] = D[i][n + 1]
        return D[m][n + 1], x


'''
از اینجا کد مربوط به خودم آغاز میشود
99105691
محمدجواد ماهرالنقش
'''
if __name__ == '__main__':
    n = int(input())
    array = np.array([])
    for i in range(n):
        new_array = np.array(list(map(int, input().split())))
        array = np.append(array, new_array, axis=0)
    array = array.reshape((n, n))

    A = np.copy(array)
    column = np.array([-1 for i in range(n)])
    A = np.append(A, column)
    A = np.transpose(A)
    A = A.reshape(n + 1, n)
    A = np.transpose(A)

    plus_ones = [[1 for i in range(n)]]
    plus_ones[0].append(0)
    plus_ones = np.array(plus_ones)
    A = np.append(A, plus_ones, axis=0)
    minus_ones = [[-1 for i in range(n)]]
    minus_ones[0].append(0)
    A = np.append(A, minus_ones, axis=0)

    A *= -1
    c = [0 for i in range(n)]
    c.append(1)
    b = [0 for i in range(n)]
    b.append(-1)
    b.append(1)

    A = A.reshape(n + 2, n + 1)

    s = LPSolver(A, b, c)
    answer = s.Solve()[1]
    for i in range(n):
        print(round(answer[i], 7), end=' ')
    print()

    array = np.transpose(array)
    array = 100 - array
    A = np.copy(array)
    column = np.array([-1 for i in range(n)])
    A = np.append(A, column)
    A = np.transpose(A)
    A = A.reshape(n + 1, n)
    A = np.transpose(A)

    plus_ones = [[1 for i in range(n)]]
    plus_ones[0].append(0)
    plus_ones = np.array(plus_ones)
    A = np.append(A, plus_ones, axis=0)
    minus_ones = [[-1 for i in range(n)]]
    minus_ones[0].append(0)
    A = np.append(A, minus_ones, axis=0)

    A *= -1
    c = [0 for i in range(n)]
    c.append(1)
    b = [0 for i in range(n)]
    b.append(-1)
    b.append(1)

    A = A.reshape(n + 2, n + 1)

    s = LPSolver(A, b, c)
    answer = s.Solve()[1]
    for i in range(n):
        print(round(answer[i], 7), end=' ')
