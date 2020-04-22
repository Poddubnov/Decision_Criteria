import numpy as np

x = np.array([[28, 41, 72, 120], [29, 38, 61, 105], [31, 40, 69, 117],
              [30, 40, 55, 115], [24, 31, 48, 95], [26, 35, 63, 110], [32, 40, 65, 115]])  # Source data
q = 0.25  # probability of appearance of an external state
c = 0.5  # weight factor (for Hurwitz)
v = 0.5  # also weight factor (for Hodge-Lehmann)
epsilon = 5


def MinMaxCriterion():
    y = np.array(x.min(axis=1)).reshape(7, 1)
    x_solution = np.append(x, y, axis=1)
    criterion = x_solution[0:, 4].max()
    print("Минимаксный критерий: вид матрицы\n", x_solution)
    print("Ответ:\n", criterion, "\n")


def SavageCriterion():
    y = np.array(x.max(axis=0))
    z = y - x
    z_solveColumn = np.array(z.max(axis=1)).reshape(7, 1)
    z_solution = np.append(z, z_solveColumn, axis=1)
    criterion = z_solution[0:, 4].max()
    print("Критерий Сэвиджа: вид матрицы\n", z_solution)
    print("Ответ:\n", criterion, "\n")


def BayesLaplaceCriterion():
    y = np.array(x * q)
    z = y.sum(axis=1).reshape(7, 1)
    z_solution = np.append(x, z, axis=1)
    criterion = z_solution[0:, 4].max()
    print("Критерий Байеса-Лапласа: вид матрицы\n", z_solution)
    print("Ответ:\n", criterion, "\n")


def HermeyerCriterion():
    y = np.array(x * q)
    y_solveColumn = np.array(y.min(axis=1)).reshape(7, 1)
    y_solution = np.append(y, y_solveColumn, axis=1)
    criterion = y_solution[0:, 4].max()
    print("Критерий Гермейера: вид матрицы\n", y_solution)
    print("Ответ:\n", criterion, "\n")


def HurwitzCriterion():
    y_cMin = np.array(x.min(axis=1) * c).reshape(7, 1)
    y_1_cMax = np.array(x.max(axis=1) * (1 - c)).reshape(7, 1)
    z = y_cMin + y_1_cMax
    z_solution = np.append(x, z, axis=1)
    criterion = z_solution[0:, 4].max()
    print("Критерий Гурвица: вид матрицы\n", z_solution)
    print("Ответ:\n", criterion, "\n")


def HodgeLehmannCriterion():
    y = np.array(x * q)
    Eij_Qj = y.sum(axis=1).reshape(7, 1)
    v_Eij_Qj = Eij_Qj * v
    inverse_v_Eij_Qj = (1 - v) * x.min(axis=1).reshape(7, 1)
    z = v_Eij_Qj + inverse_v_Eij_Qj
    z_solution = np.append(x, z, axis=1)
    criterion = z_solution[0:, 4].max()
    print("Критерий Ходжа-Лемана: вид матрицы\n", z_solution)
    print("Ответ:\n", criterion, "\n")


def CompositionCriterion():
    y = np.array(x.prod(axis=1)).reshape(7, 1)
    y_solution = np.append(x, y, axis=1)
    criterion = y_solution[0:, 4].max()
    print("Критерий произведения: вид матрицы\n", y_solution)
    print("Ответ:\n", criterion, "\n")


def MinMaxBayesLaplaceCriterion():
    y = np.array(x.min(axis=1))
    minMaxCriterion = y.max()
    iMin = (minMaxCriterion - y)
    i1 = np.empty((iMin.shape[0], 0), dtype='int32')
    for i in range(iMin.shape[0]):
        if iMin[i] <= epsilon:
            i1 = np.append(i1, iMin[i])

    b = x[np.where(x[:, 0] == minMaxCriterion)]
    iMax = (b.max(axis=1) - x.max(axis=1)).ravel()
    i2 = np.empty((iMin.shape[0], 0), dtype='int32')
    for j in range(iMin.shape[0]):
        if iMax[j] >= iMin[j]:
            i2 = np.append(i2, iMin[j])

    I1_I2 = list(set(i1) & set(i2))
    x2 = x[np.where(iMin == I1_I2[0])].ravel()
    x1 = x[np.where(iMin == I1_I2[1])].ravel()
    x_mod = np.stack((x1, x2), axis=0)
    y_mod = np.array(x_mod * q)
    z = y_mod.sum(axis=1).reshape(2, 1)
    z_solution = np.append(x_mod, z, axis=1)
    criterion = z_solution[0:, 4].max()
    print("Множество I1\n", i1)
    print("Множество I2\n", i2)
    print("Объединенное множество\n", I1_I2)
    print("Критерий минимаксный Байеса-Лапласа: вид матрицы\n", z_solution)
    print("Ответ:\n", criterion, "\n")


print("Исходная матрица\n", x, "\n")
MinMaxCriterion()
SavageCriterion()
BayesLaplaceCriterion()
HermeyerCriterion()
HurwitzCriterion()
HodgeLehmannCriterion()
CompositionCriterion()
MinMaxBayesLaplaceCriterion()
