import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


# Функция для 2D ДКП
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


# 1. Визуализация 64 базисных функций ДКП (8x8)
basis_img = np.zeros((8 * 8, 8 * 8))
for i in range(8):
    for j in range(8):
        basis = np.zeros((8, 8))
        basis[i, j] = 1
        basis_spatial = idct2(basis)
        basis_img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = basis_spatial

plt.figure(figsize=(6, 6))
plt.imshow(basis_img, cmap='gray')
plt.title("64 базисные функции ДКП-II")
plt.axis('off')
plt.tight_layout()
plt.savefig('dct_basis.png')

# 2. Создание тестовых изображений (N=256 для наглядности статистики)
N = 256
x = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, x)

# Гладкая функция (градиент)
img_smooth = np.sin(np.pi * X) * np.cos(np.pi * Y)

# Резкие границы (шахматная доска - ловушка)
img_chess = np.zeros((N, N))
img_chess[X % 0.2 > 0.1] = 1
img_chess[Y % 0.2 > 0.1] = np.logical_not(img_chess[Y % 0.2 > 0.1])


# 3. Численный эксперимент: расчет энергии отброшенных гармоник
def calc_discarded_energy(img):
    coeffs = dct2(img)
    # Сортируем коэффициенты по убыванию модуля (или берем зигзаг, но для энергии можно просто 1D массив)
    F_flat = np.abs(coeffs.flatten())
    # Сортируем от самых важных к наименее важным
    F_sorted = np.sort(F_flat)[::-1]

    total_energy = np.sum(F_sorted ** 2)
    energies = []

    # Считаем E_discarded(K)
    for K in range(1, len(F_sorted), 100):
        discarded = np.sum(F_sorted[K:] ** 2) / total_energy
        energies.append(discarded)
    return energies


K_vals = np.arange(1, N * N, 100)
E_smooth = calc_discarded_energy(img_smooth)
E_chess = calc_discarded_energy(img_chess)

# Теоретическая кривая O(1/K^3)
# Подгоняем константу под масштаб для наглядности
theoretical_O3 = 1 / (K_vals ** 3 + 1e-5)
theoretical_O3 = theoretical_O3 / theoretical_O3[1] * E_smooth[1]

plt.figure(figsize=(8, 5))
plt.loglog(K_vals, E_smooth, label="Гладкое изображение (Градиент)", linewidth=2)
plt.loglog(K_vals, E_chess, label="Ловушка: Резкие границы (Шахматы)", linestyle='--')
plt.loglog(K_vals, theoretical_O3, label="Теоретическая оценка O(1/K^3)", linestyle=':', color='red')
plt.xlabel("Номер коэффициента K")
plt.ylabel("Доля отброшенной энергии E(K)")
plt.title("Энергия отброшенных высоких гармоник ДКП")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('energy_decay.png')
plt.show()
