import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
np.random.seed(42)

def forward(x):
    y1 = 1 / np.abs(x - 0)
    y2 = 1 / np.abs(x - 1)
    return np.array([y1, y2])

x_true = 1 / np.pi
sigma = 0.2

y_true = forward(x_true)

noise = sigma * np.random.randn(2)
y_obs = y_true + noise

print("Ground truth:", x_true)
print("Observed data:", y_obs)

def posterior(x_grid, y_obs, sigma):
    values = []

    for x in x_grid:
        y_model = forward(x)
        
        misfit = np.sum((y_obs - y_model)**2)

        p = np.exp(-misfit / (2 * sigma**2))
        values.append(p)

    values = np.array(values)

    # normalizacija
    values /= np.trapz(values, x_grid)

    return values

x_grid = np.linspace(0.01, 0.99, 1000)

post = posterior(x_grid, y_obs, sigma)

plt.figure(figsize=(8,4))
plt.plot(x_grid, post, linewidth=2)
plt.axvline(x_true, color='red', linestyle='--', label='Ground truth')

plt.xlabel('Source postirion x')
plt.ylabel('Posterior density')
plt.title('Posteriorna density for source localization')
plt.legend()
plt.grid(True)
plt.show()

x_map = x_grid[np.argmax(post)]

print("MAP estimate:", x_map)
print("Absolute error:", abs(x_map - x_true))

from scipy.optimize import minimize

def objective(x, y_obs, sigma):
    x = float(x)
    y_model = forward(x)
    return np.sum((y_obs - y_model)**2)/(2*sigma**2)

res = minimize(
    lambda z: objective(z[0], y_obs, sigma),
    x0=[0.5],
    bounds=[(0.01,0.99)]
)

print(res.x)

resolutions = [20, 50, 100, 200, 500, 1000]
estimates = []

for n in resolutions:
    grid = np.linspace(0.01,0.99,n)
    post = posterior(grid, y_obs, sigma)
    x_est = grid[np.argmax(post)]
    estimates.append(x_est)

plt.plot(resolutions, estimates)
plt.axhline(x_true, color='r', linestyle='--')
plt.xlabel('Grid resolutions')
plt.ylabel('MAP estimator')
plt.title('Convergency of MAP estimator')
plt.show()

N = 200
errors = []

for i in range(N):

    noise = sigma * np.random.randn(2)

    y_obs = forward(x_true) + noise

    post = posterior(x_grid, y_obs, sigma)

    x_est = x_grid[np.argmax(post)]

    err = abs(x_est - x_true)

    errors.append(err)

errors = np.array(errors)

print("Srednja apsolutna greška:", np.mean(errors))
print("std:", np.std(errors))
print("maksimalna greška:", np.max(errors))

plt.figure(figsize=(8,4))
plt.hist(errors, bins=20)

plt.xlabel('Absolute error')
plt.ylabel('frequency')
plt.title('Empirical reconstruction error')
plt.grid(True)
plt.show()

def posterior_single(x_grid, y_obs, sigma):
    values = []

    for x in x_grid:
        y_model = 1 / np.abs(x - 1)

        misfit = (y_obs - y_model)**2

        p = np.exp(-misfit/(2*sigma**2))
        values.append(p)

    values = np.array(values)
    values /= np.trapz(values, x_grid)

    return values


y_single = y_true[1] + sigma*np.random.randn()

post_single = posterior_single(
    x_grid,
    y_single,
    sigma
)

plt.figure(figsize=(8,4))

plt.plot(x_grid, post, label='Two measurements')
plt.plot(x_grid, post_single, label='One measurement')

plt.axvline(
    x_true,
    color='red',
    linestyle='--',
    label='Ground truth'
)

plt.xlabel('Source position x')
plt.ylabel('Posterior density')
plt.title('Effect of measurement quantity')
plt.legend()
plt.grid(True)
plt.show()
