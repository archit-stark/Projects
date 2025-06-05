# Parameters
min_win = int((0.01 + 1.8 / np.sqrt(T)) * T)
split_point = int(T * coint_frac)

# Create cointegrated series
mat = np.zeros((T, num_series))
mat = (mat + common_fac) * multipliers + e

# Pre-break OLS
Y = mat[:split_point, 0]
X = mat[:split_point, 1]
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const)
results = model.fit()
a, b = results.params
pre_resid = Y - (a + b * X)

print(f"beta = {b:.4f}")
print(f"ADF on residuals  stat = {adfuller(pre_resid, autolag='AIC')[0]:.3f}")


# Bubble Phase
no_of_random_ts = int(num_of_rw * mat.shape[1])

bubble_matrix = np.zeros((T - split_point, no_of_random_ts))
bubble_matrix[0, :] = mat[split_point - 1, :no_of_random_ts]

for t in range(1, T - split_point):
    bubble_matrix[t, :] = (1 + delta) * bubble_matrix[t - 1, :] + innov[t, :]

mat[split_point:, :no_of_random_ts] = bubble_matrix

Y = mat[split_point:, 0]
X = mat[split_point:, 1]
post_resid = Y - (a + b * X)
new_res = np.concatenate([pre_resid, post_resid])
sadf_stats = mf.sadf_series(new_res, min_win)


plt.figure(figsize=(15, 15));  # Adjusted for 5 subplots

# Subplot 1: Pre-break time series
plt.subplot(5, 1, 1)
plt.plot(mat[:split_point, :], color = "magenta")
plt.plot(common_fac[:split_point], color = 'black', alpha = 1)
plt.title(f'First {coint_frac*100:.0f}% of the series')

# Subplot 2: Pre-break residuals
plt.subplot(5, 1, 2)
plt.plot(pre_resid, color='limegreen')
plt.axhline(0, color='black', lw=0.7)
plt.title('Residuals (pre-break)')

# Subplot 3: Post-break time series

plt.subplot(5, 1, 3)
plt.plot(mat[:, :no_of_random_ts], color="blue", alpha=0.7)
plt.plot(mat[:, no_of_random_ts:], color="red")
plt.title('Time Series after Explosion')

# Subplot 4: Full residuals (pre + post)
plt.subplot(5, 1, 4)
plt.plot(new_res, color='red')
plt.title('Residuals (after explosion)')

# Subplot 5: SADF Path
plt.subplot(5, 1, 5)
plt.plot(sadf_stats, label='SADF path', color='black')
plt.axhline(crit_val_95, color='blue', linestyle='--', label='95% Critical Value')
plt.axhline(crit_val_99, color='red', linestyle='--', label='99% Critical Value')
plt.title("SADF Recursive Path with Bubble Detection")
plt.xlabel("Time")
plt.ylabel("SADF Statistic")
plt.legend()

# Final layout and display
plt.tight_layout()
plt.show()


if max(sadf_stats) > crit_val_99:
    print(f"Time series is Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
else:
    print("Time series is Non-Stationary")
print("")
