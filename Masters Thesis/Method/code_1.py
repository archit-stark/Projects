""" Inputs """
B = 1000
T = 500
num_series = 2
coint_frac = 0.4
num_of_rw = 0.6
multipliers = mf.unique_rounded_uniform(-2, 2, num_series, decimals=1)

""" Function """
min_win = int((0.01 + 1.8 / np.sqrt(T)) * T)
sadf_stats = []
split_point = int(T * coint_frac)

# establishing cointegration
e = np.random.normal(0, 1, size=(T, num_series))
common_fac = mf.random_walk(T).reshape(-1,1)
mat = np.zeros((T,num_series))
mat = (mat + common_fac)*multipliers + e

# Pre-break OLS
Y = mat[:split_point, 0]
X = mat[:split_point, 1]
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const)
results = model.fit()
a, b = results.params
pre_resid = Y - (a + b * X)

# Simulation loop
no_of_random_ts = int(num_of_rw * mat.shape[1])
sadf_stats = []

for i in range(B):
    innov = np.random.normal(0, 1, size=(T - split_point, no_of_random_ts))
    last_vals = mat[split_point - 1, :no_of_random_ts]
    random_walk_matrix = np.cumsum(innov, axis=0) + last_vals
    mat[split_point:, :no_of_random_ts] = random_walk_matrix
    Y = mat[split_point:, 0]
    X = mat[split_point:, 1]
    post_resid = Y - (a + b * X)
    sadf_stats.append(mf.sadf(np.concatenate([pre_resid, post_resid]), min_win))
    loading_bar(i + 1, B)

# Compute critical values
sadf_stats = np.array(sadf_stats)
res = {
    "90%": np.percentile(sadf_stats, 90),
    "95%": np.percentile(sadf_stats, 95),
    "99%": np.percentile(sadf_stats, 99),
    "all": sadf_stats
}

# Plot
plt.figure(figsize=(6,4))
plt.hist(res['all'], bins=40, density=True, alpha=0.7, color='steelblue')
plt.axvline(res['90%'], color='orange', linestyle='--', label='90%')
plt.axvline(res['95%'], color='red', linestyle='--', label='95%')
plt.axvline(res['99%'], color='purple', linestyle='--', label='99%')
plt.title(f"SADF Null Distribution (T={T})")
plt.xlabel("SADF Statistic")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Table of critical values
pd.DataFrame({ "Critical Value": [res["90%"], res["95%"], res["99%"]] },
             index=["90%", "95%", "99%"])