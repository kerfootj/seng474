import matplotlib.pyplot as plt

X = [0.5, 0.6, 0.7, 0.8, 0.9]
# Heart Data
dt_tr = [0.89261745, 0.793296089, 0.8125, 0.852941176, 0.847014925]
dt_te = [0.797297297, 0.830508475, 0.808988764, 0.86440678, 0.862068966]

rf_tr = [0.919463087, 0.82122905, 0.990384615, 0.861344538, 0.873134328]
rf_te = [0.864864865, 0.898305085, 0.853932584, 0.881355932, 0.896551724]

nn_tr = [0.818791946, 0.815642458, 0.850961538, 0.823529412, 0.861940299]
nn_te = [0.878378378, 0.872881356, 0.853932584, 0.881355932, 0.896551724]

# Spam
# dt_tr = [0.9873967840069535,0.9634190510684535,0.9664700403601366,0.9679434936158653,0.9753682685341705]
# dt_te = [0.9191304347826087,0.9119565217391304,0.9239130434782609,0.9260869565217391,0.9260869565217391]

# rf_tr = [0.9461103867883529,0.9514668598333937,0.942564420987271,0.9418636240152133,0.9425259599130644]
# rf_te = [0.9295652173913044,0.9266304347826086,0.9282608695652174,0.9402173913043478,0.9108695652173913]

# nn_tr = [0.933941764450239,0.9460340456356393,0.941943495808755,0.9440369464819343,0.9475971987442646]
# nn_te = [0.94,0.933695652173913,0.9333333333333333,0.9434782608695652,0.9326086956521739]

fig, ax = plt.subplots()
ax.set_xlabel("percent training data")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Size of Trainig Set - Decsion Tree")
ax.plot(X, dt_tr, label="train")
ax.plot(X, dt_te, label="test")
ax.legend()
plt.savefig("heart_dt_sizes.png")

fig, ax = plt.subplots()
ax.set_xlabel("percent training data")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Size of Trainig Set - Random Forest")
ax.plot(X, rf_tr, label="train")
ax.plot(X, rf_te, label="test")
ax.legend()
plt.savefig("heart_rf_sizes.png")

fig, ax = plt.subplots()
ax.set_xlabel("percent training data")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Size of Trainig Set - Neural Networks")
ax.plot(X, nn_tr, label="train")
ax.plot(X, nn_te, label="test")
ax.legend()
plt.savefig("heart_nn_sizes.png")