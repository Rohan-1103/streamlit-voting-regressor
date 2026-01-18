import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def train_voting_regressor(algos, X_train, y_train, X_test, y_test):
    vr = VotingRegressor(algos)
    vr.fit(X_train, y_train)

    y_pred = vr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return vr, r2, mae

# --------------------------------------------------
# Data
# --------------------------------------------------
plt.style.use("seaborn-v0_8-bright")

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

X_train, X_test1, y_train, y_test1 = train_test_split(
    X, y, test_size=0.1, random_state=8
)

X_plot = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Voting Regressor")

estimators = st.sidebar.multiselect(
    "Select Estimators",
    ["Linear Regression", "SVR", "Decision Tree Regressor"],
)

algos = []

if "Linear Regression" in estimators:
    algos.append(("Linear Regression", LinearRegression()))

if "SVR" in estimators:
    algos.append(("SVR", SVR()))

if "Decision Tree Regressor" in estimators:
    algos.append(("Decision Tree", DecisionTreeRegressor(max_depth=5)))

# --------------------------------------------------
# Initial Plot
# --------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(X, y, s=100, color="yellow", edgecolor="black", label="Data")

plot_area = st.pyplot(fig)

# --------------------------------------------------
# Run Button
# --------------------------------------------------
if st.sidebar.button("Run Algorithm"):

    if len(algos) == 0:
        st.warning("Please select at least one estimator.")
    else:
        # Voting Regressor
        vr, vr_r2, vr_mae = train_voting_regressor(
            algos, X_train, y_train, X_test1, y_test1
        )

        y_vr = vr.predict(X_plot)
        ax.plot(X_plot, y_vr, linewidth=3, label="Voting Regressor")

        # Individual Models
        for name, model in algos:
            model.fit(X_train, y_train)
            y_curve = model.predict(X_plot)

            ax.plot(
                X_plot,
                y_curve,
                linestyle="dashdot",
                linewidth=1,
                label=name,
            )

        ax.legend()
        plot_area.pyplot(fig)
