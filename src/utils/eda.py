
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(df):
    """Plot the correlation matrix of the DataFrame."""
    plt.figure(figsize=(12, 10))
    
    
    sns.pairplot(df, vars=["age","salary","tenure","risk_score","censor_time"])
    plt.title("Correlation Matrix")
    plt.show()
def plot_censoring_distribution(df,true_params):
    """Plot the distribution of censoring times."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['censor_time'], bins=30, kde=True)
    plt.title("Censoring Time Distribution")
    plt.xlabel("Censoring Time")
    plt.ylabel("Frequency")
    for key in true_params:
        if key == "censoring_rate":
            plt.text(5,80, f"Censoring rate:{true_params['censoring_rate']:.2f}, Censoring type:{true_params['censoring_type']}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        elif key == 'informative_censor_strength':
            plt.text(5,80, f"Informative censor strength:{true_params['informative_censor_strength']:.2f}, Censoring type:{true_params['censoring_type']}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.grid()
    plt.show()

def true_weibell(df,true_params):
    import numpy as np 
    from scipy.stats import weibull_min # standard weibull distribution
    shape = true_params["weibull_shape"]
    scale = true_params["weibull_scale"]
    t = np.linspace(0, df.time.max(), 200)
    S_true = weibull_min.sf(t, c=shape, scale=scale)
    plt.plot(t, S_true, label="True survival", linewidth=2)
    plt.title("True Weibull Survival Function")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability S(t)")
    plt.legend()


def kap_meier_curve(df):
    """Calculate the survival function S(t) at time t given parameters."""

    km = KaplanMeierFitter()
    km.fit(durations=df['time'], event_observed=df['event'])
    plt.figure(figsize=(10, 6))
    km.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability S(t)")
    plt.text(5, 0.8, f"Number of events: {df['event'].sum()}\nTotal observations: {len(df)}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.grid()
    plt.show()

    

        

