#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
Cox Regression Simulation for Heterogeneity Analysis
==============================================================================
This script simulates survival data using Cox regression models to analyze 
the relationship between different distance metrics and privacy exposure.
It uses parallel processing to run multiple simulations efficiently and
generates an HTML report with visualizations of the results.
==============================================================================
"""

#------------------------------------------------------------------------------
# Import Required Libraries
#------------------------------------------------------------------------------
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.metrics import roc_curve, auc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import multiprocessing as mp
from functools import partial
import base64
from io import BytesIO
import webbrowser
from concurrent.futures import ProcessPoolExecutor
import warnings
import json
# MIA resources
from membership_inference_attacks import (
    split_dataframe_for_training,
    train_models,
    sample_cox_auditing_dataset,
    get_cox_model_signals_square,
    audit_cox_models,
    get_cox_model_signals_cindex
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

#------------------------------------------------------------------------------
# Utility Functions for Data Simulation
#------------------------------------------------------------------------------

def simulate_data(n, distribution=np.random.normal, **kwargs):
    """
    Generate data from a specified distribution.
    
    Parameters:
        n (int): Number of observations to generate
        distribution (function): The distribution function to use (default: normal distribution)
        **kwargs: Additional parameters passed to the distribution function
    
    Returns:
        numpy.ndarray: A vector of random values from the specified distribution
    """
    return distribution(size=n, **kwargs)

def rlaplace(n, mu=0, b=0.8):
    """
    Custom implementation of the Laplace distribution.
    
    Parameters:
        n (int): Number of observations
        mu (float): Location parameter (default: 0)
        b (float): Scale parameter (default: 0.8)
    
    Returns:
        numpy.ndarray: A vector of random values from the Laplace distribution
    """
    #print("betas selected from laplace distribution")
    # Generate uniform random values between -0.5 and 0.5
    u = np.random.uniform(-0.5, 0.5, n)
    # Transform to Laplace distribution using inverse CDF
    return mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u))

# draw samples from a uniform distribution
def uniform_distribution(n, lower_bound=-10, upper_bound=10):
    #print("betas selected from uniform distribution")
    return np.random.uniform(lower_bound, upper_bound, n)

def random_select(n, lower_bound=-10, upper_bound=10, num_candidates=1000):
    candidates = np.linspace(lower_bound, upper_bound, num_candidates)
    return np.random.choice(candidates, size=n, replace=True)

def simulate_beta(n, distribution=rlaplace, **kwargs):
    """
    Generate beta coefficients from a specified distribution.
    
    Parameters:
        n (int): Number of coefficients to generate
        distribution (function): The distribution function to use (default: Laplace distribution)
        **kwargs: Additional parameters passed to the distribution function
    
    Returns:
        numpy.ndarray: A vector of beta coefficients
    """
    return distribution(n, **kwargs)

#------------------------------------------------------------------------------
# Survival Data Generation Functions
#------------------------------------------------------------------------------

def generate_events(n, censoring_freq):
    """
    Generate censoring events for survival data.
    
    Parameters:
        n (int): Number of events to generate
        censoring_freq (float): The frequency of censoring (probability of event=0)
    
    Returns:
        numpy.ndarray: A vector of event indicators (0=censored, 1=event occurred)
    """
    # Sample event indicators with specified censoring frequency
    return np.random.choice([0, 1], size=n, p=[censoring_freq, 1-censoring_freq])

def generate_survival_data(betas, parameters):
    """
    Generate survival data based on Cox proportional hazards model.
    
    Parameters:
        betas (numpy.ndarray): Vector of coefficients for the covariates
        parameters (dict): Dictionary of parameters for data generation
    
    Returns:
        pandas.DataFrame: A DataFrame with survival times, event indicators, and covariates
    """
    # Extract parameters
    n = parameters['data']['n']  # Number of samples
    p_hidden = parameters['covariates_hidden']['n']  # Number of hidden variables
    p = parameters['covariates']['n']  # Number of covariates
    
    # Generate covariate matrix with specified mean and standard deviation
    covariate_matrix = np.random.multivariate_normal(
    mean=np.random.uniform(0, 1, p_hidden),
    cov=np.random.uniform(0, 1, (p_hidden, p_hidden)),
    size=n)
    
    # Calculate linear predictor (X * beta)
    linear_predictor = np.dot(covariate_matrix, betas)
    
    # Calculate hazard rates using exponential of linear predictor
    hazard_rate = parameters['baseline_hazard'] * np.exp(linear_predictor)
    
    # Simulate survival times from exponential distribution with calculated rates
    survival_times = np.random.exponential(scale=1/hazard_rate)
    
    # Create DataFrame with a subselection of p out of p_hidden columns 
    data = pd.DataFrame(covariate_matrix[:, :p], columns=[f'X{i+1}' for i in range(p)])
    
    # Add survival time and event indicator
    data['time'] = survival_times
    data['event'] = generate_events(n, parameters['censoring_freq'])
    
    # Reorder columns to put time and event first
    cols = ['time', 'event'] + [col for col in data.columns if col not in ['time', 'event']]
    data = data[cols]
    
    return data
#------------------------------------------------------------------------------
# Distance Metrics
#------------------------------------------------------------------------------

def cosine_similarity(x, y):
    """
    Calculate cosine similarity between two vectors.
    
    Parameters:
        x, y (numpy.ndarray): Vectors to compare
    
    Returns:
        float: The cosine similarity between x and y (range: -1 to 1)
    """
    return np.dot(x, y) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))

def cosine_distance(x, y):
    """
    Calculate cosine distance between two vectors.
    
    Parameters:
        x, y (numpy.ndarray): Vectors to compare
    
    Returns:
        float: The cosine distance between x and y (range: 0 to 2)
    """
    return 1 - cosine_similarity(x, y)

def surv_distance(data1, data2):
    """
    Calculate survival distance between two datasets using log-rank test.
    
    Parameters:
        data1, data2 (pandas.DataFrame): Datasets with survival data (time and event columns)
    
    Returns:
        float: Chi-square statistic from log-rank test (higher = more different)
    """
    # Combine datasets with group indicator
    data1_copy = data1.copy()
    data2_copy = data2.copy()
    data1_copy['group'] = 1
    data2_copy['group'] = 2
    combined = pd.concat([data1_copy, data2_copy])
    
    # Perform log-rank test
    group1 = combined[combined['group'] == 1]
    group2 = combined[combined['group'] == 2]
    
    results = logrank_test(
        group1['time'], 
        group2['time'], 
        group1['event'], 
        group2['event']
    )
    
    # Return chi-square statistic as distance measure
    return results.test_statistic  # Higher value = more different

def privacy_exposure(data1, data2, save_all_files, log_dir, size, num_model_pairs):
    """
    Calculate privacy exposure using membership inference attack.
    
    Parameters:
        data1, data2 (pandas.DataFrame): Datasets with survival data
    
    Returns:
        float: AUC score for membership inference (higher = more privacy exposure)
    """
    try:
        # Prepare training data and heterogeneous dataset (auditing dataset = half from training + half from heterogeneous dataset)
        base_training_data = data1[:size]
        base_auditing_data = data2[:size]

        # Split dataset randomly to construct training data
        data_splits, memberships, training_indices = split_dataframe_for_training(base_training_data, num_model_pairs)

        '''
        # concurrency issue!!!!!!
        for index in training_indices:
            print(index[:5])
            print("     ")
        '''

        # Train cox models
        models, metadata = train_models(data_splits, num_model_pairs, log_dir, save_all_files)

        # Prepare auditing dataset and memberships
        auditing_dataset, auditing_membership = sample_cox_auditing_dataset(training_indices, memberships, base_training_data, base_auditing_data)

        # Compute attack signals
        signals = get_cox_model_signals_square(models, auditing_dataset, log_dir, save_all_files)

        # Get privacy exposure results
        num_experiments = num_model_pairs * 2
        target_model_indices = list(range(num_experiments))

        # Only even indices
        target_model_indices = [x for x in target_model_indices if x % 2 == 0]

        # Take target model AIC values
        AIC_values = [models[idx].AIC_partial_ for idx in target_model_indices]

        #target_model_indices = [i for i in range(num_experiments) if i % 2 == 0]
        average_auc = audit_cox_models(
            target_model_indices,
            signals,
            auditing_membership,
            log_dir,
            save_all_files
        )
        
        if (save_all_files):
            print(f"average AUC: {average_auc}")
            
        return average_auc, AIC_values
        
    except Exception as e:
        print(f"Error in privacy_exposure: {e}")
        return np.nan, []


def calculate_cindex_auditing(data1, data2):
    """
    Calculate C-index (concordance index) on the auditing dataset.
    
    Parameters:
        data1, data2 (pandas.DataFrame): Datasets with survival data (time and event columns)
    
    Returns:
        float: C-index value (range: 0 to 1, where 0.5 is random and 1 is perfect prediction)
    """
    try:
        # Train a Cox proportional hazards model on dataset 1
        cph = CoxPHFitter()
        cph.fit(data1, duration_col='time', event_col='event')
        
        # Calculate C-index on dataset 2
        # This measures how well the model trained on data1 predicts outcomes in data2
        c_index = cph.score(data2, scoring_method="concordance_index")
        
        # Return C-index value
        return c_index
    except Exception as e:
        print(f"Error in calculate_cindex: {e}")
        return np.nan


def calculate_cindex_training(data1, data2):
    """
    Calculate C-index (concordance index) on the training dataset.
    
    Parameters:
        data1, data2 (pandas.DataFrame): Datasets with survival data (time and event columns)
    
    Returns:
        float: C-index value (range: 0 to 1, where 0.5 is random and 1 is perfect prediction)
    """
    try:
        # Train a Cox proportional hazards model on dataset 1
        cph = CoxPHFitter()
        cph.fit(data1, duration_col='time', event_col='event')
        
        # Calculate C-index on dataset 1
        # This measures how well the model trained on data1
        c_index = cph.score(data1, scoring_method="concordance_index")
        
        # Return C-index value
        return c_index
    except Exception as e:
        print(f"Error in calculate_cindex: {e}")
        return np.nan


#------------------------------------------------------------------------------
# Simulation Functions
#------------------------------------------------------------------------------

def run_simulation(i, betas, parameters, n_datasets, n_covariates, n_samples, save_all_files, log_dir, num_model_pairs, betas_distribution):
    """
    Run a single simulation iteration.
    
    Parameters:
        i (int): Simulation iteration number
        betas (numpy.ndarray): Vector of beta coefficients
        parameters (dict): Dictionary of parameters for data generation
        n_datasets (int): Number of datasets to generate
        n_covariates (int): Number of covariates to include in each dataset
        n_samples (int): Number of samples in each dataset
    
    Returns:
        list: Results of the simulation iteration [i, euclidean, manhattan, cosine, survival, privacy, cindex]
"""
    
    try:
        #print(f"simulation {i}")
        #Generate multiple datasets directly instead of sampling from a single dataset
        datasets = []
        selected_betas = []
        beta_list = []
        
        for dataset_idx in range(n_datasets):
            '''
            # Randomly select covariates
            variables = np.random.choice(range(parameters['covariates_hidden']['n']), n_covariates, replace=False)
            
            # Extract corresponding beta values
            subset_betas = betas[variables]
            '''

            # generate betas here! replace the above two lines
            if betas_distribution == "laplace":  
                subset_betas = simulate_beta(parameters['covariates_hidden']['n'], 
                                rlaplace,
                                mu=parameters['covariates_hidden']['mu'], 
                                b=parameters['covariates_hidden']['b'])
                beta_list.append(subset_betas)
            elif betas_distribution == "uniform":
                subset_betas = simulate_beta(parameters['covariates_hidden']['n'], 
                               uniform_distribution)
                beta_list.append(subset_betas)
            elif betas_distribution == "random":
                subset_betas = simulate_beta(parameters['covariates_hidden']['n'], 
                               random_select)
                beta_list.append(subset_betas)
            else:
                print(f"Error in simulation {i}: betas distribution not found!")
                return [i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], np.empty(1)            
            
            # Create a smaller parameter set for this dataset
            dataset_parameters = parameters.copy()
            dataset_parameters['data'] = parameters['data'].copy()
            dataset_parameters['covariates_hidden'] = parameters['covariates_hidden'].copy()
            dataset_parameters['data']['n'] = n_samples
            dataset_parameters['covariates']['n'] = n_covariates
            
            # Generate a new dataset with only the selected covariates
            # We'll use the subset of betas corresponding to the selected variables
            dataset_betas = subset_betas

            # Generate survival data directly for this dataset
            res = generate_survival_data(dataset_betas, dataset_parameters)

            # Rename columns for consistency
            res.columns = ['time', 'event'] + [f'Var{j+1}' for j in range(n_covariates)]
            
            # Add to datasets list
            datasets.append({'data': res, 'betas': subset_betas})

            ##### add beta
            selected_betas.append(subset_betas.tolist())

            ##### Save dataset to parquet file
            if (save_all_files):
                dataset_filename = f"sim_{i}_dataset_{dataset_idx}.parquet"
                res.to_parquet(dataset_filename)

        privacy_exp, AIC_values = privacy_exposure(datasets[0]['data'], datasets[1]['data'], save_all_files, log_dir, n_samples, num_model_pairs)
        average_AIC = np.mean(AIC_values)
        #print(AIC_values)
        #print(average_AIC)
        
        # Calculate distance metrics between datasets
        distances = {
            'euclidean': euclidean(datasets[0]['betas'], datasets[1]['betas']),
            'manhattan': cityblock(datasets[0]['betas'], datasets[1]['betas']),
            'cosine': cosine_distance(datasets[0]['betas'], datasets[1]['betas']),
            'survival': surv_distance(datasets[0]['data'], datasets[1]['data']),
            'privacy': privacy_exp,
            'cindex_auditing': calculate_cindex_auditing(datasets[0]['data'], datasets[1]['data']),
            'cindex_training': calculate_cindex_training(datasets[0]['data'], datasets[1]['data']),
            'AIC': average_AIC
        }

        beta_values = np.array(beta_list) 

        if (save_all_files):
            ##### save distance to json
            distances_filename = f"sim_{i}_distances.json"
            with open(distances_filename, 'w') as f:
                json.dump(distances, f, indent=4)
                
            betas_filename = f"sim_{i}_betas.json"
            print(selected_betas)
            with open(betas_filename, 'w') as f:
                json.dump(selected_betas, f, indent=4)
        
        # Return a single row of results
        return [i, distances['euclidean'], distances['manhattan'], distances['cosine'], 
                distances['survival'], distances['privacy'], distances['cindex_auditing'], distances['cindex_training'], distances['AIC']], beta_values
    except Exception as e:
        print(f"Error in simulation {i}: {e}")
        return [i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], np.empty(1)

#------------------------------------------------------------------------------
# Visualization Functions
#------------------------------------------------------------------------------

def plot_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to convert
    
    Returns:
        str: Base64 encoded string of the figure
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def create_initial_plots(betas, parameters):
    """
    Create initial plots to understand the data.
    
    Parameters:
        betas (numpy.ndarray): Vector of beta coefficients
        parameters (dict): Dictionary of parameters for data generation
    
    Returns:
        dict: Dictionary of matplotlib figures
    """
    all_plots = {}
    
    # Plot the sampled beta values
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(betas, bins=30, color='lightgreen', edgecolor='black', ax=ax)
    ax.set_title('Simulated Beta Values (Laplace)')
    ax.set_xlabel('Beta')
    all_plots['beta_histogram'] = fig
    
    # Generate survival data for visualization
    data = generate_survival_data(betas, parameters)
    
    # Plot the sampled survival times
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['time'], bins=100, color='lightgreen', edgecolor='black', ax=ax)
    ax.set_title('Simulated Survival Times')
    ax.set_xlabel('Times')
    all_plots['time_histogram'] = fig
    
    # Create a Kaplan-Meier survival curve
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    kmf.fit(data['time'], data['event'], label='Survival Curve')
    kmf.plot_survival_function(ax=ax)
    ax.set_title('Kaplan-Meier Survival Curve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    all_plots['kaplan_meier'] = fig
    
    return all_plots


def create_result_plots(results):
    """
    Create plots from simulation results and save correlation coefficients.
    
    Parameters:
        results (pandas.DataFrame): DataFrame with simulation results
    
    Returns:
        dict: Dictionary of matplotlib figures
        dict: Dictionary of correlation coefficients
    """
    all_plots = {}
    correlation_dict = {}
    
    # Create scatterplots of all metrics vs privacy
    metrics = ['euclidean', 'manhattan', 'cosine', 'survival', 'cindex_auditing', 'cindex_training', 'AIC']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=metric, y='privacy', data=results, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
        
        # Calculate and save correlation
        corr = results[[metric, 'privacy']].corr().iloc[0, 1]
        correlation_dict[metric] = corr
        
        # Add annotation to plot
        ax.annotate(f'r = {corr:.3f}', xy=(0.7, 0.1), xycoords='axes fraction', fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Privacy')
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Privacy Metric (AUC)')
        all_plots[f'{metric}_vs_privacy'] = fig

    # Create a violin plot of distance distributions
    # Reshape the data from wide to long format for violin plot
    distances_long = pd.melt(results.drop('run', axis=1), 
                            var_name='Method', 
                            value_name='Distance')
    
    # Create violin plot with embedded boxplot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(x='Method', y='Distance', data=distances_long, inner='box', ax=ax)
    ax.set_title('Distribution of Distance Metrics Between Datasets')
    ax.set_xlabel('Distance Metric')
    ax.set_ylabel('Distance Value')
    all_plots['distance_violin'] = fig
    
    return all_plots, correlation_dict

#------------------------------------------------------------------------------
# HTML Report Generation
#------------------------------------------------------------------------------

def create_html_report(results, all_plots, num_covariates, audit_size, simulation_times, num_models, sd, hidden_cov, log_dir):
    """
    Create an HTML report with simulation results and plots.
    
    Parameters:
        results (pandas.DataFrame): DataFrame with simulation results
        all_plots (dict): Dictionary of matplotlib figures
    
    Returns:
        str: Path to the generated HTML file
    """
    # Define output file name
    report_file = f"{log_dir}/hiddencov{hidden_cov}_covariates_{num_covariates}_auditsize_{audit_size}_num_models_{num_models}_simulation_{simulation_times}_seed{sd}.html"
    
    # Create summary statistics
    summary_stats = results.drop('run', axis=1).describe().to_html()
    
    # Create correlation matrix with color coding
    corr_matrix = results.drop('run', axis=1).corr()
    
    # Create HTML for color-coded correlation matrix
    corr_html = ['<table class="correlation-matrix">']
    
    # Add header row with column names
    corr_html.append('<tr><th></th>')
    for col in corr_matrix.columns:
        corr_html.append(f'<th>{col}</th>')
    corr_html.append('</tr>')
    
    # Add data rows with color coding
    for idx, row in corr_matrix.iterrows():
        corr_html.append(f'<tr><th>{idx}</th>')
        for col in corr_matrix.columns:
            value = row[col]
            
            # Color coding: blue for positive, red for negative, intensity based on magnitude
            if value >= 0:
                # Blue gradient for positive correlations
                color = f'rgba(0, 0, 255, {value})'
                # White text for strong correlations to ensure readability
                text_color = 'white' if value > 0.5 else 'black'
            else:
                # Red gradient for negative correlations
                color = f'rgba(255, 0, 0, {abs(value)})'
                # White text for strong negative correlations
                text_color = 'white' if abs(value) > 0.5 else 'black'
            
            # Create table cell with color coding
            corr_html.append(
                f'<td style="background-color: {color}; color: {text_color};">{value:.3f}</td>'
            )
        corr_html.append('</tr>')
    
    corr_html.append('</table>')
    corr_matrix_html = '\n'.join(corr_html)
    
    # Create the HTML document with CSS styling
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
      <title>Cox Regression Simulation Results (Python)</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333366; }}
        .plot-container {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #f2f2f2; }}
        .correlation-matrix td {{ text-align: center; font-weight: bold; }}
      </style>
    </head>
    <body>
      <h1>Cox Regression Simulation Results (Python)</h1>
      
      <h2>Summary Statistics</h2>
      {summary_stats}
      
      <h2>Correlation Matrix</h2>
      {corr_matrix_html}
      
      <h2>Visualizations</h2>
    """
    
    # Add each plot to the HTML content
    for plot_name, fig in all_plots.items():
        # Convert plot to base64
        img_str = plot_to_base64(fig)
        
        # Format plot name for display (capitalize first letter, replace underscores with spaces)
        display_name = plot_name.replace('_', ' ').title()
        
        # Add the plot to the HTML content
        html_content += f"""
        <div class='plot-container'>
          <h3>{display_name}</h3>
          <img src='data:image/png;base64,{img_str}' width='800'>
        </div>
        """
        
        # Close the figure to free memory
        plt.close(fig)
    
    # Close the HTML content
    html_content += """
    </body>
    </html>"""
    
    # Write the HTML content to a file
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    return report_file

#------------------------------------------------------------------------------
# Main Simulation Function
#------------------------------------------------------------------------------

def quantify_privacy(sd, log_dir, n_hidden, n_cov, sample_size, n_runs, stats_summary_dir, betas_distribution):
    """
    Main function to run the simulation.
    """
    print("Starting Cox Regression Simulation...")
    start_time = time.time()
    
    #----------------------------------------------------------------------------
    # Simulation Parameters
    #----------------------------------------------------------------------------
    # Set random seed for reproducibility
    np.random.seed(sd)
    
    # Define parameters for data generation
    parameters = {
        # Parameters for covariate generation
        'covariates_hidden': {
            'n': n_hidden,                  # Number of covariates
            'mu': 0,                   # Mean of Laplace distribution
            'b': 0.1                   # Scale parameter of Laplace distribution
        },
        'covariates': {
            'n': n_cov                   # Number of hidden covariates
        },
        # Parameters for data generation
        'data': {
            'n': sample_size,                 # Number of samples
            'mean': 10,                # Mean of normal distribution
            'sd': 1                    # Standard deviation of normal distribution
        },
        'baseline_hazard': 0.01,       # Baseline hazard for Cox model
        'censoring_freq': 0.1          # Frequency of censoring
    }

    if betas_distribution == "laplace":
    # Generate beta values using Laplace distribution
        betas = simulate_beta(parameters['covariates_hidden']['n'], 
                              rlaplace,
                              mu=parameters['covariates_hidden']['mu'], 
                              b=parameters['covariates_hidden']['b'])
    elif betas_distribution == "uniform":
        betas = simulate_beta(parameters['covariates_hidden']['n'], 
                              uniform_distribution)
    elif betas_distribution == "random":
        betas = simulate_beta(parameters['covariates_hidden']['n'], 
                              random_select)
    else:
        print("Betas distribution not found!")
        return
    
    #----------------------------------------------------------------------------
    # Initial Data Visualization
    #----------------------------------------------------------------------------
    
    # Generate plots for the first run to understand the data
    #all_plots = create_initial_plots(betas, parameters)
    
    #----------------------------------------------------------------------------
    # Dataset Generation Parameters
    #----------------------------------------------------------------------------
    # Number of samples in each dataset
    n_samples = parameters['data']['n']  # auditing dataset size. (training size = size // 2)
    
    # Number of covariates to include in each dataset
    n_covariates = parameters['covariates']['n']

    
    # Number of datasets to generate for each simulation
    n_datasets = 2
    num_model_pairs = 1  # for each pair of data, how many models are tested. (number of models = 2 * num_model_pairs)
        
    # Check if we have enough covariates
    if n_covariates > parameters['covariates_hidden']['n']:
        raise ValueError("Too many covariates chosen for datasets")
    
    #----------------------------------------------------------------------------
    # Run Parallel Simulation
    #----------------------------------------------------------------------------
    print(f"Running {n_runs} simulations in parallel...")
    
    # Determine number of processes to use (leave one core for system)
    #n_processes = max(1, mp.cpu_count() - 1)
    n_processes = 1
    print(f"Using {n_processes} processes")

    save_all_files = False
    print(f"Save files: {save_all_files}")
    
    # Create a partial function with fixed parameters
    run_sim = partial(
        run_simulation,
        betas=betas,
        parameters=parameters, 
        n_datasets=n_datasets, 
        n_covariates=n_covariates, 
        n_samples=n_samples,
        save_all_files=save_all_files,
        log_dir=log_dir,
        num_model_pairs=num_model_pairs,
        betas_distribution=betas_distribution
    )
    
    # Run the simulation in parallel
    '''
    results_list = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results_list = list(executor.map(run_sim, range(n_runs)))
    '''
    
    results = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(run_sim, range(n_runs)))

    results_list = np.array([res[0] for res in results])
    complete_beta_values = [res[1] for res in results]

    #all_plots = create_initial_plots(complete_beta_values, parameters)
    
    #----------------------------------------------------------------------------
    # Process Results
    #----------------------------------------------------------------------------
    # Convert results_list to DataFrame
    results = pd.DataFrame(
        results_list, 
        columns=['run', 'euclidean', 'manhattan', 'cosine', 'survival', 'privacy', 'cindex_auditing', 'cindex_training', 'AIC']
    )
    
    # Remove any rows with NaN values
    results = results.dropna()
    
    #----------------------------------------------------------------------------
    # Analysis and Visualization
    #----------------------------------------------------------------------------
    # Display summary statistics
    summary_stats = results.drop('run', axis=1).describe()
    print("\nSummary statistics:")
    print(summary_stats)
    
    # save summary stats to json file
    stats_file = f"{stats_summary_dir}/statistics_hiddencov{n_hidden}_covariates_{n_cov}_auditsize_{sample_size}_simulation_{n_runs}_seed{sd}.json"
    summary_stats.to_json(stats_file)   # JSON
    
    # Create correlation matrix
    corr_matrix = results.drop('run', axis=1).corr()
    
    # Create result plots
    all_plots, corrs = create_result_plots(results)

    # capture results and save to json file
    correlation_file = f"{stats_summary_dir}/correlation_hiddencov{n_hidden}_covariates_{n_cov}_auditsize_{sample_size}_simulation_{n_runs}_seed{sd}.json"
    with open(correlation_file, 'w') as f:
        json.dump(corrs, f)
    
    # Combine all plots
    #all_plots.update(result_plots)
    
    #----------------------------------------------------------------------------
    # Generate HTML Report
    #----------------------------------------------------------------------------
    report_file = create_html_report(results, all_plots, n_covariates, n_samples, n_runs, num_model_pairs, sd, parameters['covariates_hidden']['n'], log_dir)
    
    # Display message about report generation
    print(f"\nHTML report generated: {report_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    # Open the HTML report in the default web browser
    #webbrowser.open('file://' + os.path.realpath(report_file))


def main():
    sd = 123
    n_runs = 500
    betas_distribution = "random"

    # sample size 1000
    sample_size = 1000
    log_dir = f"updated_test/in_loop_beta_{betas_distribution}/samplesize_{sample_size}"
    stats_summary_dir = f"updated_test/in_loop_beta_{betas_distribution}/samplesize_{sample_size}/stats_summary"

    os.makedirs(stats_summary_dir, exist_ok=True)
    print(stats_summary_dir)

    hidden_cov_list = [2, 4, 6, 8, 10, 20, 40, 80, 100]
    cov_list = [[2], [2,4], [2,4,6], [2,4,6,8], [2, 4, 6,8, 10], [2,4,8,10,20], [2,4,8,10,20,30,40], [2,4,8,10,20,40,60,80], [2,4,8,10,20,40,60,80,100]]
    
    for i in range(len(hidden_cov_list)):
        for n_cov in cov_list[i]:
            quantify_privacy(sd, log_dir, hidden_cov_list[i], n_cov, sample_size, n_runs, stats_summary_dir, betas_distribution)

    # sample size 100
    sample_size = 100
    log_dir = f"updated_test/in_loop_beta_{betas_distribution}/samplesize_{sample_size}"
    stats_summary_dir = f"updated_test/in_loop_beta_{betas_distribution}/samplesize_{sample_size}/stats_summary"

    os.makedirs(stats_summary_dir, exist_ok=True)
    print(stats_summary_dir)

    hidden_cov_list = [2, 4, 6, 8, 10, 20, 40, 80, 100]
    cov_list = [[2], [2,4], [2,4,6], [2,4,6,8], [2, 4, 6, 8, 10], [2,4,8,10,20], [2,4,8,10,20,30,40], [2,4,8,10,20,40], [2,4,8,10,20,40]]
    
    for i in range(len(hidden_cov_list)):
        for n_cov in cov_list[i]:
            quantify_privacy(sd, log_dir, hidden_cov_list[i], n_cov, sample_size, n_runs, stats_summary_dir, betas_distribution)


if __name__ == "__main__":
    main()
