#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
Membership inference attack for Cox models
==============================================================================
"""

import time
import pickle
import json
from lifelines import CoxPHFitter
import copy
import os
import torch

## model auditing
from pathlib import Path
import numpy as np
import torch.utils.data
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Subset
from visualize import plot_roc, plot_roc_log  # original Privacy Meter visualize file

## build model and dataset
import pandas as pd
from lifelines.datasets import load_rossi
import matplotlib.pyplot as plt

## synthetic dataset
import glob


#------------------------------------------------------------------------------
# Split the whole dataset for model training process: half for training, half for validation
#------------------------------------------------------------------------------

def split_dataframe_for_training(dataframe, num_model_pairs):
    """
    Split a Pandas DataFrame into training and test partitions for model pairs.

    Args:
        dataframe (pd.DataFrame): Input dataset as a Pandas DataFrame.
        num_model_pairs (int): Number of model pairs to be trained, with each pair trained on different halves of the dataset.

    Returns:
        data_splits (list): List of dictionaries containing training and test DataFrames for each model.
        master_keep (np.array): Boolean array indicating the membership of samples in each model's training set.
    """
    dataset_size = len(dataframe)
    indices = np.arange(dataset_size)
    split_index = dataset_size // 2
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)
    data_splits = []
    
    training_indices = []

    for i in range(num_model_pairs):
        # Shuffle indices to randomize the dataset
        np.random.shuffle(indices)
        
        # Update master_keep for training and testing sets
        master_keep[i * 2, indices[split_index:]] = False
        master_keep[i * 2 + 1, indices[:split_index]] = False

        # record training indices
        training_indices.append(indices[:split_index].copy())
        training_indices.append(indices[split_index:].copy())
        
        # Generate train and test indices
        train_indices_1 = np.where(master_keep[i * 2, :])[0]
        test_indices_1 = np.where(~master_keep[i * 2, :])[0]

        train_indices_2 = np.where(master_keep[i * 2 + 1, :])[0]
        test_indices_2 = np.where(~master_keep[i * 2 + 1, :])[0]
        
        # Append training and testing DataFrames for both models in the pair
        data_splits.append(
            {
                "train": dataframe.iloc[train_indices_1],
                "test": dataframe.iloc[test_indices_1],
            }
        )
        data_splits.append(
            {
                "train": dataframe.iloc[train_indices_2],
                "test": dataframe.iloc[test_indices_2],
            }
        )

    return data_splits, master_keep, training_indices


#------------------------------------------------------------------------------
# Train the models with data splits
#------------------------------------------------------------------------------

def train_models(data_splits, num_model_pairs, log_dir, save_all_files):
    """
    Train Cox proportional hazards models using data splits and store metadata.

    Args:
        data_splits (list): List of dictionaries containing "train" and "test" DataFrames.
        num_model_pairs (int): Number of model pairs to train.
        log_dir (str): Directory to store models and metadata.

    Returns:
        list: List of trained CoxPHFitter models.
    """
    if (save_all_files):
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
        
    model_list = []
    model_metadata_dict = {}

    for split_idx, split_info in enumerate(data_splits):
        # Extract train and test sets
        train_data = split_info["train"]
        test_data = split_info["test"]

        # Initialize the CoxPH model
        model = CoxPHFitter()
        baseline_time = time.time()

        # Train the Cox model
        model.fit(train_data, duration_col='time', event_col='event')

        # Evaluate the model
        train_score = model.score(train_data, scoring_method="concordance_index")
        test_score = model.score(test_data, scoring_method="concordance_index")

        # Store the trained model
        model_list.append(copy.deepcopy(model))

        model_idx = split_idx

        if (save_all_files):
            with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
                pickle.dump(model, f)

        # Store metadata
        model_metadata_dict[model_idx] = {
            "model_name": "CoxPHFitter",
            "num_train": len(train_data),
            "num_test": len(test_data),
            "train_cindex": train_score,
            "test_cindex": test_score,
            "model_path": f"{log_dir}/cox_model_{model_idx}.pkl",
        }

    if (save_all_files):
        # Save all metadata as a JSON file
        with open(f"{log_dir}/cox_models_metadata.json", "w") as f:
            json.dump(model_metadata_dict, f, indent=4)

    return model_list, model_metadata_dict


#------------------------------------------------------------------------------
# Compute attack signals
#------------------------------------------------------------------------------

# Attack signals based on squared error. one auditing dataset for one model
def get_cox_model_signals_square(model_list, data_list, log_dir, save_all_files):
    """
    Compute attack signals for auditing data splits. One data split corresponds to one model.

    Args:
        model_list (list): List of trained CoxPHFitter models.
        data_list (list): List of auditing data splits.
        log_dir (str): Directory to store attack signals.

    Returns:
        list: List of attack signals.
    """
    if (save_all_files):
        print("calculating attack signals using squared error...")
        
    signals = []
    model_idx = 0
    
    for model in model_list:
        if (save_all_files):
            print(f"Compute attack signals for model {model_idx}...")
        
        predicted_times = torch.tensor(model.predict_expectation(data_list[model_idx]).values) 
        true_events = torch.tensor(data_list[model_idx]['event'].values)
        true_times = torch.tensor(data_list[model_idx]['time'].values)

        model_idx += 1

        n = len(predicted_times)
        sample_losses = np.zeros(n)
        
        for i in range(n):
            if true_events[i] == 1:
                if predicted_times[i] == float('inf'):
                    sample_losses[i] = float('inf')
                else:
                    sample_losses[i] = (predicted_times[i] - true_times[i]) * (predicted_times[i] - true_times[i])
            else:
                if predicted_times[i] == float('inf') or predicted_times[i] >= true_times[i]:
                    sample_losses[i] = 0
                else:
                    '''
                    # option 1: underestimate signals
                    sample_losses[i] = (true_times[i] - predicted_times[i]) * (true_times[i] - predicted_times[i])
                    '''
                    
                    # option 2: manually make signals larger. Question: how to properly choose correction_value?
                    correction_value = 100
                    sample_losses[i] = (true_times[i] - predicted_times[i] + correction_value) * 
                                       (true_times[i] - predicted_times[i] + correction_value)
                    
        signals.append(sample_losses.reshape(-1, 1))
        
    signals = np.concatenate(signals, axis = 1)

    if (save_all_files):
        np.save(
            f"{log_dir}/cox_square_signals.npy",
            signals,
        )
        print("Signals saved to disk.")
        
    return signals

# Attack signals based on individual c-index. one auditing dataset for one model. optimized with vectorization.
def get_cox_model_signals_cindex(model_list, data_list, log_dir, save_all_files):
    if (save_all_files):
        print("computing signals using individual c-index...")
        
    signals = []
    model_idx = 0
    
    for model in model_list:
        if (save_all_files):
            print(f"Compute attack signals for model {model_idx}...")
        
        # Convert data to PyTorch tensors
        predicted_times = torch.tensor(model.predict_expectation(data_list[model_idx]).values)  
        true_events = torch.tensor(data_list[model_idx]['event'].values)
        true_times = torch.tensor(data_list[model_idx]['time'].values)

        model_idx += 1
        
        # Get total number of samples
        n = len(predicted_times)
        
        # Expand dimensions to enable broadcasting
        true_times_i = true_times.view(n, 1)
        true_times_j = true_times.view(1, n)
        predicted_times_i = predicted_times.view(n, 1)
        predicted_times_j = predicted_times.view(1, n)
        true_events_i = true_events.view(n, 1)
        true_events_j = true_events.view(1, n)
        
        # Compute masks for different comparison cases
        non_censored_i = true_events_i == 1
        non_censored_j = true_events_j == 1
        censored_i = ~non_censored_i
        
        # Condition 1: Both are non-censored
        valid_pairs_1 = non_censored_i & non_censored_j
        correct_pairs_1 = ((true_times_i > true_times_j) == (predicted_times_i > predicted_times_j)).float()
        correct_pairs_1 += ((true_times_i == true_times_j) & (predicted_times_i == predicted_times_j)).float() * 0.5
        
        # Condition 2: i is non-censored, j is censored & j's time is >= i's time
        valid_pairs_2 = non_censored_i & ~non_censored_j & (true_times_j >= true_times_i)
        correct_pairs_2 = (predicted_times_j > predicted_times_i).float()
        
        # Condition 3: i is censored, j is non-censored & i's time is >= j's time
        valid_pairs_3 = censored_i & non_censored_j & (true_times_i >= true_times_j)
        correct_pairs_3 = (predicted_times_i > predicted_times_j).float()
        
        # Compute total valid pairs for each `i`
        total_pairs = valid_pairs_1.float().sum(dim=1) + valid_pairs_2.float().sum(dim=1) + valid_pairs_3.float().sum(dim=1)
        
        # Compute total correct pairs for each `i`
        correct_pairs = (valid_pairs_1 * correct_pairs_1).sum(dim=1) + (valid_pairs_2 * correct_pairs_2).sum(dim=1) + (valid_pairs_3 * correct_pairs_3).sum(dim=1)
        
        # Compute sample losses
        sample_losses = torch.where(total_pairs > 0, correct_pairs / total_pairs, torch.zeros_like(total_pairs))

    
        signals.append(sample_losses.reshape(-1, 1))
        
    signals = np.concatenate(signals, axis = 1)

    if (save_all_files):
        np.save(
            f"{log_dir}/cox_cindex_signals.npy",
            signals,
        )
        print("Signals saved to disk.")

    return signals


#------------------------------------------------------------------------------
# Utility functions to audit models using attack signals (almost the same as original Privacy Meter)
#------------------------------------------------------------------------------

def run_cox_loss(target_signals: np.ndarray) -> np.ndarray:
    """
    Attack a target model using the LOSS attack.

    Args:
        target_signals (np.ndarray): Softmax value of all samples in the target model.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member). # reverse: larger score -> not member
    """
    #mia_scores = target_signals  # for cindex signals
    mia_scores = -target_signals  # for square signals
    return mia_scores

def compute_cox_attack_results(mia_scores, target_memberships):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    fpr_list, tpr_list, _ = roc_curve(target_memberships.ravel(), mia_scores.ravel())
    roc_auc = auc(fpr_list, tpr_list)
    one_fpr = tpr_list[np.where(fpr_list <= 0.01)[0][-1]]
    one_tenth_fpr = tpr_list[np.where(fpr_list <= 0.001)[0][-1]]
    zero_fpr = tpr_list[np.where(fpr_list <= 0.0)[0][-1]]

    return {
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": roc_auc,
        "one_fpr": one_fpr,
        "one_tenth_fpr": one_tenth_fpr,
        "zero_fpr": zero_fpr,
    }

def get_cox_audit_results(report_dir, model_idx, mia_scores, target_memberships, save_all_files):
    """
    Generate and save ROC plots for attacking a single model.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        model_idx (int): Index of model subjected to the attack.
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        AUC value
        #dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    attack_result = compute_cox_attack_results(mia_scores, target_memberships)
    
    if (save_all_files):
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        print(
            f"Target Model {model_idx}: AUC {attack_result['auc']:.4f}, "
            f"TPR@0.1%FPR {attack_result['one_tenth_fpr']:.4f}, "
            f"TPR@0.0%FPR {attack_result['zero_fpr']:.4f}"
        )

        plot_roc(
            attack_result["fpr"],
            attack_result["tpr"],
            attack_result["auc"],
            f"{report_dir}/ROC_{model_idx}.png",
        )
        plot_roc_log(
            attack_result["fpr"],
            attack_result["tpr"],
            attack_result["auc"],
            f"{report_dir}/ROC_log_{model_idx}.png",
        )
    
        np.savez(
            f"{report_dir}/attack_result_{model_idx}",
            fpr=attack_result["fpr"],
            tpr=attack_result["tpr"],
            auc=attack_result["auc"],
            one_tenth_fpr=attack_result["one_tenth_fpr"],
            zero_fpr=attack_result["zero_fpr"],
            scores=mia_scores.ravel(),
            memberships=target_memberships.ravel(),
        )

    #return attack_result
    return attack_result['auc']

def audit_cox_models(
    target_model_indices,
    all_signals,
    all_memberships,
    report_dir,
    save_all_files
):
    """
    Audit target model(s) using a Membership Inference Attack algorithm.

    Args:
        report_dir (str): Folder to save attack result.
        target_model_indices (list): List of the target model indices.
        all_signals (np.array): Signal value of all samples in all models (target and reference models).
        all_memberships (np.array): Membership matrix for all models.

    Returns:
        average AUC value of all models
        #list: List of MIA score arrays for all audited target models.
        #list: List of membership labels for all target models.
    """
    all_memberships = np.transpose(all_memberships)

    mia_score_list = []
    membership_list = []
    total_auc = 0.0

    for target_model_idx in target_model_indices:
        if (save_all_files):
            print(f"Auditing the privacy risks of target model {target_model_idx}")
        
        mia_scores = run_cox_loss(all_signals[:, target_model_idx])
        target_memberships = all_memberships[:, target_model_idx]

        mia_score_list.append(mia_scores.copy())
        membership_list.append(target_memberships.copy())

        auc = get_cox_audit_results(
            report_dir, target_model_idx, mia_scores, target_memberships, save_all_files
        )
        total_auc += auc

    average_auc = total_auc / len(target_model_indices)
    return average_auc
    #return mia_score_list, membership_list


#------------------------------------------------------------------------------
# Construct heterogeneous auditing dataset
#------------------------------------------------------------------------------

# half training + half heterogeneous -> random guess = 0.5
def sample_cox_auditing_dataset(training_indices_list, membership, base_training_data, base_auditing_data):
    """
    Creates a list of DataFrames where each DataFrame is a heterogeneous auditing dataset

    Args:
        training_indices_list (list): List of NumPy arrays, each containing indices to overwrite
        base_training_data (pd.DataFrame): Training data used to construct auditing dataset
        base_auditing_data (pd.DataFrame): heterogeneous data used to construct auditing dataset
        membership: membership of training data (returned as-is)

    Returns:
        auditing_data_list: list of heterogeneous auditing dataset 
        membership
    """
    auditing_data_list = []

    for training_indices in training_indices_list:
        # Create a deep copy of auditing_data to avoid modifying the original
        heterogeneous_data = base_auditing_data.copy()
        # Override values at training_indices with data's values
        heterogeneous_data.loc[training_indices] = base_training_data.loc[training_indices]
        auditing_data_list.append(heterogeneous_data)

    return auditing_data_list, membership