import matplotlib.pyplot as plt
import numpy as np
import os
import json
from data_utils import extract_results, parse_use_scores, get_binary_scores, get_avg_probs, get_geometric_mean_probs
from platt_scale import platt_rescale
from reliability_plot import draw_reliability_graph
from sklearn import metrics
import pandas as pd

def geom_mean(probs_list):
    return np.prod(probs_list) ** (1.0 / len(probs_list))

def calculate_calibration_metrics(predicted_probabilities, 
                                true_labels,
                                num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
 
    # get binary class predictions from confidences
    predicted_labels = np.ones_like(predicted_probabilities)

    # get a boolean list of correct/false predictions
    accuracies = predicted_labels == true_labels
    bin_accuracies = []
    samples_per_bin = []
    binned_pred_probs = []
    binned_true_labels = []

    ece = 0  
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(
            predicted_probabilities > bin_lower.item(),
            predicted_probabilities <= bin_upper.item(),
        )
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        probability_in_bin = in_bin.astype(float).mean()

        if probability_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            bin_accuracies.append(accuracy_in_bin)

            labels_in_bin = true_labels[in_bin]
            binned_pred_probs.append(predicted_probabilities[in_bin])
            binned_true_labels.append(labels_in_bin)
            samples_per_bin.append(len(labels_in_bin))
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = predicted_probabilities[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            bin_ece = (
                np.abs(avg_confidence_in_bin - accuracy_in_bin) * probability_in_bin
            )
            ece += bin_ece
        else:
            bin_accuracies.append(np.nan)
            binned_pred_probs.append([])
            binned_true_labels.append([])
            samples_per_bin.append(0)

    brier_score = metrics.brier_score_loss(
        true_labels,
        predicted_probabilities,
    )
    p_correct = np.average(true_labels)
    unskilled_brier_score = p_correct * (1 - p_correct)
    skill_score = (unskilled_brier_score - brier_score) / unskilled_brier_score

    return {
        "ECE": ece,
        "p_correct": p_correct,
        "Brier Score": brier_score, 
        "Unskilled Brier Score": unskilled_brier_score, 
        "Skill score": skill_score,
        "Bins": np.linspace(0, 1, num_bins + 1),
        "Bin Confs": [np.mean(probs) for probs in binned_pred_probs],
        "Bin Accuracies": bin_accuracies,
        "Bin Sizes": np.array(samples_per_bin)/len(predicted_probabilities),
        "Num Samples": len(predicted_probabilities),
        "Samples Per Bin": samples_per_bin,
    }

def rescale(model_metric_scores):
    return (np.array(model_metric_scores) + 1) / 2

def scores_with_repo(use_scores_dict, test_data_path: str):
    repo_to_scores = {}
    test_json = []
    for line in open(test_data_path, 'r', encoding="utf-8"):
        test_json.append(json.loads(line))

    for i, score in use_scores_dict.items():
        repo = test_json[int(i)]['repo']
        if repo not in repo_to_scores:
            repo_to_scores[repo] = []
        repo_to_scores[repo].append({"index": i, "score": score})

    return repo_to_scores

def train_test_split_by_repo_balanced(repo_to_scores, num_folds=5):
    # Flatten the scores to calculate total number of samples
    total_samples = sum(len(scores) for scores in repo_to_scores.values())
    target_samples_per_fold = total_samples / num_folds
    
    # Convert repo_to_scores to a list of items, sorted by the number of samples per repo
    items = sorted(repo_to_scores.items(), key=lambda x: len(x[1]), reverse=True)
    
    folds = []
    tests_used = []
    for fold in range(num_folds):
        train_sample, test_sample = [], []
        train_metadata, test_metadata = {}, {}
        test_samples_count = 0
        
        # Dynamically allocate repositories to train or test to balance the number of samples
        for repo, scores in items:
            # If adding this repo to test won't exceed the target number of samples for the fold, add it to test
            if (repo not in tests_used) and (test_samples_count + len(scores) <= target_samples_per_fold):
                test_sample += scores
                test_metadata[repo] = len(scores)
                test_samples_count += len(scores)
                tests_used.append(repo)
            else:
                # Otherwise, add it to train
                train_sample += scores
                train_metadata[repo] = len(scores)
        
        # Save metadata and samples for the fold
        # with open(f'train_metadata_{fold}.json', 'w') as file:
        #     json.dump(train_metadata, file)
        # with open(f'test_metadata_{fold}.json', 'w') as file:
        #     json.dump(test_metadata, file)
        
        print(f"Fold {fold}")
        print(f"Train Length: {len(train_sample)}")
        print(f"Train Num Repos: {len(train_metadata.keys())}")
        print(f"Test Length: {len(test_sample)}")
        print(f"Test Num Repos: {len(test_metadata.keys())}")
        
        folds.append((train_sample, test_sample))
    
    return folds

folds_cache = None
folds_cache_by_length = {}
def get_scaled_by_length_metrics_repo_split_kfold(predicted_scores, 
                                        actual_scores,
                                        thresholds,
                                        test_data,
                                        num_bins,
                                        model_filename,
                                        length_ranges: list[tuple[int, int]] = None):
    # predicted_correctness: index -> (length, prob)
    # actual_correctness: index -> score
    for length_range in length_ranges:
        filtered_predicted_scores = {index: prob for index, (length, prob) in predicted_scores.items() if length_range[0] <= length <= length_range[1]}
        filtered_actual_scores = {index: score for index, score in actual_scores.items() if index in filtered_predicted_scores.keys()}

        global folds_cache_by_length
        if length_range not in folds_cache_by_length:
            repo_to_scores = scores_with_repo(filtered_actual_scores, test_data)
            folds_cache_by_length[length_range] = train_test_split_by_repo_balanced(repo_to_scores, 5) 
            for i in range(5):
                print(f"Fold {i} train scores length:", len(folds_cache_by_length[length_range][i][0]))
                print(f"Fold {i} test scores length:", len(folds_cache_by_length[length_range][i][1]))

        folds = folds_cache_by_length[length_range]
        fig_scaled, ax_scaled = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)   
        combined_test_scores = []
        for _, test in folds:
            for test_sample in test:
                test_i = int(test_sample["index"])
                combined_test_scores.append(test_i)
        scaled_evaluations_json = {}
        scaled_evaluations = {}
        thresholds['optimal'] = 0.49
        for threshold_label, threshold in thresholds.items():
            if threshold_label not in ("optimal_f1_threshold", "high_p_threshold", "high_r_threshold"): continue        
            if threshold_label not in scaled_evaluations_json.keys():
                scaled_evaluations_json[threshold_label] = []
            if threshold_label not in scaled_evaluations.keys():
                scaled_evaluations[threshold_label] = []
            true_binary_scores = {int(index): filtered_actual_scores[index]>threshold for index in filtered_actual_scores.keys()}
            all_scaled_test_probs = np.array([])
            all_test_true_labels = np.array([])
            for train, test in folds:
                train_probs = np.array([])
                test_probs = np.array([])
                train_true_labels = np.array([])
                test_true_labels = np.array([])
                test_probs_lengths = np.array([])
                for train_sample in train:
                    train_i = int(train_sample["index"])
                    train_probs = np.append(train_probs, filtered_predicted_scores[train_i])
                    train_true_labels = np.append(train_true_labels, true_binary_scores[train_i])            
                for test_sample in test:
                    test_i = int(test_sample["index"])
                    test_probs = np.append(test_probs, filtered_predicted_scores[test_i])
                    test_true_labels = np.append(test_true_labels, true_binary_scores[test_i])
                    test_scaled_probs = platt_rescale(train_probs, 
                                                    train_true_labels, 
                                                    test_probs, 
                                                    test_true_labels)
                all_scaled_test_probs = np.append(all_scaled_test_probs, test_scaled_probs)
                all_test_true_labels = np.append(all_test_true_labels, test_true_labels)
            
                scaled_results = calculate_calibration_metrics(predicted_probabilities=np.array(all_scaled_test_probs), 
                true_labels=np.array(all_test_true_labels, dtype=bool),
                num_bins=10)
                scaled_results["Threshold"] = threshold
                scaled_results["length_range"] = length_range
                scaled_results["Bin Confs"] = list(scaled_results["Bin Confs"])
                scaled_results["Bin Sizes"] = list(scaled_results["Bin Sizes"])
                scaled_evaluations[threshold_label].append(scaled_results)

                scaled_evaluations_json[threshold_label].append({
                    "ECE": scaled_results["ECE"],
                    "p_correct": scaled_results["p_correct"],
                    "Brier Score": scaled_results["Brier Score"], 
                    "Unskilled Brier Score": scaled_results["Unskilled Brier Score"], 
                    "Skill score": scaled_results["Skill score"],
                    "Threshold": threshold,
                    "length_range": length_range
                }) 
        
    return scaled_evaluations, scaled_evaluations_json

def get_optimal_token_cutoff(train_probs, train_true_labels, thresholds, test_data, num_bins, model_filename, length_ranges):
    position_to_ss = {}
    for token_position_cutoff in range(1, 20):
        scaled_evaluations, scaled_evaluations_json = get_scaled_metrics_repo_split_kfold(train_probs, train_true_labels, thresholds, test_data, num_bins, model_filename, length_ranges, token_position_cutoff)
        position_to_ss[token_position_cutoff] = scaled_evaluations["optimal"][0]["Skill score"]
    optimal_cutoff = max(position_to_ss, key=position_to_ss.get)
    print(f"Optimal token position cutoff: {optimal_cutoff}")
    return optimal_cutoff

def get_scaled_metrics_repo_split_kfold(predicted_scores, 
                                        actual_scores,
                                        thresholds,
                                        test_data,
                                        num_bins,
                                        model_filename,
                                        length_ranges: list[tuple[int, int]] = None,
                                        token_position_cutoff: int = None,
                                        filter_by_token_cutoff: bool = False,
                                        mean_cutoff: bool = True):
    # predicted_correctness: index -> (length, prob)
    # actual_correctness: index -> score
    # global folds_cache
    # if folds_cache is None:
    repo_to_scores = scores_with_repo(actual_scores, test_data)
    folds = train_test_split_by_repo_balanced(repo_to_scores, 5) 
    for i in range(5):
        print(f"Fold {i} train scores length:", len(folds[i][0]))
        print(f"Fold {i} test scores length:", len(folds[i][1]))

    # folds = folds_cache
    fig_scaled, ax_scaled = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)   
    combined_test_scores = []
    for _, test in folds:
        for test_sample in test:
            test_i = int(test_sample["index"])
            combined_test_scores.append(test_i)
    scaled_evaluations_json = {}
    scaled_evaluations = {}
    thresholds['optimal'] = 0.49#thresholds['optimal_f1_threshold']#0.36
    for threshold_label, threshold in thresholds.items():
        if threshold_label not in ("optimal"): continue        
        if threshold_label not in scaled_evaluations_json.keys():
            scaled_evaluations_json[threshold_label] = []
        if threshold_label not in scaled_evaluations.keys():
            scaled_evaluations[threshold_label] = []
        true_binary_scores = {int(index): actual_scores[index]>threshold for index in actual_scores.keys()}

        all_scaled_test_probs = np.array([])
        all_test_true_labels = np.array([])
        all_test_probs_lengths = np.array([])
        token_position_cutoffs = []
        for train, test in folds:
            train_probs = []
            test_probs = []
            train_true_labels = []
            test_true_labels = []
            train_predicted_scores_dict = {}
            test_predicted_scores_dict = {}
            train_actual_scores_dict = {}
            test_actual_scores_dict = {}
            for train_sample in train:
                train_i = int(train_sample["index"])
                train_probs.append(predicted_scores[train_i])
                train_true_labels.append(true_binary_scores[train_i])
                train_predicted_scores_dict[train_i] = predicted_scores[train_i]
                train_actual_scores_dict[train_i] = actual_scores[train_i]
            for test_sample in test:
                test_i = int(test_sample["index"])

                # print(f"test_i: {test_i}")
                # print(f"Keys in predicted_scores: {list(predicted_scores.keys())[:10]}...")  # Print first 10 keys
                # print(f"Number of keys in predicted_scores: {len(predicted_scores)}")
                # print(f"Keys in actual_scores: {list(actual_scores.keys())[:10]}...")  # Print first 10 keys
                # print(f"Number of keys in actual_scores: {len(actual_scores)}")

                test_probs.append(predicted_scores[test_i])
                test_predicted_scores_dict[test_i] = predicted_scores[test_i]
                test_true_labels.append(true_binary_scores[test_i])
                test_actual_scores_dict[test_i] = actual_scores[test_i]
            print("computing optimal cutoff")
            cutoff_computed = False
            if not token_position_cutoff:
                token_position_cutoff = get_optimal_token_cutoff(train_predicted_scores_dict.copy(), train_actual_scores_dict.copy(), thresholds, test_data, num_bins, 
                model_filename, length_ranges)
                print(f"Token position cutoff: {token_position_cutoff}")
                token_position_cutoffs.append(token_position_cutoff)
                cutoff_computed = True

            if filter_by_token_cutoff:
                train_probs_mask = [len(probs_list) >= token_position_cutoff for probs_list in train_probs]
                test_probs_mask = [len(probs_list) >= token_position_cutoff for probs_list in test_probs]
                train_probs = [probs_list for i, probs_list in enumerate(train_probs) if train_probs_mask[i]]
                train_true_labels = [train_true_labels[i] for i, train_true_label in enumerate(train_true_labels) if train_probs_mask[i]]
                test_probs = [probs_list for i, probs_list in enumerate(test_probs) if test_probs_mask[i]]
                test_true_labels = [test_true_labels[i] for i, test_true_label in enumerate(test_true_labels) if test_probs_mask[i]]

            if mean_cutoff:
                train_probs = [geom_mean(probs_list[:token_position_cutoff]) for probs_list in train_probs]
                test_probs = [geom_mean(probs_list[:token_position_cutoff]) for probs_list in test_probs]
            else:
                train_probs = [geom_mean(probs_list) for probs_list in train_probs]
                test_probs = [geom_mean(probs_list) for probs_list in test_probs]

            test_scaled_probs = platt_rescale(train_probs, 
                                                train_true_labels, 
                                                test_probs, 
                                                test_true_labels)
            all_scaled_test_probs = np.append(all_scaled_test_probs, test_scaled_probs)
            # all_test_probs_lengths = np.append(all_test_probs_lengths, test_probs_lengths)
            all_test_true_labels = np.append(all_test_true_labels, test_true_labels)
            if cutoff_computed: token_position_cutoff = None
        
        for length_range in length_ranges:
            # filtered_test_probs = [prob for i, prob in enumerate(all_scaled_test_probs) if length_range[0] <= all_test_probs_lengths[i] <= length_range[1]]
            # filtered_test_true_labels = [label for i, label in enumerate(all_test_true_labels) if length_range[0] <= all_test_probs_lengths[i] <= length_range[1]]
            scaled_results = calculate_calibration_metrics(predicted_probabilities=np.array(all_scaled_test_probs), 
            true_labels=np.array(all_test_true_labels, dtype=bool),
            num_bins=10)
            scaled_results["Threshold"] = threshold
            scaled_results["length_range"] = length_range
            scaled_results["Bin Confs"] = list(scaled_results["Bin Confs"])
            scaled_results["Bin Sizes"] = list(scaled_results["Bin Sizes"])
            scaled_results["token_position_cutoff"] = token_position_cutoff
            scaled_results["num_samples"] = len(all_scaled_test_probs)
            scaled_evaluations[threshold_label].append(scaled_results)

            scaled_evaluations_json[threshold_label].append({
                "ECE": scaled_results["ECE"],
                "p_correct": scaled_results["p_correct"],
                "Brier Score": scaled_results["Brier Score"], 
                "Unskilled Brier Score": scaled_results["Unskilled Brier Score"], 
                "Skill score": scaled_results["Skill score"],
                "Threshold": threshold,
                "length_range": length_range,
                "token_position_cutoffs": token_position_cutoffs,
                "num_samples": len(all_scaled_test_probs),
                "errors": [test_prob - test_true_label for test_prob, test_true_label in zip(all_scaled_test_probs, all_test_true_labels)]
            }) 
    return scaled_evaluations, scaled_evaluations_json
                
def get_calibration_metrics_raw(predicted_correctness_dict: dict, 
                                actual_correctness_dict: dict,
                                thresholds: list,
                                model_filename: str,
                                avg: bool = False,
                                length_ranges: list[tuple[int, int]] = None):
    raw_evaluations = {}
    raw_evaluations_json = {}
    for length_range in length_ranges:
        # predicted_correctness: index -> (length, prob)
        # actual_correctness: index -> score
        filtered_predicted_correctness = {index: prob for index, (length, prob) in predicted_correctness_dict.items() if length_range[0] <= length <= length_range[1]}
        filtered_actual_correctness = {index: score for index, score in actual_correctness_dict.items() if index in filtered_predicted_correctness.keys()}
        predicted_correctness, actual_correctness = aligned_lists_from_dictionary(filtered_predicted_correctness, filtered_actual_correctness)

        thresholds['optimal'] = 0.49#thresholds['optimal_f1_threshold']#0.36
        for threshold_label, threshold in thresholds.items():
            if threshold_label != "optimal": continue
            if threshold_label not in raw_evaluations_json.keys():
                raw_evaluations_json[threshold_label] = []
            if threshold_label not in raw_evaluations.keys():
                raw_evaluations[threshold_label] = []
            true_binary_scores = get_binary_scores(actual_correctness, threshold)
            
            # raw_results = ExperimentResults(predicted_probabilities=np.array(predicted_correctness), 
            #                                 true_labels=true_binary_scores,
            #                                 bin_strategy=BinStrategy.Uniform,
            #                                 num_bins=10)

            raw_results = calculate_calibration_metrics(predicted_probabilities=np.array(predicted_correctness), 
                                                    true_labels=true_binary_scores,
                                                    num_bins=10)
            raw_results["Threshold"] = threshold
            raw_results["length_range"] = length_range
            raw_evaluations[threshold_label].append(raw_results)

            raw_evaluations_json[threshold_label].append({
                "ECE": raw_results["ECE"],
                "p_correct": raw_results["p_correct"],
                "Brier Score": raw_results["Brier Score"], 
                "Unskilled Brier Score": raw_results["Unskilled Brier Score"], 
                "Skill score": raw_results["Skill score"],
                "Threshold": threshold,
                "length_range": length_range
            }) 
    return raw_evaluations, raw_evaluations_json

def aligned_lists_from_dictionary(dict1, dict2):
    common_keys = dict1.keys() & dict2.keys()
    values_from_dict1 = [dict1[key] for key in common_keys]
    values_from_dict2 = [dict2[key] for key in common_keys]
    return values_from_dict1, values_from_dict2

def odds(probs):
    odds=[]
    for prob in probs:
        if prob == 1:
            odds.append(1)
        else:
            odds.append(prob / (1 - prob))
    return odds

if __name__ == "__main__":
    languages = ["Python", "Java"]
    model_filenames = [
        "gpt-3.5-turbo_ASAP.txt",
        "gpt-3.5-turbo_BM25.txt",
        "CodeLlama-70b-hf_ASAP.txt",
        "CodeLlama-70b-hf_BM25.txt",
        "deepseek-coder-33b-instruct_ASAP.txt",
        "deepseek-coder-33b-instruct_BM25.txt"
    ]
    metric = "BERT Score-R"
    summary_length_filters = [(0, 100)]
    scaled_evaluations = {}

    for language in languages:
        scaled_evaluations[language] = {}
        for model_filename in model_filenames:
            print(f"Processing {language} - {model_filename}")
            
            gold_file_path = f"./data/{language}/prompting_data/gold.txt"
            results_dir = f"./data/{language}/model_outputs/"
            metric_path = f"./data/{language}/metrics_results/bert-score-recall/"
            model_name = model_filename[:-9]
            prompting_method = model_filename[-8:-4]
            results_path = os.path.join(results_dir, model_filename)

            raw_calib_by_position = []
            scaled_calib_by_position = []

            scaled_evaluations[language][model_filename] = {}

            for token_position_cutoff in range(1, 101):
                print(f"At position {token_position_cutoff}")
                # Some log probs are actually probs in the data
                # get them in log probs and convert all into probs here
                if (model_filename == "gpt-3.5-turbo_ASAP.txt" or model_filename == "gpt-3.5-turbo_BM25.txt") and language == "Java":
                    model_results, invalid_results = extract_results(results_path, probs=False)
                else:
                    model_results, invalid_results = extract_results(results_path, probs=True)

                probs_per_summary = [np.exp(result['log_probs']) for result in model_results.values()]
                odds_per_summary = [odds(probs) for probs in probs_per_summary]
                log_probs_per_summary = [result['log_probs'] for result in model_results.values()]
            
                avg_probs_dict = {int(result['index']): (len(result['log_probs']), np.mean(np.exp(result['log_probs'][:min(token_position_cutoff, len(result['log_probs']))]))) for result in model_results.values()}
                probs_list_dict = {int(result['index']): np.exp(result['log_probs']) for result in model_results.values()}

                model_metric_path = os.path.join(metric_path, model_filename)
                indices, model_metric_scores = parse_use_scores(model_metric_path)
                length = min(5000, len(model_metric_scores))
                model_metric_scores = model_metric_scores[:length]
                model_metric_scores_dict = {int(index): score for index, score in zip(indices, model_metric_scores)}

                model_metric_scores = rescale(model_metric_scores)
                
                threshold_stats = json.loads(open("./results/threshold_stats.json", "r").read())
                thresholds = threshold_stats[metric] 
                # calibration_metrics_raw, calibration_metrics_json = get_calibration_metrics_raw(predicted_correctness_dict=avg_probs_dict,
                # actual_correctness_dict=model_metric_scores_dict,
                # thresholds=thresholds,
                # model_filename=model_filename,
                # length_ranges=summary_length_filters)
                # raw_calib_entries = calibration_metrics_raw["optimal"]
                # for entry in raw_calib_entries:
                #     entry['token_position'] = token_position_cutoff
                # raw_calib_by_position.append(raw_calib_entries)
                
                calibration_metrics_scaled, calibration_metrics_json = get_scaled_metrics_repo_split_kfold(predicted_scores=probs_list_dict,
                actual_scores=model_metric_scores_dict,
                thresholds=thresholds,
                test_data=f"./data/{language}/prompting_data/test.jsonl",
                num_bins = 10,
                model_filename=model_filename,
                length_ranges=summary_length_filters,
                token_position_cutoff=token_position_cutoff,
                filter_by_token_cutoff=False,
                mean_cutoff=True)
                scaled_calib_entries = calibration_metrics_scaled["optimal"]
                for entry in scaled_calib_entries:
                    entry['token_position'] = token_position_cutoff
                scaled_calib_by_position.append(scaled_calib_entries)
                scaled_evaluations[language][model_filename] = calibration_metrics_json
    # print(scaled_evaluations)
    # with open(f"./results/scaled_evaluations_optimal_token_cutoff_geometric_mean_with_cutoff.json", "w") as f:
        # json.dump(scaled_evaluations, f, indent=4)

            # Convert raw_calib_by_position and scaled_calib_by_position to DataFrame
            raw_calib_df = pd.DataFrame([entry for sublist in raw_calib_by_position for entry in sublist])
            scaled_calib_df = pd.DataFrame([entry for sublist in scaled_calib_by_position for entry in sublist])
            # Save DataFrames as CSV
            # results/Python_CodeLlama-70b-hf_BM25_BERT Score-R_position_vs_raw_calib.csv
            raw_calib_df.to_csv(f'./results/{language}_{model_name}_{prompting_method}_{metric}_position_vs_raw_calib_geometric_mean.csv', index=False)
            scaled_calib_df.to_csv(f'./results/{language}_{model_name}_{prompting_method}_{metric}_optimal_position_vs_scaled_calib_by_all_geometric_mean.csv', index=False)
