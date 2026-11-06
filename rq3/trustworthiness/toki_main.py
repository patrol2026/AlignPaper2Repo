import argparse

from sklearn.metrics import classification_report

from src.file_loader import save_json
from src.relate_words import load_lemmatization

from src.toki import TOKI
from src.explainer import EmptyExplainer
from src.preload_explanation import preload_train_expl, preload_test_expl
from src.options_loader import get_options_from_file, get_options_huggingface
from src.data_methods import get_load_ensemble, get_data, get_model

all_dataset = {
    "issues": {
        "model": "models/nlbse",
        "dataset": "issues"
    },

    "movies": {
        "model": "mlp",
        "dataset": "rationales"
    },

    "20news": {
        "model": "mlp",
        "dataset": "20newsgroup"
    },

    "hateXplain": {
        "model": "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
    },

    "mental": {
        "model": "Tianlin668/CAMS"
    },

    "amazon_polarity": {
        "model": "pig4431/amazonPolarity_roBERTa_5E",
        "dataset": "fancyzhx/amazon_polarity",
    },

    "ag_news": {
        "model": "mrm8488/bert-mini-finetuned-age_news-classification",
        "dataset": "fancyzhx/ag_news",
    },

    "rotten_tomatoes": {
        "model": "xianzhew/distilbert-base-uncased_rotten_tomatoes",
        "dataset": "cornell-movie-review-data/rotten_tomatoes",
    },

    "imdb": {
        "model": "lvwerra/distilbert-imdb",
        "dataset": "stanfordnlp/imdb",
    },

    "emotion": {
        "model": "sabre-code/distilbert-base-uncased-finetuned-emotion",
        "dataset": "dair-ai/emotion"
    },

    "yahoo_answers_topics": {
        "model": "fabriceyhc/bert-base-uncased-yahoo_answers_topics",
        "dataset": "community-datasets/yahoo_answers_topics"
    },
}

def eval_toki(dataset, model, t_dist, expl_method="lime", log_dir=None):
    load_lemmatization()

    print("Evaluating", dataset, "with model", model)

    we_models = ["fasttext", "glove", "USE", "nnlm", "bert", "google_news_swivel"]

    datasets_dong = ["rationales", "20newsgroup"]
    if dataset == "issues" or dataset in datasets_dong:
        options = get_options_from_file("config_toki.ini")
        options["data"] = dataset
        options["model"] = model
        train_explanations = preload_train_expl(dataset, model, expl_method)
        if dataset_name == datasets_dong[1]:
            we_models = we_models[:-1]
    else:
        options = get_options_huggingface(model, dataset)
        dataset = options["hf_data_name"]
        train_explanations = preload_train_expl(dataset, "huggingface", expl_method)

    ensemble = get_load_ensemble(we_models)

    options['threshold_cluster_dist'] = t_dist
    majority_scores = {
        "20newsgroup": 0.05,
        "rotten_tomatoes": 0.45,
        "yahoo_answers_topics": 0.2,
        "mental": 0.5,
        "hateXplain": 0.5,
    }
    if dataset in majority_scores:
        options['threshold_majority_score'] = majority_scores[dataset]
    else:
        options['threshold_majority_score'] = 0.36
    options["explanation_models"] = [expl_method]

    dataset_data = get_data(options)
    _, predictor = get_model(dataset_data, options)

    X_test, y_test, explanations_test, y_trust = preload_test_expl(dataset, expl_method)

    explainer = EmptyExplainer()

    toki = TOKI(predictor, dataset_data['category_names'], ensemble, explainer, options)
    toki.identify_keywords(dataset_data["X_train"], dataset_data["y_train"], train_explanations)
    result = toki.assess(X_test, y_test, explanations_test)

    for ins, y in zip(result, y_trust):
        ins["user_score"] = y
    y_trust_pred = [x['trustworthy'] for x in result]

    print(classification_report(y_trust, y_trust_pred, labels=[0, 1], digits=3))

    if log_dir is not None:
        save_json(result, f"{log_dir}/{dataset}-{t_dist}.json")

    return y_trust, y_trust_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help=f"The dataset under attack ({list(all_dataset)}). ")
    parser.add_argument("--tdist", default=0.3, type=int, required=False,
                        help="Threshold distance for clustering.")
    parser.add_argument("--log_dir", default=None, type=str, required=False, help="Log directory path.")
    parser.add_argument("--explanation", default="lime", type=str, required=False, help="Explanation method.")
    args = parser.parse_args()

    dataset_name = args.dataset

    if dataset_name not in all_dataset:
        raise Exception("Invalid dataset!")

    model_name = all_dataset[dataset_name]["model"]

    if dataset_name in ["movies", "20news"]:
        dataset_name = all_dataset[dataset_name]["dataset"]

    eval_toki(dataset_name, model_name, t_dist=args.tdist, log_dir=args.log_dir, expl_method=args.explanation)