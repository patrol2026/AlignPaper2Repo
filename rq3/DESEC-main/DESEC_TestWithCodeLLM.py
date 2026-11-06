import logging
import os
import re
from pathlib import Path

import joblib
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from DESEC_BeamSearch import beam_search


modelname = "Your_Code_LLM"
model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model.eval()

# 'secret_name': [Prefix(If not, it is empty), Secret's text length, Token Constraint]
regex_dict = {
    'google_api_key': [r"AIza", 35, r"[0-9A-Za-z\-_]"],
    'stripe_test': [r"sk_test_", 24, r"[0-9a-zA-Z]"],
    'tencent_cloud_secret_id': [r"AKID", 32, r"[0-9a-zA-Z]"],
    'google_oauth_client_id': [r"", 72, r"[a-z0-9\-\.]"],
    'alibabacloud': [r"LTAI", 20, r"[a-zA-Z0-9]"],
    'slack_incoming_web_hook_url': [r"https://hooks.slack.com/services/", 46, r"[A-Za-z0-9+\/]"]
}


def process_files_in_target_subfolder(dataset_folder, result_folder, model, tokenizer, token_num, pattern, lda_model):
    for subdir, dirs, files in os.walk(dataset_folder):
        for filename in files:
            # Each prompt is written in a txt file
            if filename.endswith('.txt'):
                file_path = os.path.join(subdir, filename)
                name_without_extension, extension = os.path.splitext(filename)
                target_subdir = subdir.replace(dataset_folder, result_folder)
                os.makedirs(target_subdir, exist_ok=True)

                try:
                    outputfile = f'{target_subdir}/{name_without_extension}_result.txt'
                    if os.path.exists(outputfile):
                        continue
                    with open(file_path, 'r', encoding='utf-8') as file:
                        prompt = file.read()
                    # Probabilities, Entropies, Average Probabilities, Entropy Ratios, Text, Probability Advantages, Score, Decoded text token sequence
                    prob_seqs, entropies, prob_avgs, entropy_percent, out_text, prob_distance, self_score, de_tokens = beam_search(
                        prompt, model, tokenizer, pattern, lda_model, num_beams=5, max_length=token_num, max_text_length=token_num,
                        num_candidates=10)
                    # Output Text
                    output = prompt + out_text

                    with open(outputfile, "w", encoding="utf-8") as file:
                        file.write(output)
                        # You can output detailed content here, like prob_seqs, entropies, prob_avgs, entropy_percent, out_text, prob_distance, self_score, de_tokens
                    print("Output written to output.txt")

                except Exception as e:
                    logging.error(f"Code generation failed: {file_path},{e}")
                    continue


if __name__ == '__main__':

    current_dir = Path(__file__).resolve().parent

    for key, value in regex_dict.items():
        # Setting up file locations
        dataset_folder = f"Your_Prompts_Folder/prompts_{key}"
        write_folder = f"Your_Result_Folder/DESEC_{key}"
        ldamodel_path = f"{current_dir.parent}/LDA_Models/LDA_{key}.pkl"

        if not os.path.exists(write_folder):
            os.makedirs(write_folder)

        logging.basicConfig(filename=f'{write_folder}/error.log', level=logging.INFO,
                            filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        pattern = re.compile(value[2])
        lda_model = joblib.load(ldamodel_path)
        process_files_in_target_subfolder(dataset_folder, write_folder, model, tokenizer, value[1], pattern, lda_model)




