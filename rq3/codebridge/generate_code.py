from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from text_dataset import TextDataset
import re
from tqdm import tqdm
import os
import json
def extract_code(generation, lang):
    lang = lang.lower()
    generation = generation.replace(f"[{lang.upper()}]", f'```{lang}').replace(f"[/{lang.upper()}]", '```')

    if f'```{lang}' in generation:
        r_str = f"```{lang}\n(.*?)\n```"
        code = re.compile(r_str, flags=re.DOTALL)
        code_block = code.findall(generation)
        ret =  code_block[0] if len(code_block) >= 1 else generation.split(f'```{lang}')[-1]
        return ret.strip()
    elif '```' in generation:
        r_str = f"```\n(.*?)\n```"
        code = re.compile(r_str, flags=re.DOTALL)

        code_block = code.findall(generation)
        ret = code_block[0] if len(code_block) >= 1 else generation.split(f'```')[-1]
        return ret.strip()
    else:
        return generation



class DeepSeekModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    def generate(self, content, max_new_tokens=256):
        messages=[
            { 'role': 'user', 'content': content}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def generate_code(model, dataset, args,  max_new_tokens=256):
    # check dir
    if not os.path.exists(args.output_path):
        print(f"creating dir: {args.output_path}")
        os.makedirs(args.output_path)

    # check 
    prompt = """Write a code for the following query in {language} without comments. You must return a code and must not refuse to answer.
{query}
"""
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    final_ans = []
    for i, item in tqdm(enumerate(dataset)):
        query = item['nl_input']
        prompt_text = prompt.format(language=language, query=query)
        gencode = model.generate(prompt_text, max_new_tokens=max_new_tokens)
        final_ans.append(
            {   'code_input': extract_code(gencode, language), 
                'nl_input': query,
                'gt': item['code_input'],
                'url': item['url']
             }
        )
       

    filename = os.path.join(args.output_path, f'{args.lang}_test_gen_code.jsonl')
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(final_ans, file, indent=4)

    print(f"Data has been written to {filename}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="samples", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--query_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--output_path",default=None, type=str, help="output path")
    parser.add_argument("--is_dev", action='store_true')
    parser.add_argument("--lang", default="python", type=str, help="language")

    args = parser.parse_args()
    print(args)
    query_dataset = TextDataset(args,None,"text",args.query_data_file)
    if 'deepseek' in args.model_path:
        gen_model = DeepSeekModel(args.model_path)

    generate_code(gen_model, query_dataset, args, max_new_tokens=256)


