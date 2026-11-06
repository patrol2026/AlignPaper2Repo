import pandas as pd
import numpy as np
from ProbabilisticTokenPositioning import ProbabilisticTokenPositioning
from PivotWordIdentification import PivotWordIdentification
from LLMBasedFeedback import LLMBasedFeedback
from BestExample import BestExample
# ContextCraft class 
class ContextCraft:
    def __init__(self, train_txt_path, test_txt_path, api_client):
        # Load the text files into DataFrames
        self.train_df = self.load_txt_file(train_txt_path)
        self.test_df = self.load_txt_file(test_txt_path)
        
        # Initialize components
        self.prob_token_positioning = ProbabilisticTokenPositioning()
        self.pivot_word_id = PivotWordIdentification()
        self.llm_feedback = api_client
        self.best_example = BestExample(self.train_df)
        self.probability_df = None

    def load_txt_file(self, txt_file_path):
        """Load a .txt file and create a DataFrame with 'Functional Description' and 'Method Name'."""
        data = []
        with open(txt_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('_')
                if len(parts) == 2:
                    data.append({'Functional Description': parts[0], 'Method Name': parts[1]})
        return pd.DataFrame(data)
    
    def process_train_file(self):
        """Process the training text file using various components."""
        # Calculate position probabilities
        position_probabilities = self.prob_token_positioning.calculate_position_probabilities(self.train_df)
        self.probability_df = self.prob_token_positioning.create_probability_df(position_probabilities)

        # Add a new column for the example description
        self.train_df['Example Description'] = ''

        # Process each row in the DataFrame
        for index, row in self.train_df.iterrows():
            func_desc = row['Functional Description']
            method_name = row['Method Name']

            # Get prefix, infix, and suffix probabilities
            tokens = self.prob_token_positioning.extract_tokens(func_desc)
            prefix, infix, suffix = self.get_token_positions(tokens)

            # Get pivot words
            pivot_words = self.pivot_word_id.find_best_description_tokens(method_name, func_desc)
            pivot_words_str = self.format_pivot_words(pivot_words)

            # Get LLM feedback
            predicted_name = self.llm_feedback.predict_method_name(func_desc)
            feedback = self.llm_feedback.generate_feedback(predicted_name, method_name)

            # Format example description
            example_desc = self.format_example_description(
                func_desc, method_name, prefix, infix, suffix, pivot_words_str, feedback
            )
            self.train_df.at[index, 'Example Description'] = example_desc

        # Save processed DataFrame to a new CSV file
        self.train_df.to_csv('processed_train_data.csv', index=False)

    def get_token_positions(self, tokens):
        """Determine the most likely prefix, infix, and suffix for tokens."""
        prefix = None
        infix = None
        suffix = None
        max_prefix_prob = max_infix_prob = max_suffix_prob = 0

        for token in tokens:
            if token in self.probability_df['Token'].values:
                row = self.probability_df[self.probability_df['Token'] == token]
                prefix_prob = row['Prefix Probability'].values[0]
                infix_prob = row['Infix Probability'].values[0]
                suffix_prob = row['Suffix Probability'].values[0]

                if prefix_prob > max_prefix_prob:
                    max_prefix_prob = prefix_prob
                    prefix = token
                if infix_prob > max_infix_prob:
                    max_infix_prob = infix_prob
                    infix = token
                if suffix_prob > max_suffix_prob:
                    max_suffix_prob = suffix_prob
                    suffix = token

        return (prefix, max_prefix_prob), (infix, max_infix_prob), (suffix, max_suffix_prob)

    def format_pivot_words(self, pivot_words):
        """Format pivot words into a string for the example description."""
        return ", ".join(
            [f"'{k}': ('{v[0]}', {v[1] * 100:.2f}%)" for k, v in pivot_words.items()]
        )

    def format_example_description(self, func_desc, method_name, prefix, infix, suffix, pivot_words, feedback):
        """Format the example description according to the specified style."""
        prefix_str = f"Prefix: ('{prefix[0]}', Probability: {prefix[1] * 100:.2f}%)" if prefix[0] else ""
        infix_str = f"Infix: ('{infix[0]}', Probability: {infix[1] * 100:.2f}%)" if infix[0] else ""
        suffix_str = f"Suffix: ('{suffix[0]}', Probability: {suffix[1] * 100:.2f}%)" if suffix[0] else ""
        context = f"In this example, the most likely words are: {prefix_str} {infix_str} {suffix_str}."
        return (
            f"Functional Description: \"{func_desc}\"\n"
            f"Method Name: '{method_name}'\n"
            f"Context:\n"
            f"• {context}\n"
            f"• The semantic similarity between tokens of description and method name: {pivot_words}.\n"
            f"• {feedback}"
        )

    def process_test_file(self):
        """Process each functional description in the test file and find similar examples."""
        # Initialize results DataFrame
        results = self.test_df.copy()
        results['Example Description'] = ''

        for index, row in self.test_df.iterrows():
            input_description = row['Functional Description']
            similar_examples = self.find_similar_examples(input_description)
            
            # Combine similar example descriptions into a single string
            example_descs = "\n".join(similar_examples['Example Description'].tolist())
            results.at[index, 'Example Description'] = example_descs

        # Save processed test results to a new CSV file
        results.to_csv('processed_test_data.csv', index=False)

    def find_similar_examples(self, input_description):
        """Find the top 10 similar functional descriptions."""
        top_similar = self.best_example.find_top_n_similar_descriptions(input_description, n=10)

        # Initialize results DataFrame
        results = top_similar.copy()
        results['Example Description'] = ''

        for index, row in results.iterrows():
            func_desc = row['Functional Description']
            method_name = row['Method Name']

            # Get prefix, infix, and suffix probabilities
            tokens = self.prob_token_positioning.extract_tokens(func_desc)
            prefix, infix, suffix = self.get_token_positions(tokens)

            # Get pivot words
            pivot_words = self.pivot_word_id.find_best_description_tokens(method_name, func_desc)
            pivot_words_str = self.format_pivot_words(pivot_words)

            # Get LLM feedback
            predicted_name = self.llm_feedback.predict_method_name(func_desc)
            feedback = self.llm_feedback.generate_feedback(predicted_name, method_name)

            # Format example description
            example_desc = self.format_example_description(
                func_desc, method_name, prefix, infix, suffix, pivot_words_str, feedback
            )
            results.at[index, 'Example Description'] = example_desc

        return results

# Example Usage
if __name__ == "__main__":
    # Paths to the training and test text files
    train_txt_path = 'Dataset/java_train.txt'  # Path to training file
    test_txt_path = 'Dataset/java_test.txt'  # Path to test file

    # Create an instance of LLMAPIClient with your OpenAI API key
    api_key = 'your_openai_api_key_here'  # Replace it with your OpenAI API key
    api_client = LLMAPIClient(api_key)

    # Create an instance of ContextCraft with the text files and the LLM API client
    context_craft = ContextCraft(train_txt_path, test_txt_path, api_client)

    # Process the training text file to generate and save results
    context_craft.process_train_file()

    # Process the test text file to generate and save results
    context_craft.process_test_file()
