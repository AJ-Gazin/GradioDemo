import pandas as pd
from tqdm import tqdm
import concurrent.futures
from langchain_community.llms import Ollama

# Initialize the model
model = Ollama(model="llama3")

# Load the data
df = pd.read_csv("./data/amazon_reviews.csv")

# Dictionary to map switched category-subcategory pairs
category_mapping = {}

# Function to predict the category and subcategory
def predict_category(review_text):
    prompt = (
        "Task: Read the following product review and try to determine the category of the product. "
        "Possible categories include electronics, pet food, clothing, home goods, etc. "
        "Focus on identifying words or phrases that give clues about the type of product. "
        "Write your predicted category after writing The Category is:, and any subcategories after writing The Subcategory is:. "
        "Keep both categories and subcategories to 1 word.\n\nReview Text: " + review_text
    )
    response = model.invoke(prompt)
    category, subcategory = "Unknown", "Unknown"  # Default values
    lines = response.split('\n')
    for line in lines:
        if line.startswith("The Category is:"):
            category = line.replace("The Category is:", "").strip()
        elif line.startswith("The Subcategory is:"):
            subcategory = line.replace("The Subcategory is:", "").strip()
    return category, subcategory

# Function to process a chunk of the DataFrame
def process_chunk(df_chunk):
    for index, row in tqdm(df_chunk.iterrows(), total=len(df_chunk)):
        combined_text = row['Summary'] + " " + row['Text']
        category, subcategory = predict_category(combined_text)
        
        # Check and update category mapping
        pair = (category, subcategory)
        if pair not in category_mapping:
            category_mapping[pair] = pair
        else:
            category, subcategory = category_mapping[pair]
        
        df_chunk.loc[index, ['category', 'subcategory']] = category, subcategory
    return df_chunk

# Function to process chunks in parallel
def process_chunks_in_parallel(df, chunk_size):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), chunk_size):
            df_chunk = df.iloc[i:i+chunk_size]
            futures.append(executor.submit(process_chunk, df_chunk))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result.to_csv('./data/organized_reviews.csv', mode='a', header=not i, index=False)

# Set chunk size
chunk_size = 100

# Process chunks in parallel
process_chunks_in_parallel(df, chunk_size)
