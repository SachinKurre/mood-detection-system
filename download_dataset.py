from datasets import load_dataset
import os

# Download dataset
dataset = load_dataset("dair-ai/emotion")

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Save the three parts
dataset["train"].to_parquet("dataset/train.parquet")
dataset["validation"].to_parquet("dataset/validation.parquet")
dataset["test"].to_parquet("dataset/test.parquet")

print("Dataset downloaded and saved successfully!")

print("\nDataset information:")
print(dataset)

print("\nFirst training example:")
print(dataset["train"][0])
