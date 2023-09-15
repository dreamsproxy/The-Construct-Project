import pandas as pd
import random

class DataProcessor:
    def __init__(self):
        self.data = pd.DataFrame()

    def generate_and_add_to_dataframe(self, num_iterations=5):
        for _ in range(num_iterations):
            # Generate 12 random numbers
            random_numbers = [random.randint(1, 100) for _ in range(12)]
            print(random_numbers)
            # Create a new DataFrame with 12 columns and add the random numbers
            df = pd.DataFrame([random_numbers], columns=[f'{i+1}' for i in range(12)])
            
            # Concatenate the new DataFrame to the existing DataFrame
            self.data = pd.concat([self.data, df], ignore_index=True)

# Create an instance of the DataProcessor class
data_processor = DataProcessor()

# Call the generate_and_add_to_dataframe function in a loop of 5
for _ in range(5):
    data_processor.generate_and_add_to_dataframe()

# Print the resulting DataFrame
print(data_processor.data)
