import json
import re
# Load the JSON data
with open('./haystack-results/llama-v3p1-405b-instruct_results-2.json', 'r') as file:
    data = json.load(file)

# Process each element
for element in data:
    response = element['model_response'].lower()
    points = 0
    purple_match = re.search(r'\bpurple\b', response)
    white_match = re.search(r'\bwhite\b', response)
    
    if purple_match and white_match:
        points = 2
    elif purple_match or white_match:
        points = 1
    
    element['points'] = points
    
    # Add the points field to the element
    element['points'] = points
    
    # Remove the context_with_needle field
    if 'context_with_needle' in element:
        del element['context_with_needle']

# Save the updated JSON data
with open('processed-result.json', 'w') as file:
    json.dump(data, file, indent=2)

print("Processing complete. Points have been added and context_with_needle field has been removed from each element in temp.json.")
