import random

filename = "city_map.ussa3d"

# Read the file content
with open(filename, "r") as file:
    content = file.read()

# Find all indices of '2' in the content
indices_of_2 = [i for i, char in enumerate(content) if char == '2']

# Calculate 30% of them to replace
num_to_replace = int(len(indices_of_2) * 0.2)

# Randomly pick indices to replace
indices_to_replace = set(random.sample(indices_of_2, num_to_replace))

# Possible replacement digits
replacement_choices = ['5', '6', '7', '8']

# Build the new content replacing selected '2's with a random choice among 5,6,7,8
new_content = []
for i, char in enumerate(content):
    if i in indices_to_replace:
        new_content.append(random.choice(replacement_choices))
    else:
        new_content.append(char)

new_content = "".join(new_content)

# Write back to file
with open(filename, "w") as file:
    file.write(new_content)

print(f"Replaced {num_to_replace} occurrences of '2' with one of {replacement_choices} in {filename}.")
