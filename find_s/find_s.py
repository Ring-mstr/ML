import csv

# Load data from CSV file
with open("./tennis_data.csv","r",) as f:
    reader = csv.reader(f)
    data = list(reader)

# Initialize hypothesis with most specific values
hypothesis = [["0", "0", "0", "0", "0", "0"]]

# Iterate through each instance in the dataset
for instance in data:
    print(instance)
    # If the instance is positive (the last attribute is "True")
    if instance[-1] == "True":
        j = 0
        # Update hypothesis based on the current instance
        for x in instance:
            if x != "True":
                if x != hypothesis[0][j] and hypothesis[0][j] == "0":
                    hypothesis[0][j] = x
                elif x != hypothesis[0][j] and hypothesis[0][j] != "0":
                    hypothesis[0][j] = "?"
                else:
                    pass
                j = j + 1

# Print the most specific hypothesis
print("Most specific hypothesis is:")
print(hypothesis)
