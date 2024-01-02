import random 
# A list of 5 of my favourite fruits
fruit_list = ["Apple", "Banana", "Mango", "Strawberry", "Pineapple"]
# Generate a random choice of fruit from the list of fruits above
random_fruit = random.choice(fruit_list)
# Ask the user to enter a single letter
guess_first_letter = input("Please enter a single letter: ")

# Check if the input is a single letter and alphabetical
if len(guess_first_letter) == 1 and guess_first_letter.isalpha():
    print("Good Guess!:", guess_first_letter)
else:
    print("Oops! That is not a vaild input.")

# Display the entered letter
print("You entered:", guess_first_letter)
# Display randomly selected fruit
print("Randomly selected fruit:", random_fruit)
