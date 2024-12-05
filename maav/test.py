# Define a function in the main program
def greet(name):
    print("Hello,", name)

# Define a variable in the main program
message = "Welcome to the program!"

# Define a Python code as a string
code_str = """
# Access the variable defined in the main program
print(message)

# Call the function defined in the main program
greet("Alice")
"""

# Execute the code string
exec(code_str)
