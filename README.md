# Jaithon

This is a simple compiler/interpreter that supports array declarations and operations. It can parse and execute code containing array declarations, variable assignments, and print statements.

## Features

- Variable Assignment: You can assign values to variables using the var keyword. Syntax: *var x = 5*
- Array Declaration: You can init arrays using the array keyword. Syntax: *array arr = [1]*
- Array Element Modification: You can modify elements of an array using the dot notation. Syntax: *arr.add(0, 4).*
- Print Statement: You can use the print keyword to print values to the console. Syntax: *print x*
- Mathematical Expressions: You can perform arithmetic operations (+, -, *, /) on numbers.
- Comparrisions: you can preform comparrisions by using the > and < symbols. Syntax: *var a = 1>4* >> the output would be a = 0 because that statement is false.
- Parentheses: You can use parentheses to group expressions and control operator precedence. *PEMDAS*
- Trigonometric Functions: You can use trigonometric functions such as sin, cos, tan, asin, acos, atan, sqrt. Syntax: *sin(3)* *sin(x)*
- Detecting User input: You can detect user input. Syntax: *input variable_name*
- Time: You can access the users internal Clock by using the time keyword. Syntax: *var t = time()* This saves the time as an integer.
- Comments: You can add comments using the # symbol. Commented lines are displayed in green.
<br>

**Note:** Strings are not supported in any part of this language yet, including the print statement. 
<br>

*yes you cant do a helloworld in this language*


# Flowchart: JAI Programming Language Interpreter

1. **Read Input Code**: Read the code to be compiled/interpreted.
2. **Initialize Token List**: Initialize the list of tokens by converting the input code into individual tokens.
3. **Initialize Variables and Arrays**: Set up variables and arrays to store values during execution.
4. **Start Parsing Program**: Begin parsing the program by checking if there are remaining tokens.
5. **Are Tokens Remaining?**: If there are remaining tokens, proceed to the next step. Otherwise, go to step 14.
    6. **Current Token is an Array Declaration**: If the current token is an array declaration, parse it and store the array's name and size.
        7. **Parse Array Declaration**: Parse the array declaration statement.
    8. **Current Token is an Identifier**: If the current token is an identifier, check if it corresponds to an array declaration or a variable assignment.
        9. **Check Identifier is an Array**: If the identifier is an array, proceed to parse the statement related to array access or modification. Otherwise, go to step 10.
            10. **Parse Variable Assignment**: Parse the variable assignment statement.
    11. **Current Token is a Print Statement**: If the current token is a print statement, parse it and print the corresponding value.
        12. **Parse Print Statement**: Parse the print statement and print the variable or expression value.
    13. **Current Token is an Input Statement**: If the current token is an input statement, parse it and read user input.
        14. **Parse Input Statement**: Parse the input statement and read the user's input value.
    15. **Current Token is a Trigonometric Function**: If the current token is a trigonometric function, parse it and compute the corresponding value.
        16. **Parse Trigonometric Function**: Parse the trigonometric function and compute its value.
    17. **Current Token is a Time Function**: If the current token is a time function, parse it and compute the corresponding value.
        18. **Parse Time Function**: Parse the time function and compute its value.
19. **Loop through Tokens**: Continue looping through the remaining tokens in the program.
20. **All Tokens Processed**: If all tokens have been processed, go to step 22.
21. **Error Handling**: Handle errors during parsing, such as encountering invalid tokens or accessing undefined variables/arrays.
22. **Stop**: End the execution of the compiler/interpreter.

## Usage

To use the array compiler/interpreter, follow these steps:
```bash 
git clone https://github.com/abhiramasonny/jaithon
cd jaithon
gcc -o interpreter interpreter.c 
./interpreter
```
Then type this as follow up to *Enter file name to interpret*
```
jaithon
```

4. OPTIONAL: Write the code to be compiled/interpreted, including array declarations, variable assignments, and print statements to the file called jaithon.jai.
the jaithon.jai file is the file that the interpreter reads from. It by default has test cases.
5. Execute the compiler/interpreter with *filename* **in this case jaithon** as input.
6. The output will be displayed, showing the results of any print statements or errors encountered during execution.


## Syntex Highlighting

This is a really cool feature and takes like 3 seconds to do so dont be lazy and follow the steps.
*Only if you have VS-Code as your IDE*
```sh
cd tmp
cp -r jai ~/.vscode/extensions
```
**Reload VS-Code**
Note: this is for mac/linux.
<br>
dont own a windows laptop so havent tested on there.
## Examples

Here are some examples of code that can be compiled/interpreted using this interpreter:

```python
array a = [3]
print a
# Output: [0.000000, 0.000000, 0.000000] as there are 3 spots in memory initilized to 0.
a.add(0, 2)
print a
# Output: [2.000000, 0.000000, 0.000000] as the 0th term is set to 2
var x = 5
print x
# Output: 5.000000
```
