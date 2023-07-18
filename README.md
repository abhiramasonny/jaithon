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

**Note:** Strings arnt supported in any part of this language yet, including the print statement. 
*yes you cant do a helloworld in this language*


## How it Works

The compiler/interpreter follows the following workflow:


1. **Read Input Code**: Read the code to be compiled/interpreted.
2. **Initialize Token List**: Initialize the list of tokens by converting the input code into individual tokens.
3. **Initialize Variables and Arrays**: Set up variables and arrays to store values during execution.
4. **Start Parsing Program**: Begin parsing the program by checking if there are remaining tokens.
5. **Are Tokens Remaining?**: If there are remaining tokens, proceed to the next step. Otherwise, go to step 14.
6. **Current Token is an Array Declaration**: If the current token is an array declaration, parse it and store the array's name and size.
7. **Parse Array Declaration**: Parse the array declaration statement.
8. **Current Token is an Identifier**: If the current token is an identifier, check if it corresponds to an array declaration.
9. **Check Identifier is an Array**: If the identifier is an array, proceed to parse the statement related to array access or modification. Otherwise, go to step 10
10. **Parse Statement**: Parse the variable assignment or print statement.
11. **Loop through Tokens**: Continue looping through the remaining tokens in the program.
12. **All Tokens Processed**: If all tokens have been processed, go to step 14.
13. **Parse Variable Assignment**: If the identifier is not an array, parse the variable assignment statement.
14. **Stop**: End the execution of the compiler/interpreter.
![FlowChart](Flowchart.png)
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
