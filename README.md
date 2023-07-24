# Jaithon
***We suffer so that you don't***
This is a simple compiler/interpreter that supports array declarations and operations. It can parse and execute code containing array declarations, variable assignments, print statements, import statements, input statements, time functions, and much more. The objective behind this interpreter is to provide a fun and easy way to get introduced to programming.

### Description

**JAITHON**: *An Age-Friendly Introduction to Coding*

JAITHON is a programming language carefully put together for teaching children and adults the fundamentals of coding. Its simplicity and intuitive syntax make it an ideal choice for young learners, allowing them to grasp programming concepts easily and enjoyably. The JAI programming Language is perfect for anyone regardless of age. By providing a beginner-friendly environment, The JAITHON Language removes the tech-savvy knowledge needed with traditional programming languages, making it accessible to people with little to no prior coding experience. The intended use of my language is to help the transition to Python. JAITHON will help you get from complete beginner, to having somewhat knowledge of python syntax.

**Crossing the age barrier**

The primary problem JAITHON addresses is the apprehension often felt by young learners when approaching programming. By offering straightforward and visually appealing syntax, JAITHON enables children to express their creativity through code without feeling overwhelmed. This approach fosters a positive learning experience, encouraging children to explore and develop problem-solving skills, logical thinking, and computational creativity from an early age. Our syntax is structured in such a way, that anyone who reads it will be able to tell you what the program does.

## Features

- Variable Assignment: You can assign values to variables using the var keyword. Syntax: *var x = 5*
- Array Declaration: You can init arrays using the array keyword. Syntax: *array arr = [1]*
- Array Element Modification: You can modify elements of an array using the dot notation. Syntax: *arr.add(0, 4).*
- Print Statement: You can use the print keyword to print values to the console. Syntax: *print x*
- Mathematical Expressions: You can perform arithmetic operations (+, -, *, /) on numbers.
- Comparisons: you can preform comparisons by using the > and < symbols. Syntax: *var a = 1>4* >> the output would be a = 0 because that statement is false.
- Parentheses: You can use parentheses to group expressions and control operator precedence. *PEMDAS*
- Trigonometric Functions: You can use trigonometric functions such as sin, cos, tan, asin, acos, atan, sqrt. Syntax: *sin(3)* *sin(x)*
- Detecting User input: You can detect user input. Syntax: *input variable_name*
- Time: You can access the users internal Clock by using the time keyword. Syntax: *var t = time()* This saves the time as an integer.
- Comments: You can add comments using the # symbol. Commented lines are displayed in green.
- If loops: You can do if loops by using the if/then/do keywords. Syntax: *if 1>4 then do print 1*
<br>

**Note:** Strings are not supported in any part of this language yet, including the print statement. 
<br>
<br>
**You can find more documentation at** 
```
Documentation/grammer.txt
```
<br>

*yes you cant do a helloworld in this language*


# Flowchart: JAI Programming Language Interpreter

1. **Read Input Code**: Read the code to be compiled/interpreted.
2. **Initialize Token List**: Convert the input code into tokens.
3. **Initialize Variables and Arrays**: Initialize storage for values.
4. **Start Parsing Program**: Begin parsing the program.
5. **Are Tokens Remaining Loop:**: Loop while tokens exist, otherwise GOTO 22.
6. **If Array Declaration**: Parse and store array details.
7. **Parse Array Declaration**: Read array size and elements.
8. **Identifier Check**: Handle variable assignment and print statements.
9. **If Array**: Parse array access or variable assignment.
10. **Parse Variable Assignment**: Assign variable a value.
11. **If Print Statement**: Parse and display variable/expression value.
12. **Parse Print Statement**: Evaluate and print.
13. **If Input Statement**: Parse and read user input.
14. **Parse Input Statement**: Read and store user's input value.
15. **If Trigonometric Function**: Parse and compute trigonometric value.
16. **Parse Trigonometric Function**: Calculate the trigonometric value.
17. **If Time Function**: Parse and compute time-related value.
18. **Parse Time Function**: Calculate the time-related value.
19. **End of loop**: Continue processing remaining tokens. GOTO 5 (Loop back).
20. **All Tokens Processed**: Execution complete, proceed to the end.
21. **Error Handling**: Handle parsing errors or undefined variables/arrays.
22. **Stop**: End the interpreter's execution.

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

This is a really cool feature and takes like 3 seconds to do so don't be lazy and follow the steps.
*Only if you have VS-Code as your IDE*
```sh
cd vscodeext
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
# Output: [0.000000, 0.000000, 0.000000] as there are 3 spots in memory initialized to 0.
a.add(0, 2)
print a
# Output: [2.000000, 0.000000, 0.000000] as the 0th term is set to 2
var x = 5
print x
# Output: 5.000000
```
## Info
For more info *and to appreaciate my hard webdev work* go to the html page located at 
```
https://abhiramasonny.github.io/Projects/JAITHON%20website/index.html
```
or the youtube video
```
https://www.youtube.com/watch?v=4CcsI82OGq0
```
or contact me at abhirama.sonny@gmail.com.