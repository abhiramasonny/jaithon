var mul = "*"
var div = "/"
var add = "+"
var minus = "-"
input num1
input opperator, string
input num2
var nmul = num1 * num2
var nadd = num1 + num2
var nsub = num1 - num2
var ndiv = num1 / num2
if compare opperator, mul then do print nmul
if compare opperator, div then do print ndiv
if compare opperator, minus then do print nsub
if compare opperator, add then do print nadd