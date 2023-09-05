var x = 10
var h = 0.0001

var fx = x ^ 2
var fxplush = x + h
var fxplush = fxplush ^ 2
var derivative = (fxplush - fx)
var derivative = derivative / h
print derivative