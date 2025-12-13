import math

hypo = 0
adg = 0
degrees = float(input("Enter a value for degrees: "))
degrees = degrees * 3.1415/180
opposite = float(input("Enter a value for opposite: "))
hypo = opposite / math.sin(degrees)
print(hypo)