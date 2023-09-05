import random

total_points = 0
circle_points = 0
i = 0
while i<10000000000:
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    
    total_points += 1
    
    distance = x**2 + y**2
    if distance <= 1:
        circle_points += 1

    pi_estimate = 4 * circle_points / total_points
    
    print(f"{pi_estimate}, with {total_points}")
    i+=1