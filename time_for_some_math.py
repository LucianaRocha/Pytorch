x=1
y=1
b=-10
p1=3
p2=4
line = (p1*x)+(p2*y)+b
count = 0

print(line)

while line < 0:
    p1 = p1 + (0.1 * x)
    p2 = p2 + (0.1 * y)
    b = b + 0.1
    line = (p1*x)+(p2*y)+b
    line = round(line,2)
    count += 1
    print(count,p1,p2,b,line)

