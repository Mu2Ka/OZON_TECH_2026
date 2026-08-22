flag=0
for i in range(10000000,100000000):
    dig_lis = list(map(int, str(i)))
    if sum(dig_lis) == 3375:
        flag+=1
    print(flag)
print(flag)