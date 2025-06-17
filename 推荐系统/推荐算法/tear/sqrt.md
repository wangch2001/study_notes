开根号(判断精确度的时候记得绝对值)
1.二分法
```py
def sqrt(n,a):
    if (n<0) return -1
    if (n==0) return 0
    low = 0*1.0
    up = n*1.0
    mid =(low+up)/2.0
    while(low<up):
        if abs(mid*mid-n)<a:
            return mid
        else mid*mid>n:
            up=mid
        else:
            low=mid
        mid=(low+ip)/2.0
    return -1
```
2.牛顿迭代法
```py
def sqrt(n,a):
    if (n<0) return -1
    if (n==0) return 0
    x=n
    while abs(x**2-n)>a:
        x=(x+n/x)/2
    return x
```