import numpy as np
import matplotlib.pyplot as plt


x_train=np.array([1.0,2.0])
y_train=np.array([300,500])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

print(f"x_train.shape: {x_train.shape}")
m=x_train.shape[0]
print(f"numbers of training sampel: {m}")

print("numbers of training sampel: {}".format(len(x_train)))

i = 0 # Change this to 1 to see (x^1, y^1)
for i in range(len(x_train)):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

#Plot the data

plt.scatter(x_train, y_train, marker='x', color='r')
plt.title("housing prices")
plt.ylabel("price")
plt.xlabel("size")
plt.show()



w=100
b=100
print(f"w: {w}")
print(f"b: {b}")


def compute_model_output(x,w,b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb

tmp_f_wb=compute_model_output(x_train,w,b)

plt.plot(x_train,tmp_f_wb,c='b',label="Our prediction") 

plt.scatter(x_train, y_train, marker='x', color='r',label="Actual values")

plt.title("housing prices")
plt.ylabel("price")
plt.xlabel("size")
plt.legend()
plt.show()