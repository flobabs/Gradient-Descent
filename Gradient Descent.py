#!/usr/bin/env python
# coding: utf-8

# ## Notebook Imports and Packages

# In[31]:


import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols, diff
from math import log

get_ipython().run_line_magic('matplotlib', 'inline')


# # Example 1 - A simple cost function
# 
# $f(x) = x^2 + x + 1$

# In[2]:


def f(x):
    return x**2 + x + 1


# In[3]:


# Make Data
x_1 = np.linspace(start=-3, stop=3, num=500)


# In[4]:




plt.subplot(1, 2, 1)

plt.xlim([-3, 3])
plt.ylim(0, 8)
plt.xlabel('X', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.plot(x_1, f(x_1))

plt.show


# In[5]:


type(x_1)


# # Slope & Derivatives
# Challenge: Create a python function for the derivative of $f(x)$ and $df(x)$

# In[6]:


def df(x):
    return 2*x + 1


# In[7]:


# Plot function and derivative side by side

plt.figure(figsize=[15, 5])

# 1 Chart: Cost Function
plt.subplot(1, 2, 1)

plt.xlim([-3, 3])
plt.ylim(0, 8)
plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('f(x)', fontsize=16)

plt.plot(x_1, f(x_1), c='b', linewidth=3)

# 2 Chart: derivative
plt.subplot(1, 2, 2)

plt.xlim([-2, 3])
plt.ylim(-3, 6)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('df(x)', fontsize=16)
plt.grid()

plt.plot(x_1, df(x_1), c='r', linewidth=5)

plt.show


# ## Python Loops & Gradient Descent

# In[8]:


# python for loop
for n in range(5):
    print('Hello World', n)
print('End of Loop')


# In[9]:


# python while loop
counter = 0
while counter < 5:
    print('counting ...', counter)
    counter = counter + 1
print('Ready or not, here i come!')


# In[10]:


# Gradient Descent
new_x = 3
previous_x = 0
step_multiplier = 0.1

for n in range(30):
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - step_multiplier * gradient
    
print('Local minimum occurs at:', new_x)
print('Slope of df(x) value at this point is:', df(new_x))
print('f(x) value or cost at this point is:', f(new_x))


# In[11]:


# Gradient Descent
new_x = 3
previous_x = 0
step_multiplier = 0.1
precision = 0.0001

x_list = [new_x]
slope_list = [df(new_x)]

for n in range(500):
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - step_multiplier * gradient
    
    step_size = abs(new_x - previous_x)
    #print(step_size)
    
    x_list.append(new_x)
    slope_list.append(df(new_x))
    
    if step_size < precision:
        print('Loop ran this many times:', n)
        break
        
print('Local minimum occurs at:', new_x)
print('Slope of df(x) value at this point is:', df(new_x))
print('f(x) value or cost at this point is:', f(new_x))


# In[12]:


# Plot function and derivative side by side

plt.figure(figsize=[20, 5])

# 1 Chart: Cost Function
plt.subplot(1, 3, 1)

plt.xlim([-3, 3])
plt.ylim(0, 8)
plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('f(x)', fontsize=16)

plt.plot(x_1, f(x_1), c='b', linewidth=3, alpha=0.8)


values = np.array(x_list)
plt.scatter(x_list, f(values), color='r', s=100, alpha=0.6)

# 2 Chart: derivative
plt.subplot(1, 3, 2)

plt.xlim([-2, 3])
plt.ylim(-3, 6)


plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('df(x)', fontsize=16)
plt.grid()

plt.plot(x_1, df(x_1), c='r', linewidth=5, alpha=0.6)

plt.scatter(x_list, slope_list, color='b', s=100, alpha=0.6)

# 3 Chart: Derivative (Close up)
plt.subplot(1, 3, 3)

plt.xlim([-0.55, 0.2])
plt.ylim(-0.3, 0.8)


plt.title('Gradient Descent (close up)', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.grid()

plt.plot(x_1, df(x_1), c='r', linewidth=5, alpha=0.8)

plt.scatter(x_list, slope_list, color='b', s=100, alpha=0.6)

plt.show


# # Example 2 - Multiple Minima vs Initial Guess & Advanced Functions
# 
# ## $$g(x) = x^4 - 4x^2 +5$$
#  

# In[13]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


# Make some data

x_2 = np.linspace(-2, 2, 1000)

def g(x):
    return x**4 - 4*x**2 + 5

def dg(x):
    return 4*x**3 - 8*x


# In[15]:


# Plot function and derivative side by side

plt.figure(figsize=[20, 5])

# 1 Chart: Cost Function
plt.subplot(1, 3, 1)

plt.xlim([-2, 2])
plt.ylim(0.5, 5.5)
plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), c='b', linewidth=3, alpha=0.8)


values = np.array(x_list)

# 2 Chart: derivative
plt.subplot(1, 2, 2)

plt.xlim([-2, 2])
plt.ylim(-6, 8)


plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()

plt.plot(x_2, dg(x_2), c='r', linewidth=5, alpha=0.6)


# ## Gradient Decent as a Python Function

# In[18]:


local_min, list_x, deriv_list = gradient_descent(dg, 0.5, 0.02, 0.001)
print('Local min occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[19]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= -0.5, multiplier=0.01, precision=0.0001)
print('Local min occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[20]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= -0.1)
print('Local min occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[21]:


#calling gradient descent function

local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= -0)

# Plot  function and derivative and scatter plot side by side

plt.figure(figsize=[15, 5])

# 1 Chart: Cost Function
plt.subplot(1, 2, 1)

plt.xlim([-2, 2])
plt.ylim(0.5, 5.5)
plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), c='b', linewidth=3, alpha=0.8)
plt.scatter(list_x, g(np.array(list_x)), c='red', s=100, alpha=0.6)


# 2 Chart: derivative
plt.subplot(1, 2, 2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-6, 8)

plt.plot(x_2, dg(x_2), color='red', linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, c='indigo', s=100, alpha=0.5)
            
plt.show()


#  ## Example 3 - Divergence, Overflow and Python Tuples
#  ## $$ h(x) = x^5 - 2x^4 + 2$$

# In[22]:


# Make Data
x_3 = np.linspace(start=-2.5, stop=2.5, num=1000)

def h(x):
    return x**5 - 2*x**4 + 2

def dh(x):
    return 5*x**4 - 8*x**3


# In[23]:


#calling gradient descent function

local_min, list_x, deriv_list = gradient_descent(derivative_func=dh, initial_guess=-0.2, max_iter=70)

# Plot  function and derivative and scatter plot side by side

plt.figure(figsize=[15, 5])

# 1 Chart: Cost Function
plt.subplot(1, 2, 1)

plt.xlim([-1.2, 2.5])
plt.ylim(-1, 4)
plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('h(x)', fontsize=16)

plt.plot(x_2, h(x_2), c='b', linewidth=3, alpha=0.8)
plt.scatter(list_x, h(np.array(list_x)), c='red', s=100, alpha=0.6)


# 2 Chart: derivative
plt.subplot(1, 2, 2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dh(x)', fontsize=16)
plt.grid()
plt.xlim(-1, 2)
plt.ylim(-4, 5)

plt.plot(x_2, dh(x_2), color='red', linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, c='indigo', s=100, alpha=0.5)
            
plt.show()
print('Local min occurs at: ', local_min)
print('cost at this minimum is: ', h(local_min))
print('Number of Steps: ', len(list_x))


# In[ ]:


import sys
sys.version


# ## Python Tuple

# In[ ]:


# Creating a tuple

Nigeria = 'lagos', 'Ibadan', 'Ekiti'
unlucky_number = 13, 4, 9, 26, 17

print('I hate ', Nigeria[0])
print('Nigeria stresses me out specifically, ' + str(unlucky_number[1]) + 'lagos')
tuple_with_single_value = 50,
print(tuple_with_single_value)


# In[ ]:


type(tuple_with_single_value)


# In[ ]:


data_tuple = gradient_descent(derivative_func=dh, initial_guess=0.2)
print('Local min is', data_tuple[0])
print('Cost at the last x value is', h(data_tuple[0]))
print('Number of steps is', len(data_tuple[1]))


# ## The Learning Rate 

# In[25]:


#calling gradient descent function

local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= 1.9, multiplier=0.3, max_iter=5)

# Plot  function and derivative and scatter plot side by side

plt.figure(figsize=[15, 5])

# 1 Chart: Cost Function
plt.subplot(1, 2, 1)

plt.xlim([-2, 2])
plt.ylim(0.5, 5.5)
plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), c='b', linewidth=3, alpha=0.8)
plt.scatter(list_x, g(np.array(list_x)), c='red', s=100, alpha=0.6)


# 2 Chart: derivative
plt.subplot(1, 2, 2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-6, 8)

plt.plot(x_2, dg(x_2), color='red', linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, c='indigo', s=100, alpha=0.5)
            
plt.show()

print('Number of steps is: ', len(list_x))


# In[26]:


#Run gradient descent 3 times
n = 100

low_gamma = gradient_descent(derivative_func=dg, initial_guess= 3, multiplier=0.0005, precision=0.0001, max_iter=n)

mid_gamma = gradient_descent(derivative_func=dg, initial_guess= 3, multiplier=0.001, precision=0.0001, max_iter=n)

high_gamma = gradient_descent(derivative_func=dg, initial_guess= 3, multiplier=0.002, precision=0.0001, max_iter=n)

#experiment

crazy_gamma = gradient_descent(derivative_func=dg, initial_guess= 1.9, multiplier=0.25, precision=0.0001, max_iter=n)

# Plot  function and derivative and scatter plot side by side



# plotting reduction in cost for each iteration

plt.figure(figsize=[20, 10])

plt.xlim([0, n])
plt.ylim(0, 50)
plt.title('Effect of the learning rate', fontsize=17)
plt.xlabel('Nr of Iterations', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

# Values for our charts 
# 1) y Axis Data: convert the lists to numpy arrays 

low_values = np.array(low_gamma[1])

# 2) x axis data: create a list from 0 to n+!

iteration_list = list(range(0, n+1))

# Plotting Low learning rate
plt.plot(iteration_list, g(low_values), c='blue', linewidth=3, alpha=0.8)
plt.scatter(iteration_list, g(low_values), c='blue',s=80)

# Plotting Mid learning rate
plt.plot(iteration_list, g(np.array(mid_gamma[1])), c='purple', linewidth=3, alpha=0.8)
plt.scatter(iteration_list, g(np.array(mid_gamma[1])), c='red',s=80)

# Plotting High learning rate
plt.plot(iteration_list, g(np.array(high_gamma[1])), c='r', linewidth=3, alpha=0.8)
plt.scatter(iteration_list, g(np.array(high_gamma[1])), c='red',s=80)

# Plotting crazy learning rate
plt.plot(iteration_list, g(np.array(crazy_gamma[1])), c='hotpink', linewidth=3, alpha=0.8)
plt.scatter(iteration_list, g(np.array(crazy_gamma[1])), c='hotpink',s=80)

plt.show()


# # Example 4 - Data Viz with 3D Charts
# 
# ## Minimise  $$f(x, y) = \frac{1}{3^{-x^2 - y^2} + 1}$$
# Minimise $$(x, y) = \frac{1}{r + 1}$$ where $r$ is $3^{-x^2 - y^2}$
# 

# In[27]:


def f(x, y):
    r = 3**(-x**2 - y**2)
    return 1 / (r + 1)


# In[28]:


x_4 = np.linspace(start=-2, stop=2, num=200)
y_4 = np.linspace(start=-2, stop=2, num=200)

type(x_4)
print('shape of X array', x_4.shape)

x_4, y_4 = np.meshgrid(x_4, y_4)
print('Array after meshgrid: ', x_4.shape)


# In[29]:


#Generating 3D Plot
fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('y', fontsize=20)
ax.set_zlabel('f(x, y) - Cost', fontsize=20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap=cm.BuPu, alpha=0.4)

plt.show()


# # Partial Derivatives & Symbolic Computation
# 
# ## $$\frac{\partial f}{\partial x} = \frac{2x \ln(3) \cdot  3^{-x^2 - y^2}}{\left( 3^{-x^2 - y^2} + 1 \right) ^2}$$
# 
# ## $$\frac{\partial f}{\partial y} = \frac{2y \ln(3) \cdot  3^{-x^2 - y^2}}{\left( 3^{-x^2 - y^2} + 1 \right) ^2}$$

# In[30]:


a, b = symbols('x, y')

print('Our cost function f(x, y) is: ', f(a, b))
print('Partial derivative wrt x is: ', diff(f(a, b), a))
print('Value of f(x,y) at x=1.8 y=1.0 is: ',
      f(a, b).evalf(subs={a:1.8, b:1.0})) #python dictonary
print('Value of partial derivative wrt x is: ',diff(f(a,b), a).evalf(subs={a:1.8, b:1.0}))


# In[32]:


# Partial derivatives functions example 4

def fpx(x, y):
    r = 3**(-x**2 - y**2)
    return 2*x*log(3)*r / (r + 1)**2

def fpy(x, y):
    r = 3**(-x**2 - y**2)
    return 2*y*log(3)*r / (r + 1)**2


# In[33]:


fpx(1.8, 1.0)


# In[ ]:





# In[ ]:




