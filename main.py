#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""  This script to control a plant with a artificial neural network control """

__author__ = '{Miguel Angel Pimentel Vallejo}'
__email__ = '{pimentel.vallejo.ma@gmail.com}'
__date__= '{09/Jun/2020}'

# Import the modules needed to run the script
import random
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import optimize

def random_reference(size_of):
    # Probability to move up or down 
    prob = [0.2, 0.8] 

    # statically defining the starting position 
    start = 2
    positions = [start] 

    # creating the random points 
    rr = np.random.rand(size_of) 
    downp = rr < prob[0] 
    upp = rr > prob[1] 


    for idownp, iupp in zip(downp, upp): 
        down = idownp and positions[-1] > 1
        up = iupp and positions[-1] < 4
        positions.append(positions[-1] - down + up) 
 
    return positions



# Function with the model to simulate 
def model(x,t,u):
 
    # Derivatives vector 
    xdot = [0,0]

    # Parameter of the system
    Cb1 = 24.9
    Cb2 = 0.1
    k1 = 1
    k2 = 1

    # Constant input
    w2 = 0.2

    # Dynamic input
    w1 = u

    # Calculate the states of the system
    xdot[0] = w1 + w2 - 0.2*np.sqrt(x[0])
    xdot[1] = (Cb1 - x[1])*(w1/x[0]) + (Cb2 - x[1])*(w2/x[0]) - (k1*x[1])/(1 + k2*x[1])**2

    return xdot 

#Cost Function
def cost_fun_min(u,y,yn,ym,r,b,s,w1,w2,w3,b1,b2):

    com_arg = b1 + y*w1 + u*w2
    dyn_du = w3*w2*((1/(np.cosh(com_arg))**2))
    d2yn_du2 = -2*w3*(w2**2)*(np.tanh(com_arg))*((1/(np.cosh(com_arg)))**2)
    dJ_dU = -2*(ym - yn)*dyn_du - s/((u + r/2 - b)**2) + s/((r/2 + b - u)**2)
    d2J_dU2 = 2*((dyn_du**2) - d2yn_du2*(ym - yn)) + (2*s)/((u + r/2 - b)**3) + (2*s)/((r/2 + b - u)**3)
    u_next = u - dJ_dU/d2J_dU2

    if u_next < 0:
        u_next = 0*u_next

    return u_next


# initial condition
x0 = [1,23]

# time vector
n = 3000
t = np.linspace(0,3000,n)

# Random input
u = random_reference(n-1)

# Vector to save the result
x = np.array([x0]) 

for i in range(1,n):
    # Delat time 
    t_delta = [t[i-1],t[i]]

    # Integrate one step
    xint = odeint(model,x[-1],t_delta,args=(u[i],))
    
    # Save de result for the next step
    x = np.concatenate((x,np.array([xint[1]])),axis=0)


# Plot the system with a random input
plt.figure()
plt.title("System with random input")
plt.subplot(311)
plt.plot(t,x[:,0])
plt.legend(['$h$'])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)

plt.subplot(312)
plt.plot(t,x[:,1])
plt.legend(['$C_b$'])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)

plt.subplot(313)
plt.plot(t,u)
plt.legend(['w_1'])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)

# Prepare the data for training
datainp = np.append( x[:-1,1].reshape(n-1,1), np.array(u[:-1]).reshape(n-1,1) ,axis=1) 
datatar = x[1:,1].reshape(n-1,1)

# Set a ANN with two inputs, one hidden layer and one output layer
net = nl.net.newff([[0, 25], [0, 4]], [1, 1],[nl.trans.TanSig(),nl.trans.PureLin()])

# Set the intial value to the weights
net.layers[0].np['w'][:] = 0
net.layers[1].np['w'][:] = 0
net.layers[0].np['b'][:] = 1
net.layers[1].np['b'][:] = 1

# Change error function
net.error = nl.error.MAE

# Train network
num_epochs = 200
error = net.train(datainp, datatar, epochs=num_epochs, show=1, goal=0.1)

# Simulate network
out = net.sim(datainp)

# Print parameters of training
w = np.concatenate((net.layers[0].np['w'],net.layers[1].np['w']),axis=1)
print("Weights")
print(w)

b = np.concatenate((np.array([net.layers[0].np['b']]),np.array([net.layers[1].np['b']])),axis=1)
print("Bias")
print(b)

# Plot the error
plt.figure()
plt.title("Training error")
plt.plot(list(range(1,num_epochs+1)),error)
plt.xlabel('epoch')
plt.ylabel('magnitude')
plt.grid(True)

# Plot the ANN and original system
plt.figure()
plt.title("Artificial neural network and original system with noise input")
system_label = plt.plot(t,x[:,1])
ann_label = plt.plot(t[1:],out.reshape(n-1),'--')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(system_label + ann_label, ['$C_b$', 'ANN'])

# Intial conditions
x0 = [1,1]

# Time vector
n = 10000
tf = 1000
t = np.linspace(0,tf,n)

# Reference for the system
ym = np.array(int(n/4)*[19] + int(n/4)*[20] + int(n/4)*[21] + int(n/4)*[20.5]) 

# Initial condition for the input
u = np.array([[0]])

# Vector to save the result
x = np.array([x0]) 

# Predict value
xp = np.array([[x0[1]]])

for i in range(1,n):
    # Delat time 
    t_delta = [t[i-1],t[i]]

    # Integrate one step
    xint = odeint(model,x[-1],t_delta,args=(u[-1,0],))
    
    # Predict teh system with Neural Network
    yn = net.sim(np.array([[xint[1,1],u[-1,0]]]))
  
    # Calcule next input
    u_next = cost_fun_min(u[-1,0],xint[1,1],yn,ym[i-1],10,10,10e-20,w[0,0],w[0,1],w[0,2],b[0,0],b[0,1])

    # Save de result for the next step
    x = np.concatenate((x,np.array([xint[1]])),axis=0)
    u = np.concatenate((u,np.array(u_next)),axis=0)
    xp = np.concatenate((xp,np.array(yn)),axis=0)


# Plot the system with a random input
plt.figure()
plt.title("System with control")
plt.subplot(311)
plt.plot(t,x[:,0])
plt.legend(['$h$'])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)

plt.subplot(312)
plt.plot(t,x[:,1])
plt.legend(['$C_b$'])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)

plt.subplot(313)
plt.plot(t,u)
plt.legend(['$w_1$'])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)

# Plot the ANN and system with control
plt.figure()
plt.title("Artificial neural network and original system with control input")
system_label = plt.plot(t,x[:,1])
ann_label = plt.plot(t,xp,'--')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(system_label + ann_label, ['$C_b$', 'ANN'])

# Plot the reference and output
plt.figure()
plt.title("Refence and system with control")
ref_label = plt.plot(t,ym)
system_label = plt.plot(t,x[:,1],'--')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(system_label + ref_label, ['$C_b$', 'ref'])


plt.show()