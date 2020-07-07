from sklearn.neighbors import NearestNeighbors as nn
import numpy as np
import scipy.signal as ss

class local_search():
	def __init__(self,R,S,tau,p):
		self.R = R
		self.S = S
		self.tau =tau
		self.p = p
	def SOR(self):
		'''
		single object replacement

		'''
		for k in range(size(self.tau)):
			c = self.tau[k]
			Prob = tau_prob(tau)
			for j in prob:
				for i in range(n):
					tau_c = tau
					x1 = nn(n_neighbors =2, algorithm = 'ball_tree').fit(R)
					distance, x = x1.kneighbors(R)
					f_best = 0
					for e in x:
						tau_c[i] = e
						#evalute the fitness: function calling fitness
						if f(tau_c)> f_best:
							x_best = e
							f_best = f(tau_c)
					tau_c[i] = x_best
					c = c.union(tau_c)
		return c

def fitness(t_c, S):
	n = len(S)
	t = 2/(n(n-1))
	phi_inner = 0
	phi_outer = 0
	for i in range(1,n+1):
		for j in range(i+1,n+1):
			p = trap(theta)*mu_pair() # mu_pair function calling
			phi_inner  = phi_inner+p
		phi_outer = phi_outer+phi_inner	
	return t*phi_outer


def mu_pair(o_i,o_j,theta_ij,o_i_dash,o_j_dash,theta_ij_dask):
	beta = 0.5
	force_0_ab = force()
	force_2_ab_dash = force()


	uc_0 = ss.corelate(force_0_ab,force_0_ab_das)
	uc_2 = ss.correlate(force_2_ab,force_2_ab_dash)

	return 0.5*uc_0 + 0.5*uc_2

def force():
	const_force = 10
	[n,m] = size(A)
	[n1,m1] = size(B)
	mass_A = n*m
	mass_B = n1*m1
	for i in range(10):
		'''
		constant force change as the change in the thickness between the object
		'''
		pass
	F_2 = []
	for j in arrary_A: # verify or change range(10), by the number of elements in the for matching
		for k in array_B:
			[n,m] = size(A)
			[n1,m1] = size(B)
			mass_A = n*m
			mass_B = n1*m1
			d = ((x1-x2)**2+(y1-y2)**2)**(1/2)        # use distance formula
			f_2 = mass_A*mass_B/(d**2)
		F_2.apped(f_2)
	return F_2, f_0


def trap(theta):
	if (theta > -(np.pi/4) and theta < -(np.pi/8)):
		theta1 = np.linspace(-np.pi/4,-np.pi/8,num=1000)
		val = np.linspace(0,1,num=100000)
		z = zip(theta1,val)
		dict = dict(list(z))
		return dict[theta]

	elif (theta >= (-np.pi/8) and theta <= (np.pi/8)):
		theta1 =  np.linspace(-np.pi/8,np.pi/4,num=1000)
		val = np.linspace(1,1,num = 100000)
		z = zip(theta1,cal)
		dict = dict(list(z))
		return dict[theta]

	elif (theta > (np.pi/8) and theta < (np.pi/4)):
		theta1 = np.linspace(np.pi/8,np.pi/4,num=1000)
		val = np.linspace(1,0,num = 100000)
		z = zip(theta1,val)
		dict = dict(list(z))
		return dict[theta]

	else:
		return 0


def permutationanother (list):
	'''
	generating permutation list for the given 'list'
	'''

	from itertools import permutations
	perm = permutations(list)
	for t in perm:
        	print (t)

def permutation(list):
	'''
	generation of the permutation fo rthe given list i.e., 'list'	
	'''
	if len(list) == 0:
		return []

	if len(list) == 1:
		return [list]
	l = []
	for i in range(len(list)):
		m = list[i]
		remlist = list[:i] + list[i+1:]
		for p in permutation(remlist):
			l.append([m] +p)
	return l

data = list('123')
#for q in permutation (data):
#	print (q)
data = permutationanother(data)
