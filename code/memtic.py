import cv2
from local_search import *
import random
import scipy.io as io

#def load(load):
input = io.loadmat('pat_m_10.mat')
refernce = cv2.imread('../fused_image_middle10.png',0)
refernce = np.array(refernce)

class memtic():
	def __init__(self, input, reference, u=10, lam = 9, tau=7, u_elite = 7):
		self.input = input
		self.reference = reference
		self.u = u
		self.lam = lam
		self.tau = tau
		self.u_elite = 7
		
	def memtic_algo(self):
		Input = self.input
		Reference  = self.reference
		t = 0
		p1 = self.tau
#		print (p1.type())
#		exit()
		p =  Input  # population
#		print (p)
#		exit()
		img = []
		while (t< len(p)):
			for q in range(10):
				for j in range(31):
	
					i = p[q][j]
#					print (i)
#					exit()
					if i.all() > self.tau:
						i = random.choice(p)      # replace with random individual
					c1 = local_search(Input, Reference, i)
					c = c1.SOR()
					i = c
					p_d = remove_duplicate(p)
				while (len(p) <  u):
					img.append(p(i))
				t = t+1		

		return img


def remove_duplicate(x):
	return list(dist.fromkeys(x))

p = memtic(input['patches'],refernce)
p1 = p.memtic_algo()
print (p1)
