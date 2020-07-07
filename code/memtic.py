from local_search import *
import random
class memtic():
	def __init__(self, Input,Reference, Parent, u=10, lam = 9, tau=7, u_elite = 7):
		self.Input = Input
		self.Reference = Reference
		self.u = u
		self.lam = lam
		self.tau = tau
		self.u_elite = u_elite

	def memtic_algo():
		t = 0
		p =    []  # population
		while (t< len(P)):
			for i in p:
				if i > self.tau:
					i = random.choice(p)      # replace with random individual
				c = local_search.local_search.SOR(Input, Reference, tau, p)
				i = c
				p_d = remove_duplicate(p)
			while (len(p) <  u):
				p.append(p(i))
			t = t+1

		return p


def remove_duplicate(x):
	return list(dist.fromkeys(x))

