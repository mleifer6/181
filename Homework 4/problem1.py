def l1(x1, x2):
	total = 0
	for i in range(2):
		total += abs(x1[i] - x2[i])
	return total

def l2(x1, x2):
	total = 0
	for i in range(2):
		total += (x1[i] - x2[i]) ** 2
	return total ** 0.5

def l_infty(x1, x2):
	max_i = 0
	for i in range(2):
		if abs(x1[i] - x2[i]) > max_i:
			max_i = abs(x1[i] - x2[i]) 
	return max_i

xs = [(0.1,0.5), (0.35,0.75), (0.28,1.35), (0,1.01)]
norms = [l1, l2, l_infty]

for norm in norms:
	for i in range(4):
		for j in range(i + 1,4):
			print i + 1, j + 1, norm(xs[i], xs[j])
	print "#" * 80