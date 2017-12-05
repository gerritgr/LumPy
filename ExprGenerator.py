import numpy as np

def gen_ame(s, k, m, i_rules, c_rules, states):
	#s, k, m, i_rules, c_rules, states = ame_basis_tuple
	assert(k >= 0)
	assert(np.sum(m) == k)
	assert(s.isalpha())
	assert(states == sorted(states))

	m_id = str(list(m)).replace(', ','_').replace('[', '_').replace(']', '')
	line = 'dt_x["{s}_{k}_{m_id}"] = 0'.format(s=s, k=k, m_id=m_id)

	for rule in i_rules:
		consume, product, rate = rule
		if product != s:
			continue
		line += '+({r}*x["{c}_{k}_{m_id}"])'.format(c=consume, r=rate, m_id=m_id, k=k)

	for rule in i_rules:
		consume, product, rate = rule
		if consume != s:
			continue
		line += '-({r}*x["{s}_{k}_{m_id}"])'.format(s=s, r=rate, m_id=m_id, k=k)

	for rule in c_rules:
		consume1 = rule[0][0]
		consume2 = rule[0][1]
		product1 = rule[1][0]
		product2 = rule[1][1]
		rate = rule[2]

		if product1 != s:
			continue
		m_rj2 = m[states.index(consume2)]
		line += '+({r}*x["{c1}_{k}_{m_id}"]*{m_rj2})'.format(c1=consume1, r=rate, m_id=m_id, k=k, m_rj2=m_rj2)

	for rule in c_rules:
		consume1 = rule[0][0]
		consume2 = rule[0][1]
		product1 = rule[1][0]
		product2 = rule[1][1]
		rate = rule[2]

		if consume1 != s:
			continue
		m_rj2 = m[states.index(consume2)]
		line += '-({r}*x["{s}_{k}_{m_id}"]*{m_rj2})'.format(s=s, r=rate, m_id=m_id, k=k, m_rj2=m_rj2)

	for s1 in states:
		for s2 in states:
			if s1 == s2:
				continue
			beta = 'beta_{s}_{s1}_to_{s}_{s2}'.format(s=s, s1=s1,s2=s2)
			m_ngbr = list()
			for i in range(len(m)):
				if states[i] == s1:
					m_ngbr.append(m[i]+1)
				elif states[i] == s2:
					m_ngbr.append(m[i]-1)
				else:
					m_ngbr.append(m[i])
			m_ngbr_id = str(list(m_ngbr)).replace(', ','_').replace('[', '_').replace(']', '')
			x_ngbr = 'x["{s}_{k}_{m_ngbr_id}"]'.format(s=s, k=k, m_ngbr_id = m_ngbr_id)
			m_ngbr_s1 = m_ngbr[states.index(s1)]
			x_ngbr = '0.0' if '-1' in x_ngbr else x_ngbr
			line += '+({beta}*{x_ngbr}*{m_ngbr_s1})'.format(beta=beta, x_ngbr = x_ngbr, m_ngbr_s1 = m_ngbr_s1)

	for s1 in states:
		for s2 in states:
			if s1 == s2:
				continue
			beta = 'beta_{s}_{s1}_to_{s}_{s2}'.format(s=s, s1=s1,s2=s2)
			m_s1 = m[states.index(s1)]
			#print(m, states, 's1', s1, 's1count', m_s1)
			line += '-({beta}*x["{s}_{k}_{m_id}"]*{m_s1})'.format(beta=beta, s = s, k = k, m_id = m_id, m_s1 = m_s1)

	#print(line)
	return line



def gen_beta(states_tuple):
	state, s1, s2 = states_tuple
	beta = 'beta_{s}_{s1}_to_{s}_{s2} = 1.0/({edge_count})*({agg_rate})'
	edge_count = "edges('{s}','{s1}')"
	agg_rate = "rates('{s}','{s1}','{s2}')"
	edge_count = edge_count.format(s=state, s1 = s1, s2 = s2)
	agg_rate = agg_rate.format(s=state, s1 = s1, s2 = s2)
	beta = beta.format(s=state, s1 = s1, s2 = s2, edge_count = edge_count, agg_rate = agg_rate)
	return beta


def delte_unused_betas(odes, beta_exprs):
	used_betas = list()
	seperators = " ,;,+,-,=,),(,*,/,**".split(',')
	for ode in odes:
		ode_formula = ode[0]
		for ch in seperators:
			ode_formula = ode_formula.replace(ch, ' '+ch+' ')
		tokens = ode_formula.split(' ')
		tokens = [t.strip() for t in tokens if 'beta_' in t]
		for t in tokens:
			used_betas.append(t)
	beta_exprs = [beta for beta in beta_exprs if beta.split('=')[0].strip() in used_betas]
	return beta_exprs
