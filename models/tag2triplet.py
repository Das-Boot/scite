# -*- coding: utf-8 -*-

'''
Author: Zhaoning Li
'''

import numpy as np
from itertools import combinations
import re


def find_idx(ls):
    """
    Find index of causality
    """
    r = []
    element = set(ls)
    if 1 in element:
        for i in re.finditer('1+', ''.join([str(i) for i in [i if i != 2 else 1 for i in ls]])):
            r.append([i for i in range(i.start(), i.end())])
    if 3 in element:
        for i in re.finditer('3+', ''.join([str(i) for i in [i if i != 4 else 3 for i in ls]])):
            r.append([i for i in range(i.start(), i.end())])
    if 5 in element:
        for i in re.finditer('5+', ''.join([str(i) for i in [i if i != 6 else 5 for i in ls]])):
            r.append([i for i in range(i.start(), i.end())])

    return sorted(r)


def check_degree(num, edge, out_degree, in_degree):
    """
    Check out degree and in degree
    """
    out_d, in_d = [0]*num, [0]*num
    for e in edge:
        out_d[e[0]] = 1
        in_d[e[-1]] = 1
    if out_d == out_degree and in_d == in_degree:
        return 1
    else:
        return 0


def check_clause(sw, idx, out_degree, in_degree, cjc_idx, c_and_e):
    """
    Check out clause
    """
    if ',' not in sw:
        f_flag, b_flag = 0, 0
        for i in range(0, cjc_idx+2, 1):
            if out_degree[i] != out_degree[cjc_idx] or in_degree[i] != in_degree[cjc_idx]:
                f_flag = 1
        for i in range(cjc_idx+1, len(idx), 1):
            if out_degree[i] != out_degree[cjc_idx] or in_degree[i] != in_degree[cjc_idx]:
                b_flag = 1
        if f_flag == 1 and b_flag == 1:
            return 0
        else:
            return 1

    if [','] == sw[max(idx[cjc_idx])+1:min(idx[cjc_idx+1])]:
        if ', and' not in ' '.join(sw[min(idx[cjc_idx+1]):]) and ', plus' not in ' '.join(sw[min(idx[cjc_idx+1]):]) and ', or' not in ' '.join(sw[min(idx[cjc_idx+1]):]):
            return 1

    for i in c_and_e:
        if i < cjc_idx and ',' not in sw[max(idx[i])+1:min(idx[cjc_idx])]:
            return 0
    return 1


def check_coordinating_cjc(sw, idx, edge, c_cjc, e_cjc, n_cjc):
    """
    Check out coordinating conjunctions
    """
    if c_cjc != []:
        for cj in c_cjc:
            if [e[-1] for e in edge if e[0] == cj] != [e[-1] for e in edge if e[0] == cj+1]:
                return 0
    if e_cjc != []:
        for ej in e_cjc:
            if [e[0] for e in edge if e[1] == ej] != [e[0] for e in edge if e[1] == ej+1]:
                return 0
    if n_cjc != []:
        for nj in n_cjc:
            for e in [e[-1] for e in edge if e[0] == nj]:
                if (nj+1, e) in edge:
                    return 0
            for e in [e[0] for e in edge if e[-1] == nj]:
                if (e, nj+1) in edge:
                    return 0

    conjunction = [',', 'and', 'or', 'also', ';',
                   'as well as', 'comparable with', 'either', 'plus']
    for e1 in edge:
        for e2 in edge:
            if e1 != e2:
                if e1[0] == e2[0]:
                    count = 0
                    for cjc in conjunction:
                        if cjc not in ' '.join(sw[max(idx[min(e1[1], e2[1])])+1:min(idx[max(e1[1], e2[1])])]):
                            count += 1
                    if count == len(conjunction):
                        return 0
                if e1[1] == e2[1]:
                    count = 0
                    for cjc in conjunction:
                        if cjc not in ' '.join(sw[max(idx[min(e1[0], e2[0])])+1:min(idx[max(e1[0], e2[0])])]):
                            count += 1
                    if count == len(conjunction):
                        return 0
    return 1


def rule_one(sw, out_degree, in_degree, idx, ls):
    """
    From tag sequence to final extracted results:
        one n-ary causal relation:
            C...E... (C >= 1 and E >= 1)
            E...C... (C >= 1 and E >= 1)
    """
    edge = []
    c_flag, e_flag = 0, 0
    conjunction = [',', 'and', 'or', 'also', ';',
                   'as well as', 'comparable with', 'either']
    c_idx = [i.span()[0] for i in re.finditer(
        '1', ''.join([str(i) for i in out_degree]))]
    e_idx = [i.span()[0] for i in re.finditer(
        '1', ''.join([str(i) for i in in_degree]))]
    c_span = [(max(idx[c_idx[c]])+1, min(idx[c_idx[c+1]]))
              for c in range(len(c_idx)-1)]
    e_span = [(max(idx[e_idx[e]])+1, min(idx[e_idx[e+1]]))
              for e in range(len(e_idx)-1)]
    for s in c_span:
        for cjc in conjunction:
            if cjc in ' '.join([sw[i] for i in range(s[0], s[-1], 1)]):
                c_flag += 1
                break
    for s in e_span:
        for cjc in conjunction:
            if cjc in ' '.join([sw[i] for i in range(s[0], s[-1], 1)]):
                e_flag += 1
                break
    if c_flag == len(c_span) or e_flag == len(e_span) or sum(out_degree) == 1 or sum(in_degree) == 1:
        for x in range(len(idx)):
            if out_degree[x] > 0:
                for z in range(len(idx)):
                    if in_degree[z] > 0 and x != z:
                        edge.append((x, z))
    return edge
    

def rule_n(sw, ls, out_degree, in_degree, idx):
    """
    From tag sequence to final extracted results:
        n-ary causal relation
    """
    candidate, c_cjc, e_cjc, n_cjc = [], [], [], []
    conjunction = [',', ';',
                   'and', 'plus', 'also', 'to', 'then', 'of',
                   ', and', 'and ,', 'plus ,', ', plus', ', also', 'also ,', ', of', 'of ,',
                   '; and', 'and ;', 'plus ;', '; plus', '; also', 'also ;', '; of', 'of ;']
    c_and_e = [i for i in range(
        len(idx)) if out_degree[i] == 1 and in_degree[i] == 1]
    for i in range(len(idx)-1):
        if out_degree[i] != in_degree[i] and out_degree[i+1] != in_degree[i+1]:
            for cjc in [',', 'or', 'and', 'plus']:
                if out_degree[i] == 1 and out_degree[i+1] == 1:
                    if sw[max(idx[i])+1:min(idx[i+1])][-1] == cjc and check_clause(sw, idx, out_degree, in_degree, i, c_and_e):
                        c_cjc.append(i)
                    for ce in c_and_e:
                        if ce < i and i not in n_cjc and ',' not in sw[max(idx[ce])+1:min(idx[i])] and ' '.join(sw[max(idx[i])+1:min(idx[i+1])]) in [', and', ', plus', ', or']:
                            n_cjc.append(i)
                if in_degree[i] == 1 and in_degree[i+1] == 1:
                    if sw[max(idx[i])+1:min(idx[i+1])][-1] == cjc and check_clause(sw, idx, out_degree, in_degree, i, c_and_e):
                        e_cjc.append(i)
                    for ce in c_and_e:
                        if ce < i and i not in n_cjc and ',' not in sw[max(idx[ce])+1:min(idx[i])] and ' '.join(sw[max(idx[i])+1:min(idx[i+1])]) in [', and', ', plus', ', or']:
                            n_cjc.append(i)
    for x in range(len(idx)):
        if out_degree[x] > 0:
            for z in range(len(idx)):
                if in_degree[z] > 0:
                    flag = 0
                    if x > z:
                        for cjc in conjunction:
                            if cjc == ' '.join([sw[i] for i in range(max(idx[z])+1, min(idx[x]), 1)]):
                                flag = 1
                                break
                        if flag == 0:
                            candidate.append((x, z))
                    elif x < z:
                        for cjc in conjunction:
                            if cjc == ' '.join([sw[i] for i in range(max(idx[x])+1, min(idx[z]), 1)]):
                                flag = 1
                                break
                        if flag == 0:
                            candidate.append((x, z))
    record = []
    for t in range(max(sum(out_degree), sum(in_degree)), len(candidate)+1, 1):
        flag = 0
        for i in combinations(candidate, t):
            if check_degree(len(idx), i, out_degree, in_degree):
                if 5 not in ls:
                    record.append(([np.abs(e[0]-e[1])
                                    for e in i], list(i)))
                    flag = 1
                else:
                    if check_coordinating_cjc(sw, idx, i, c_cjc, e_cjc, n_cjc):
                        record.append(
                            (sum([np.abs(e[0]-e[1]) for e in i]), list(i)))
                        flag = 1
        if flag == 1:
            break
    if record != []:
        return min(record)[-1]
    else:
        return 0


def final_result(ls, sw):
    """
    From tag sequence to final extracted results
    """
    len_sen = len(sw)
    ls = ls[:len_sen]
    idx = find_idx(ls)
    
    if set(ls) == {0}:
        return 0
    
    if idx == []:
        return 0
    
    out_degree, in_degree = [0]*len(idx), [0]*len(idx)
    for i in range(len(idx)):
        if ls[idx[i][0]] == 1:
            out_degree[i] = 1
        if ls[idx[i][0]] == 3:
            in_degree[i] = 1
        if ls[idx[i][0]] == 5:
            out_degree[i] = 1
            in_degree[i] = 1
    
    if sum(out_degree) == 0 or sum(in_degree) == 0:
        return 0
    
    if 5 in ls:
        if sum(out_degree) < 2 or sum(in_degree) < 2:
            return 0
    
    Edge = []
    if 5 not in ls:
        c = [i for i in re.finditer(
            '1+', ''.join([str(i) for i in out_degree]))]
        e = [i for i in re.finditer(
            '1+', ''.join([str(i) for i in in_degree]))]
        if len(c) == 1 and len(e) == 1:
            Edge = rule_one(sw, out_degree, in_degree, idx, ls)
    if 5 in ls or Edge == []:
        Edge = rule_n(sw, ls, out_degree, in_degree, idx)

    if Edge == [] or Edge == 0:
        return 0
    else:
        return [[idx[ee] for ee in e] for e in Edge]