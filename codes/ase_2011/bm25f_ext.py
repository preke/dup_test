# coding = utf-8

from math import log

def countw(word, wlist):
    cnt = 0
    for w in wlist:
        if w == word:
            cnt += 1
    return cnt

def bm25f_ext(d, q, idf_dict, k1, k3, field_weights):
    '''
        fields : summary and description
    '''
    ans = 0.0
    ts = list((set(d['summary']) | set(d['description'])) & (set(q['summary']) | set(q['description'])))
    fields = ['summary', 'description']
    ws = field_weights
    
    for i in ts:
        tfd_i = 0.0
        tfq_i = 0.0
        for j in range(len(fields)):
            try:
                tfd_i += ws[j] * float( countw(i, d[fields[j]]) ) / len(d[fields[j]])
            except:
                tfd_i += 0.0
            try:
                tfq_i += ws[j] * float( countw(i, d[fields[j]]) ) / len(q[fields[j]])
            except:
                tfq_i += 0.0
        
        try:
            idf_i = idf_dict[i]
        except:
            idf_i = 0.0
        ans += ( idf_i \
                 + tfd_i / (k1 + tfd_i) \
                 + (k3 + 1) * tfq_i / (k3 + tfq_i) )
    return ans