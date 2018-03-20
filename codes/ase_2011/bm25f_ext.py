# coding = utf-8

from math import log

def countw(word, wlist):
    cnt = 0
    for w in wlist:
        if w == word:
            cnt += 1
    return cnt

# def idf(corpus, word):
#     '''
#         the corpus means corpus of d (all fields)
#     '''
#     total_len = len(corpus)
#     total_wcount = countw(word, corpus)
#     return log((total_len - total_wcount + 0.5)/(total_wcount + 0.5))

def bm25f_ext(d, q, idf_dict, k1, k3, field_weights):
    '''
        fields : summary and description
    '''

    ans = 0.0
    ts = list( (set(d['summary']) | set(d['description'])) & (set(q['summary']) | set(q['description'])) )
    # tuning parameters, ignore bfs
    fields = ['summary', 'description']
    
    ws = field_weights
    
    # k1 = 2.0
    # ws = [2.980,5.0] 
    # k3 = 0.1
    
    for i in ts:
        tfd_i = 0.0
        tfq_i = 0.0
        # print len(fields)
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
            # print i
            idf_i = 0.0
        ans += ( idf_i \
                 + tfd_i / (k1 + tfd_i) \
                 + (k3 + 1) * tfq_i / (k3 + tfq_i) )
    return ans