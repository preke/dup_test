import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def train(train_iter, vali_iter, model, args):
    if args.cuda:
        model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))    
    optimizer  = torch.optim.Adam(parameters, lr=args.lr)
    steps      = 0
    best_acc   = 0
    last_step  = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        print('Epoch:%s\n'%epoch)
        for batch in train_iter:
            query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4, neg_doc_5 = \
            batch.query, batch.pos_doc, batch.neg_doc_1, batch.neg_doc_2, batch.neg_doc_3, batch.neg_doc_4, batch.neg_doc_5
            # query.t_(), pos_doc.t_(), neg_doc_1.t_(), neg_doc_2.t_(), neg_doc_3.t_(), neg_doc_4.t_(), neg_doc_5.t_()
            if args.cuda:
                query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4, neg_doc_5 = \
                query.cuda(), pos_doc.cuda(), neg_doc_1.cuda(), neg_doc_2.cuda(), neg_doc_3.cuda(), neg_doc_4.cuda(), neg_doc_5.cuda()
            
            optimizer.zero_grad()
            results = torch.cat([model(query, pos_doc).view(-1,1), model(query, neg_doc_1).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_2).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_3).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_4).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_5).view(-1,1)], 1)
            print(results.shape)
            criterion  = nn.NLLLoss()
            target_tmp = Variable(torch.FloatTensor(np.array([1, 0,0,0,0,0], dtype=float).reshape(6,1)))
            target     = target_tmp
            for i in range(args.batch_size - 1):
                target = torch.cat([target, target_tmp, 1])
            print(target.shape)
            
            loss = criterion(results, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}'.format(steps, loss.data[0]))
            
            if steps % args.test_interval == 0:
                pass
                # vali_acc = eval(vali_iter, model, args)
                # if vali_acc > best_acc:
                #     best_acc = vali_acc
                #     last_step = steps
                #     if args.save_best:
                #         save(model, args.save_dir, 'best', steps)
                # else:
                #     if steps - last_step >= args.early_stop:
                #         print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                print('save loss: %s' %str(loss.data))
                save(model, args.save_dir, 'snapshot', steps)


# def eval(data_iter, model, args):
#     model.eval()
#     accuracy, avg_loss = 0, 0
#     for batch in data_iter:
#         query, doc = batch.query, batch.doc
#         # query.data.t_(), feature2.data.t_(), target.data.sub_(1), pairid.data.t_() # batch first, index align
#         if args.cuda:
#             query, doc = query.cuda(), doc.cuda()

#         logit = model(feature1, feature2)
#         target = target.type(torch.cuda.FloatTensor)
#         criterion = nn.MSELoss()
#         loss_list = []
#         length = len(target.data)
#         for i in range(length):
#             a = logit.data[i]
#             b = target.data[i]
#             loss_list.append(float(0.5*(b-a)*(b-a)))
#         corrects = 0
#         for item in loss_list:
#             avg_loss += item 
#             if item <= 0.125:
#                  corrects += 1
#         accuracy = 100.0 * float(corrects)/batch.batch_size 
#     size = float(len(data_iter.dataset))
#     avg_loss /= size
#     accuracy = 100.0 * float(corrects)/size
#     print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
#                                                                        accuracy, 
#                                                                        corrects, 
#                                                                        size))
#     return accuracy




