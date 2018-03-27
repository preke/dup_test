import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def train(train_iter, vali_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        logger.info('Epoch:%s\n'%epoch)
        for batch in train_iter:
            query, doc_list = batch.query, batch.doc_list
            # feature1.data.t_(), feature2.data.t_(), target.data.sub_(1), pairid.data.t_()# batch first, index align
            if args.cuda:
                query, doc_list = query.cuda(), doc_list.cuda()

            optimizer.zero_grad()
            results = Variable(torch.Tensor([model(feature1, doc) for doc in doc_list]))
            criterion = nn.NLLLoss()
            loss = criterion(nn.LogSoftmax(results[0]))
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




