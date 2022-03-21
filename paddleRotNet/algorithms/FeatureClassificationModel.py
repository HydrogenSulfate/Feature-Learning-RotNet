import numpy as np
import paddle, torch

import time

from . import Algorithm
from pdb import set_trace as breakpoint

# from paddleRotNet.algorithms import Algorithm
# SEED=100
# torch.manual_seed(SEED)
# paddle.seed(SEED)
# np.random.seed(SEED)


def accuracy(output, target, topk=(1, ), flag='paddle'):
    """Computes the precision@k for the specified values of k"""
    if flag == 'torch':
        maxk = max(topk)

        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return np.array([ele.data.cpu().numpy() for ele in res])
    elif flag == 'paddle':
        maxk = max(topk)
        batch_size = target.shape[0]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target = paddle.reshape(target, (1, -1)).expand_as(pred)

        # print(pred.shape)
        # print(target.shape)
        correct = pred.equal(target)
        # print(correct)
        res = []
        for k in topk:
            correct_k = paddle.reshape(correct[:k], [-1]).numpy()
            correct_k = correct_k.sum(0)
            res.append((correct_k * 100.0) / batch_size)
        return np.array(res)


class FeatureClassificationModel(Algorithm):
    def __init__(self, opt):
        self.out_feat_keys = opt['out_feat_keys']
        # print(self.out_feat_keys)
        # exit(0)s
        Algorithm.__init__(self, opt)
        self.feat_extractor = self.networks['feat_extractor']
        self.classifier = self.networks['classifier']

    def forward(self, x: paddle.Tensor):
        y = self.feat_extractor(x, ['conv2'])
        y = self.classifier(y)
        return y

    def allocate_tensors(self):
        self.tensors = {}
        # self.tensors['dataX'] = paddle.to_tensor(dtype='float32')
        # self.tensors['labels'] = paddle.to_tensor(dtype='int64')
    def _trans(
        self,
        path_to_checkpoint_file="./inference.pdparams"
    ):
        # import argparse
        # import os
        # from os import path as osp

        # import paddle
        # from paddle import inference
        # from paddle.inference import Config, create_predictor
        from paddle.jit import to_static
        from paddle.static import InputSpec
        # from paddle.vision import transforms
        # from paddlevideo.utils import get_config
        print(f"Loading params from ({path_to_checkpoint_file})...")
        params = paddle.load(path_to_checkpoint_file)
        self.set_dict(params)

        self.eval()

        input_spec = InputSpec(shape=[None, 3, 32, 32],
                               dtype='float32',
                               name='input'),
        smodel = to_static(self, input_spec=input_spec)
        paddle.jit.save(smodel, "inference")
        print(f"model (RotNet) has been already saved in (inference).")

    def save_pdparams(self):
        self.feat_extractor.set_dict(paddle.load("./feat_extractor.pdparams"))
        self.classifier.set_dict(paddle.load("./classifier.pdparams"))
        paddle.save(self.state_dict(), "inference.pdparams")
        print(f"save RotNet")

    def random_single_forward(
        self,
    ):
        self.eval()
        self.feat_extractor.set_dict(paddle.load("./feat_extractor.pdparams"))
        self.classifier.set_dict(paddle.load("./classifier.pdparams"))
        import numpy as np
        dataX = paddle.to_tensor(np.load("./random_cifar.npy"))
        out_feat_keys = self.out_feat_keys
        finetune_feat_extractor = None
        try:
            finetune_feat_extractor = self.optimizers[
                'feat_extractor'] is not None
        except:
            pass

        import numpy as np
        dataX_var = dataX
        pred_var = self(dataX_var)
        print(pred_var)

    def save_pdparams(self):
        paddle.save(self.state_dict(), "inference.pdparams")
        print(f"save RotNet")

    def _infer_static(self, model_file, params_file):
        from paddle.inference import Config, create_predictor
        config = Config(model_file, params_file)
        config.enable_use_gpu(8000, 0)
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        input_tensor_list = []
        output_tensor_list = []
        for item in input_names:
            input_tensor_list.append(predictor.get_input_handle(item))
        for item in output_names:
            output_tensor_list.append(predictor.get_output_handle(item))
        outputs = []
        img = np.load("./random_cifar.npy")

        for i in range(len(input_tensor_list)):
            input_tensor_list[i].copy_from_cpu(img)
        predictor.run()
        for j in range(len(output_tensor_list)):
            outputs.append(output_tensor_list[j].copy_to_cpu())
        predict = outputs[0]
        print(predict)

    def train_step(self, batch):
        # print(batch[0].shape)
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        # print('0---------')
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        # self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        # self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        self.tensors['dataX'] = batch[0]
        self.tensors['labels'] = batch[1]
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        # ********************************************************

        # ********************************************************
        start = time.time()
        out_feat_keys = self.out_feat_keys
        finetune_feat_extractor = None
        try:
            finetune_feat_extractor = self.optimizers[
                'feat_extractor'] is not None
        except:
            # print('not finetune_feat_extractor ')
            pass

        if do_train:  # zero the gradients
            self.optimizers['classifier'].clear_grad()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].clear_grad()
            else:
                self.networks['feat_extractor'].eval()
        # ********************************************************

        # ***************** SET TORCH fluid.dygraph.to_variableS ******************
        import paddle.fluid as fluid
        import numpy as np
        # with fluid.dygraph.guard():
        dataX_var = dataX
        # dataX_var = fluid.dygraph.to_variable(dataX, volatile=((not do_train) or (not finetune_feat_extractor)))
        labels_var = labels
        # labels_var = fluid.dygraph.to_fluid.dygraph.to_variable(labels, requires_grad=False)
        # ********************************************************

        # ************ FORWARD PROPAGATION ***********************
        # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh', )
        # print(dataX_var)
        print(dataX_var.shape)
        # print(labels_var.shape)
        # exit(0)
        feat_var = self.networks['feat_extractor'](dataX_var,
                                                   out_feat_keys=out_feat_keys)
        if not finetune_feat_extractor:
            if isinstance(feat_var, (list, tuple)):
                for i in range(len(feat_var)):
                    feat_var[i] = fluid.dygraph.to_variable(
                        feat_var[i].numpy())
            else:
                feat_var = fluid.dygraph.to_variable(feat_var.numpy())
        pred_var = self.networks['classifier'](feat_var)
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        record = {}
        if isinstance(pred_var, (list, tuple)):
            loss_total = None
            for i in range(len(pred_var)):
                loss_this = self.criterions['loss'](pred_var[i], labels_var)
                loss_total = loss_this if (loss_total is None) else (
                    loss_total + loss_this)
                record['prec1_c' + str(1 + i)] = accuracy(pred_var[i].numpy(),
                                                          labels,
                                                          topk=(1, ))[0][0]
                record['prec5_c' + str(1 + i)] = accuracy(pred_var[i].numpy(),
                                                          labels,
                                                          topk=(5, ))[0][0]
        else:
            print(pred_var[0])
            exit(0)
            loss_total = self.criterions['loss'](pred_var, labels_var)
            record['prec1'] = accuracy(pred_var, labels, topk=(1, ))[0]
            record['prec5'] = accuracy(pred_var, labels, topk=(5, ))[0]
        record['loss'] = float(loss_total.numpy()[0])
        # ********************************************************

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['classifier'].step()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100 * (batch_load_time / total_time)
        record['process_time'] = 100 * (batch_process_time / total_time)

        return record
