from TALENT.model.methods.base import Method
import torch
from tqdm import tqdm
import os.path as osp
import numpy as np
class T2GFormerMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        from TALENT.model.models.t2gformer import T2GFormer
        if model_config is None:
            model_config = self.args.config['model']
        self.frozen_switch = self.args.config['training']['frozen_switch']
        self.model = T2GFormer(
                d_numerical=self.d_in,
                categories=self.categories,
                d_out=self.d_out,
                **model_config
                ).to(self.args.device) 
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()


    def validate(self, epoch):
        """
        Validate the model.

        :param epoch: int, the current epoch
        """
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            

                pred = self.model(X_num, X_cat)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()   

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)


        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                if self.frozen_switch:
                    self.model.froze_topology()
                    self.val_count = 0
                    self.frozen_switch = False
                    print('froze topology')
                else:    
                    self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))   
    