import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
# Source: https://github.com/HangtingYe/PTaRL
def run_one_epoch(model, data_loader, loss_func, model_type, config, regularize, ot_weight, diversity_weight, r_weight, diversity, optimizer=None):
    import torch.nn.functional as F
    running_loss = 0.0

    for bid, (X, y) in enumerate(data_loader):
        if data_loader.dataset.X_num is not None and data_loader.dataset.X_cat is not None:
            X_n, X_c = X[0], X[1]
        elif data_loader.dataset.X_cat is not None and data_loader.dataset.X_num is None:
            X_n, X_c = None, X
        else:
            X_n, X_c = X, None
        if model_type[-2:] == 'ot':
            pred, r_, hidden, weight_ = model(X_n, X_c)
            if loss_func == F.cross_entropy:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y.reshape(-1,1))
        
        else:
            pred = model(X_n, X_c)

            if loss_func == F.cross_entropy:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y.reshape(-1,1))

        if optimizer is not None and model_type[-2:] == 'ot':

            norm = torch.mm(torch.sqrt(torch.sum(hidden**2, axis=1, keepdim=True)), torch.sqrt(torch.sum(model.topic.T**2, axis=0, keepdim=True)))

            loss_ot = torch.mean(torch.sum(r_*(torch.mm(hidden.float(), model.topic.T.float()) / norm), axis=1))
            loss += ot_weight * loss_ot

            if diversity == True: 
                selected_rows = np.random.choice(r_.shape[0], int(r_.shape[0] * 0.5), replace=False)

                distance = (r_[selected_rows].reshape(r_[selected_rows].shape[0],1,r_[selected_rows].shape[1])-r_[selected_rows]).abs().sum(dim=2)
                
                if loss_func == F.cross_entropy:
                    label_similarity = (y.reshape(-1,1)[selected_rows] == y.reshape(-1,1)[selected_rows].T).float()
                else:
                    y_min = min(y)
                    y_max = max(y)
                    # using Sturges equation select bins in manuscripts
                    num_bin = 1 + int(np.log2(y.shape[0]))
                    # num_bin = 5
                    interval_width = (y_max - y_min) / num_bin
                    y_assign = torch.max(torch.tensor(0).cuda(),torch.min(((y.reshape(-1,1)-y_min)/interval_width).long(),torch.tensor(num_bin-1).cuda()))
                    label_similarity = (y_assign.reshape(-1,1)[selected_rows] == y_assign.reshape(-1,1)[selected_rows].T).float()
                
                positive_mask = label_similarity
                positive_loss = torch.sum(distance * positive_mask) / (torch.sum(distance)+1e-8)
                loss_diversity = positive_loss

                loss += diversity_weight*loss_diversity

            # first should be sure that the the topic is learnable!
            if regularize == True:
                r_1 = torch.sqrt(torch.sum(model.topic.float()**2,dim=1,keepdim=True))
                topic_metrix = torch.mm(model.topic.float(), model.topic.T.float()) / torch.mm(r_1, r_1.T)
                topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)

                l1 = torch.sum(topic_metrix.abs())
                l2 = torch.sum(topic_metrix ** 2)

                loss_sparse = l1 / l2
                loss_constraint = torch.abs(l1 - topic_metrix.shape[0])

                r_loss = loss_sparse + 0.5*loss_constraint
                
                # loss += 3.0 * r_loss
                loss += r_weight * r_loss

            else:
                None
        else:
            None

        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(data_loader)


def run_one_epoch_val(model, data_loader, loss_func, model_type, config, is_regression):
    pred = []
    ground = []
    for bid, (X, y) in enumerate(data_loader):
        if data_loader.dataset.X_num is not None and data_loader.dataset.X_cat is not None:
            X_n, X_c = X[0], X[1]
        elif data_loader.dataset.X_cat is not None and data_loader.dataset.X_num is None:
            X_n, X_c = None, X
        else:
            X_n, X_c = X, None
        if model_type[-2:] == 'ot':
            pred.append(model(X_n, X_c)[0].data.cpu().numpy())
        else:
            pred.append(model(X_n, X_c).data.cpu().numpy())
        ground.append(y)
    pred = np.concatenate(pred, axis=0)
    y = torch.cat(ground, dim=0)
    
    y = y.data.cpu().numpy()
    try:
        import sklearn
        if not is_regression:
            pred = pred.argmax(1)
            score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
            return score
        else:
            score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 
            return score
    except:
        return 1000000



def fit_Ptarl(args,model, train_loader, val_loader, loss_func, model_type, config, regularize, is_regression,  ot_weight, diversity_weight, r_weight, diversity,seed,save_path):
    if is_regression:
        best_val_loss = 1e30
    else:
        best_val_loss = 0
    best_model = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    early_stop = 20
    epochs = args.max_epoch

    patience = early_stop

    for eid in tqdm(range(epochs)):
        model.train()
        train_loss = run_one_epoch(
            model, train_loader, loss_func, model_type, config, regularize, ot_weight, diversity_weight, r_weight, diversity, optimizer
        )

        model.eval()
        val_loss = run_one_epoch_val(
            model, val_loader, loss_func, model_type, config, is_regression, 
        )

        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')
        if is_regression:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                import copy
                best_model = copy.deepcopy(model)
                if "_ot" in model_type:
                    torch.save(
                        dict(params=best_model.state_dict()),
                        osp.join(save_path, 'best-val-{}.pth'.format(str(seed)))
                    )
                patience = early_stop
            else:
                patience = patience - 1
        else:
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                import copy
                best_model = copy.deepcopy(model)
                if "_ot" in model_type:
                    torch.save(
                        dict(params=best_model.state_dict()),
                        osp.join(save_path, 'best-val-{}.pth'.format(str(seed)))
                    )
                patience = early_stop
            else:
                patience = patience - 1
        if patience == 0:
            break
    if "_ot" in model_type:
        torch.save(
            dict(params=best_model.state_dict()),
            osp.join(save_path, 'epoch-last-{}.pth'.format(str(seed)))
        )
    return best_model,best_val_loss

def test(model, test_loader,no_ot=False):

    model.eval()

    pred = []
    ground = []
    for bid, (X, y) in enumerate(test_loader):
        if test_loader.dataset.X_num is not None and test_loader.dataset.X_cat is not None:
            X_n, X_c = X[0], X[1]
        elif test_loader.dataset.X_cat is not None and test_loader.dataset.X_num is None:
            X_n, X_c = None, X
        else:
            X_n, X_c = X, None
        pred.append(model(X_n, X_c)[0].data.cpu().numpy())
        ground.append(y)
    pred = np.concatenate(pred, axis=0)
    y = torch.cat(ground, dim=0)
    y = y.data.cpu().numpy()
    return pred, y

def generate_topic(model, train_loader,n_clusters):
    hid_ = []
    for bid, (X, y) in enumerate(train_loader):
        if train_loader.dataset.X_num is not None and train_loader.dataset.X_cat is not None:
            X_n, X_c = X[0], X[1]
        elif train_loader.dataset.X_cat is not None and train_loader.dataset.X_num is None:
            X_n, X_c = None, X
        else:
            X_n, X_c = X, None
        hid = model.encoder(X_n, X_c)
        hid_.append(hid.data.cpu().numpy())
    hid_ = np.concatenate(hid_, axis=0)

    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_centers_ = kmeans.fit(hid_).cluster_centers_

    return cluster_centers_

