import time
from datetime import timedelta
import torch
from torch import optim


def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"

def compute_adj_matrix(data):
    d_hat = torch.eye(len(data.x)) + torch.diag(torch.bincount(data.edge_index[0]))

    adj_mat = torch.zeros((len(data.x), len(data.x)))

    for i in range(len(data.edge_index)):
        adj_mat[data.edge_index[0, i], data.edge_index[1, i]] += 1

    adj_mat.requires_grad = True

    a_tilde = torch.eye(len(data.x)) + adj_mat

    # d_hat_pow = torch.matrix_power(d_hat,)

    evals, evecs = torch.eig(d_hat, eigenvectors=True)  # get eigendecomposition
    evals = evals[:, 0]  # get real part of (real) eigenvalues

    evpow = evals ** (-1 / 2)  # raise eigenvalues to fractional power

    # build exponentiated matrix from exponentiated eigenvalues
    d_hat_pow = torch.matmul(evecs, torch.matmul(torch.diag(evpow), torch.inverse(evecs)))

    a_hat = torch.matmul(torch.matmul(d_hat_pow, a_tilde), d_hat_pow)
    return a_hat, adj_mat

class NetWrapper:

    def __init__(self, model, loss_function, device='cpu', classification=True):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification

    def _train(self, train_loader, optimizer, clipping=None):
        model = self.model.to(self.device)

        model.train()

        loss_all = 0
        acc_all = 0
        for data in train_loader:

            data = data.to(self.device)

            data['a_hat'], _ = compute_adj_matrix(data)

            optimizer.zero_grad()
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)
                
            if self.classification:
                loss, acc = self.loss_fun(data.y, *output)
                loss.backward()

                # torch.autograd.grad(output[0][:, 0], data.edge_index, retain_graph=True)

                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    num_graphs = data.adj.size(0)

                loss_all += loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
            else:
                loss = self.loss_fun(data.y, *output)
                loss.backward()
                loss_all += loss.item()

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()

        if self.classification:
            return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)
        else:
            return None, loss_all / len(train_loader.dataset)

    def classify_graphs(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        for data in loader:
            data = data.to(self.device)
            data['a_hat'], _ = compute_adj_matrix(data)
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            if self.classification:
                loss, acc = self.loss_fun(data.y, *output)

                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    num_graphs = data.adj.size(0)

                loss_all += loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
            else:
                loss = self.loss_fun(data.y, *output)
                loss_all += loss.item()

        if self.classification:
            return acc_all / len(loader.dataset), loss_all / len(loader.dataset)
        else:
            return None, loss_all / len(loader.dataset)

    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=10):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_loss, val_acc = -1, -1
        test_loss, test_acc = None, None

        best_val_loss = 100

        time_per_epoch = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_acc, train_loss = self._train(train_loader, optimizer, clipping)
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step(epoch)

            if test_loader is not None:
                test_acc, test_loss = self.classify_graphs(test_loader)

            if validation_loader is not None:
                val_acc, val_loss = self.classify_graphs(validation_loader)

                # Early stopping (lazy if evaluation)
                if early_stopper is not None and early_stopper.stop(epoch, val_loss, val_acc,
                                                                    test_loss, test_acc,
                                                                    train_loss, train_acc):
                    msg = f'Stopping at epoch {epoch}, best is {early_stopper.get_best_vl_metrics()}'
                    if logger is not None:
                        logger.log(msg)
                        print(msg)
                    else:
                        print(msg)
                    break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} ' \
                    f'TE loss: {test_loss} TE acc: {test_acc}'
                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving...")
                torch.save(self.model.state_dict(), "../graph-sage-binary-maxpool.pt")

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())

        elapsed = format_time(avg_time_per_epoch)

        if early_stopper is not None:
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, best_epoch = early_stopper.get_best_vl_metrics()

        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, elapsed
