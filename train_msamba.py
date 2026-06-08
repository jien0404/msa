import os
import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from tensorboardX import SummaryWriter
from models.msamba import build_model
from core.metric import MetricsTop
from core.contrastive_loss import contrastive_loss

opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))

train_mae, val_mae = [], []

# Metrics where a HIGHER value is better; everything else (e.g. MAE, loss) is
# treated as lower-is-better for checkpoint selection / early stopping.
HIGHER_IS_BETTER = {
    'Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'Mult_acc_7',
    'Has0_acc_2', 'Non0_acc_2', 'Has0_F1_score', 'Non0_F1_score',
    'F1_score', 'Corr',
}


def is_better(metric_name, new_value, best_value):
    """Return True if new_value is a better score than best_value for metric_name."""
    if best_value is None:
        return True
    if metric_name in HIGHER_IS_BETTER:
        return new_value > best_value
    return new_value < best_value  # MAE / loss: lower is better


def detect_sub_loss_support(model, dataLoader):
    """Dry-run one mini-batch to check the model actually emits the sub-loss
    outputs. Avoids the brittle `'c' in project_name` heuristic that silently
    disabled the auxiliary supervision for models such as MSAmba_ALMT."""
    needed = ['sub_output_V', 'sub_output_T', 'sub_output_A',
              'cls_V', 'cls_A', 'cls_T']
    model.eval()
    with torch.no_grad():
        sample = next(iter(dataLoader['train']))
        img = sample['vision'][:2].to(device)
        audio = sample['audio'][:2].to(device)
        text = sample['text'][:2].to(device)
        out = model(img, audio, text)
    model.train()
    return all(out.get(k) is not None for k in needed)


def main():
    opt = parse_opts()
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("--------------------------OPT--------------------------")
    print("seed: {}".format(opt.seed))

    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)

    save_path = os.path.join(opt.models_save_root, opt.project_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    print(opt)
    print("-------------------------------------------------------")

    model = build_model(opt).to(device)

    dataLoader = MMDataLoader(opt)


    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(model.text_model.named_parameters())
    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    all_params = list(model.named_parameters())
    alpha_params = [p for n, p in all_params if 'alpha' in n]
    other_params = [p for n, p in all_params if 'text_model' not in n]


    optimizer = torch.optim.AdamW(
        [
            {'params': bert_params_decay, 'weight_decay': 0.01, 'lr':1e-5},
            {'params': bert_params_no_decay, 'weight_decay':0.0, 'lr': 1e-5},
            {'params': other_params, 'weight_decay': 1e-4, 'lr': 5e-4},
        ]
    )

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.L1Loss()
    cls7_loss_fn = torch.nn.CrossEntropyLoss()   # auxiliary 7-class head
    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)

    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters count: ", param_cnt / 1000000, 'M')

    # ---- B4: enable auxiliary (sub) losses based on what the model actually
    #          produces, instead of the fragile project-name string check. ----
    if opt.sub_loss:
        if detect_sub_loss_support(model, dataLoader):
            print("Sub-loss outputs detected -> auxiliary supervision ENABLED.")
        else:
            print("Model does not expose sub-loss outputs -> disabling sub_loss.")
            opt.sub_loss = False
    else:
        print("sub_loss flag is False -> auxiliary supervision disabled.")

    # ---- B1/B2/B3: honest model selection + early stopping bookkeeping ----
    select_metric = opt.select_metric
    best_select_value = None      # best validation value of the selection metric
    best_epoch = -1               # epoch at which the selected checkpoint was saved
    best_val_results = None       # full valid metric dict at the selected epoch
    best_test_results = None      # test metric dict at the SAME (selected) epoch
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(save_path, 'best.pth')

    for epoch in range(1, opt.n_epochs + 1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics, opt.sub_loss, opt.sub_loss_lambda,
              opt=opt, cls7_loss_fn=cls7_loss_fn)
        val_results = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, writer, save_path, metrics, opt=opt)
        test_results = None
        if opt.is_test is not None:
            test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, metrics)
        scheduler_warmup.step()

        # ---- checkpoint selection by the chosen VALIDATION metric (default MAE) ----
        if select_metric not in val_results:
            raise ValueError(f"--select_metric '{select_metric}' not in valid metrics {list(val_results.keys())}")
        cur = val_results[select_metric]
        if is_better(select_metric, cur, best_select_value):
            best_select_value = cur
            best_epoch = epoch
            best_val_results = val_results
            best_test_results = test_results
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[select] New best valid {select_metric}={cur:.4f} @ep{epoch} -> saved {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if opt.patience and epochs_no_improve >= opt.patience:
                print(f"[early-stop] No valid {select_metric} improvement for {opt.patience} epochs. "
                      f"Stopping at epoch {epoch}.")
                break

    writer.close()

    # ---- B2: honest single-checkpoint report (the deployable model) ----
    print("\n==================== FINAL (single checkpoint) ====================")
    print(f"Selected by valid {select_metric} @ epoch {best_epoch} "
          f"(valid {select_metric}={best_select_value:.4f}); checkpoint: {best_ckpt_path}")
    if best_test_results is not None:
        # machine-parseable line to make multi-seed averaging trivial
        kv = " ".join(f"{k}={v:.4f}" for k, v in best_test_results.items())
        print(f"FINAL[seed={opt.seed}] epoch={best_epoch} TEST {kv}")
    else:
        print("(is_test disabled -> no test metrics recorded for the selected checkpoint)")
    print("===================================================================\n")


def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics, sub_loss=False, sub_loss_lambda=0.0, opt=None, cls7_loss_fn=None):
    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()

    y_pred, y_true = [], []

    if opt.use_con_loss:
        con_loss_fn = contrastive_loss(opt.datasetName, device, 0.4)

    model.train()
    # print(model)
    print("-----------------------TRAIN---------------------")
    for cur_iter, data in train_pbar:
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(img, audio, text)


        if not sub_loss:
            loss = loss_fn(output['output'], label)
        else:
            if not opt.use_con_loss:
                loss = loss_fn(output['output'], label) + sub_loss_lambda * (loss_fn(output['sub_output_V'], label) +
                                                                             loss_fn(output['sub_output_T'], label) +
                                                                             loss_fn(output['sub_output_A'], label) +
                                                                             loss_fn(output['cls_V'], label) +
                                                                             loss_fn(output['cls_A'], label) +
                                                                             loss_fn(output['cls_T'], label)
                                                                             )
            else:
                loss = loss_fn(output['output'], label) + sub_loss_lambda * (loss_fn(output['sub_output_V'], label) +
                                                                             loss_fn(output['sub_output_T'], label) +
                                                                             loss_fn(output['sub_output_A'], label)) + \
                       opt.con_loss_lambda * con_loss_fn(output, label)

        # auxiliary 7-class loss (only when model exposes cls7_logits)
        if cls7_loss_fn is not None and 'cls7_logits' in output:
            label7 = label.squeeze(1).round().add(3).long().clamp(0, 6)
            loss = loss + 0.3 * cls7_loss_fn(output['cls7_logits'], label7)

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        output = output['output']

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('TRAIN')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][-1]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    print('TRAIN: ', train_results)
    train_mae.append(train_results['MAE'])
    print("-------------------------------------------------")

    writer.add_scalar('train/loss', losses.value_avg, epoch)
    for k, v in train_results.items():
        writer.add_scalar(f'train/{k}', v, epoch)


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics, opt=None):
    test_pbar = tqdm(enumerate(eval_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        print("-----------------------EVAL----------------------")
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)['output']

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)
        print("-------------------------------------------------")

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)
        for k, v in test_results.items():
            writer.add_scalar(f'evaluate/{k}', v, epoch)

    # Selection / early-stopping is handled centrally in main(); just return.
    return test_results


def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics, opt=None):
    test_pbar = tqdm(enumerate(test_loader))
    print("-----------------------TEST----------------------")
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)['output']
            # if type(output) is tuple:
            #     output, _, _, _ = output

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)
        print("-------------------------------------------------")

        writer.add_scalar('test/loss', losses.value_avg, epoch)
        for k, v in test_results.items():
            writer.add_scalar(f'test/{k}', v, epoch)

    return test_results


if __name__ == '__main__':
    main()
