import os
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from networks import build_model
from opts import opts # options for the project
from prepare_data import generate_dataloader # prepare the data and dataloader
from evaluation import eval, eval_fig2
from step3 import train_step3
from utils import get_clu_centers, resume_pretrained_weights  # *
from pseudo_label_analysis import (
    PseudoLabelTracker,
    evaluate_pseudo_labels,
    run_k_sensitivity,
    run_k_misspecification,
    load_target_arrays,
    plot_k_sensitivity,
    plot_k_misspec,
    write_summary_report,
    collect_pseudo_from_loader,
)
from torch.distributions.uniform import Uniform
args = opts()
# I2AwA: 3D2 (source) -> AwA2 (target); override via CLI if needed
args.step3 = 'train'
args.att_type = 'binary'  # 'binary'|'continuous'
args.combine_za = True  # f=[z,a] | f=z
args.init_prot_type = 'sample'  # 'center' | 'sample'
args.GenA_type = 'FC'  # 'GCN', 'FC', 'MulDis'

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
if cuda:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def main():
    # (1) process data and prepare dataloaders
    train_loader_source, train_loader_target, test_loader_target = generate_dataloader(args)
    dataloaders = {'tr_loader_src': train_loader_source,
                   'tr_loader_tgt': train_loader_target,
                   'te_loader_tgt': test_loader_target}
    # (1.1) Attributes centers groungd truth: (50 x 85)
    if args.att_type == 'binary':
        if args.dataset == 'I2AwA':
            from i2awa_config import load_attribute_matrix
            import torch as _torch
            att_mat = load_attribute_matrix()
            att_cents = Variable(_torch.tensor(att_mat).type(FloatTensor))
            df_att = None
        else:
            att_path = os.path.join(args.data_path_target, 'attributes', 'attributes_bi.csv')
            df_att = pd.read_csv(att_path, header=None, index_col=None)
    elif args.att_type == 'continuous':
        raise Exception("No continuous attributes")
    else:
        raise ValueError("att type does not exist")

    if df_att is not None:
        att_cents = Variable(torch.tensor(df_att.values).type(FloatTensor))

    # (1.2) Init prot centers pseudo: (50 x 512), first 40 are shared classes
    if args.init_prot_type == 'center':
        raise Exception("No center based clu prot")
        # xt_clu_cents = pd.read_csv('./data/N2AwA/pseudo/'+args.src+'2'+args.tgt+'_sample_xt_clu_cents17.csv').values
        # xt_clu_cents = Variable(torch.tensor(xt_clu_cents).type(FloatTensor))
    elif args.init_prot_type == 'sample':
        cent_path = os.path.join(
            args.data_path_target, 'pseudo',
            args.src + '2' + args.tgt + '_sample_xt_clu_cents{}.csv'.format(args.tgt_nc))
        if os.path.isfile(cent_path):
            from data_paths import load_cluster_centers
            xt_clu_cents = load_cluster_centers(cent_path, expected_k=args.tgt_nc)
            xt_clu_cents = Variable(torch.tensor(xt_clu_cents).type(FloatTensor))
        else:
            xt_clu_cents = Variable(torch.zeros(args.tgt_nc, 512).type(FloatTensor))
    else:
        raise ValueError("init proto type does not exist")
    centers = {'att_cents': att_cents, 'xt_clu_cents': xt_clu_cents}


    # (2) Buid the mdoels
    x_dim = 2048  # ResNet-50 feature dim
    z_dim = 512
    a_dim = 85
    if args.combine_za:
        f_dim = z_dim + a_dim
    else:
        f_dim = z_dim
    shr_nc = args.shr_nc

    GenZ = build_model(args, 'GenZ', input_size=x_dim, h1=1024, h2=z_dim).to(device)
    if args.GenA_type == 'GCN':  # To be explored in the future
        GenA = build_model(args, 'GCN', input_size=z_dim, h1=256, h2=a_dim).to(device)
    elif args.GenA_type == 'FC':
        GenA = build_model(args, 'GenA', input_size=z_dim, h1=256, h2=a_dim).to(device)
    elif args.GenA_type == 'MulDis':
        GenA = build_model(args, 'MulDis', input_size=z_dim, h1=256, h2=a_dim).to(device)  # Binary classifier
    else:
        raise ValueError("GenA_type not exists.")
    Clf = build_model(args, 'Clf', input_size=f_dim, h1=256, h2=shr_nc+1).to(device)
    ClfSU = build_model(args, 'ClfSU', input_size=f_dim, h1=256, h2=2).to(device)  # Seen/Unseen Classifier

    ProtClf = build_model(args, 'Prot').to(device)

    toTrModels = {'GenZ': GenZ, 'Clf': Clf, 'GenA': GenA, 'ClfSU': ClfSU}
    nonTrModels = {'ProtClf': ProtClf}  # No trainable params in the Prototype-classifier


    # (3) Specify loss functions and initialization
    BCELoss = torch.nn.BCELoss().to(device)
    CELoss = torch.nn.CrossEntropyLoss().to(device)
    MSELoss = torch.nn.MSELoss().to(device)
    KLDivLoss = torch.nn.KLDivLoss(reduction='sum').to(device)
    lossFunctions = {'BCELoss': BCELoss, 'CELoss': CELoss, 'MSELoss': MSELoss, 'KLDivLoss': KLDivLoss}

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    # (4) Create optimizers
    optimizers = {}
    for m in toTrModels:
        opt_name = 'opt_' + m
        optimizers[opt_name] = torch.optim.Adam(toTrModels[m].parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)


    # # (5) Create log directory
    # if not os.path.isdir(args.log):
    #     os.makedirs(args.log)
    # log = open(os.path.join(args.log, 'train_log.txt'), 'w')
    # state = {k: v for k, v in args._get_kwargs()}
    # log.write(json.dumps(state) + '\n')
    # log.close()
    #
    # log = open(os.path.join(args.log, 'train_log.txt'), 'a')
    # log.write('\n-------------------------------------------\n')
    # log.write(time.asctime(time.localtime(time.time())))
    # log.write('\n-------------------------------------------')
    # log.close()

    cudnn.benchmark = True

    # Optional: pseudo-label analysis only (K-sensitivity + K-misspecification)
    if getattr(args, 'pseudo_analysis_only', False):
        run_offline_pseudo_analysis(args, dataloaders)
        return

    if getattr(args, 'method', 'srosda') == 'fig2':
        run_fig2_training(args, dataloaders, att_cents)
        return

    # (6). Start training (original SROSDA Step 3)
    centers['zt_clu_cents'], centers['at_clu_cents'] = get_clu_centers(args, toTrModels, dataloaders, centers, label='init')  # label='init' | 'clf'
    # (6.1) Training (step1&2 are pre-training steps. Deprecated due to none contribution)
    pseudo_tracker = None
    if getattr(args, 'pseudo_analysis', False):
        out_sub = os.path.join(args.pseudo_analysis_dir, args.src + '2' + args.tgt)
        pseudo_tracker = PseudoLabelTracker(shr_threshold=args.shr_nc, output_dir=out_sub)
        y_true, clu = collect_pseudo_from_loader(dataloaders['te_loader_tgt'])
        init_m = evaluate_pseudo_labels(y_true, clu, shr_threshold=args.shr_nc)
        print('[Pseudo analysis] Step-1 init pseudo labels:', init_m)
        pseudo_tracker.evaluate_batch(y_true, clu, epoch=-1, phase='init_pseudo')

    if args.step3 == 'train':
        print('begin training step3')
        train_step3(args, dataloaders, toTrModels, nonTrModels, lossFunctions, optimizers, centers,
                    pseudo_tracker=pseudo_tracker)
    elif args.step3 == 'resume':
        print("Keep training step3!")
        resume_pretrained_weights(args, toTrModels, step='step3')
        train_step3(args, dataloaders, toTrModels, nonTrModels, lossFunctions, optimizers, centers,
                    pseudo_tracker=pseudo_tracker)
    elif args.step3 == 'load':
        print(" Step 3 trained weights already exits!!!")
        resume_pretrained_weights(args, toTrModels, step='step3')
    else:
        print(" Nothing to do with Step 3...")

    eval(args, -3, dataloaders['te_loader_tgt'], toTrModels, nonTrModels, centers)

    # log = open(os.path.join(args.log, 'test_log.txt'), 'a')
    # log.write('\n***   best val acc: %3f   ***' % best_prec1)
    # log.write('\n***   best test acc: %3f   ***' % best_test_prec1)
    # log.write('\n***   cond best test acc: %3f   ***' % cond_best_test_prec1)
    # # end time
    # log.write('\n-------------------------------------------\n')
    # log.write(time.asctime(time.localtime(time.time())))
    # log.write('\n-------------------------------------------\n')
    # log.close()
    #
    #
    #


def run_fig2_training(args, dataloaders, att_cents):
    """Figure-2: stage1 decoupling (Gc,Gd,D,C) + stage2 recovery (phi,psi,W)."""
    from fig2_networks import build_fig2_models
    from train_fig2 import train_fig2
    from pseudo_label_bank import PseudoLabelBank

    print("=" * 60)
    print("Method: Fig2 / Algorithm 1 (decoupling + semantic recovery)")
    print("E1:", args.stage1_epochs, "| E2:", args.stage2_epochs,
          "| phi_align:", getattr(args, "stage1_phi_align_epochs", 3),
          "| phi_warmup:", getattr(args, "phi_warmup_epochs", 5))
    if not getattr(args, "no_early_stop", False):
        print("Early stop: on | metric:", getattr(args, "early_stop_metric", "paper"),
              "| patience:", getattr(args, "early_stop_patience", 2))
    if getattr(args, "fixed_pseudo", True):
        print("Pseudo-labels: round-0 K-means on D_t^u (init_pseudo_i2awa.py), FIXED in training")
    else:
        print("Pseudo-labels: iterative update (eq.5 / classifier refresh)")
    print("=" * 60)

    pseudo_bank = None
    if getattr(args, "fixed_pseudo", True):
        from dataset_paths import pseudo_file
        import os
        ppath = pseudo_file(args)
        if not os.path.isfile(ppath):
            raise FileNotFoundError(
                "Fixed pseudo mode requires K-means CSV. Run: python init_pseudo_i2awa.py --k 10"
            )
    else:
        pseudo_bank = PseudoLabelBank(args)
        loaders = generate_dataloader(args, pseudo_bank=pseudo_bank)
        dataloaders = {
            "tr_loader_src": loaders[0],
            "tr_loader_tgt": loaders[1],
            "te_loader_tgt": loaders[2],
        }

    fig2_models = build_fig2_models(args)
    for name, m in fig2_models.items():
        if hasattr(m, "to"):
            fig2_models[name] = m.to(device)

    loss_fn = {
        "BCELoss": torch.nn.BCELoss().to(device),
        "CELoss": torch.nn.CrossEntropyLoss().to(device),
    }

    pseudo_tracker = None
    if getattr(args, "pseudo_analysis", False):
        out_sub = os.path.join(args.pseudo_analysis_dir, args.src + "2" + args.tgt)
        pseudo_tracker = PseudoLabelTracker(shr_threshold=args.shr_nc, output_dir=out_sub)
        y_true, clu = collect_pseudo_from_loader(dataloaders["te_loader_tgt"])
        init_m = evaluate_pseudo_labels(y_true, clu, shr_threshold=args.shr_nc)
        print("[Fig2] init pseudo labels:", init_m)
        pseudo_tracker.evaluate_batch(y_true, clu, epoch=-1, phase="init_pseudo")

    train_fig2(
        args, dataloaders, fig2_models, loss_fn, att_cents,
        pseudo_tracker=pseudo_tracker, pseudo_bank=pseudo_bank,
    )
    eval_fig2(args, -1, dataloaders["te_loader_tgt"], fig2_models, att_cents)


def run_offline_pseudo_analysis(args, dataloaders):
    """K-sensitivity + K-misspecification without Step-3 training."""
    out_dir = os.path.join(args.pseudo_analysis_dir, args.src + '2' + args.tgt)
    os.makedirs(out_dir, exist_ok=True)
    print('=' * 72)
    print('Offline pseudo-label analysis:', args.src, '->', args.tgt)
    print('=' * 72)
    try:
        feats, y_true, y_pseudo_init = load_target_arrays(args)
    except FileNotFoundError as e:
        print(e)
        print('Need feature/label CSVs under data/N2AwA/features/.')
        y_true, y_pseudo_init = collect_pseudo_from_loader(dataloaders['te_loader_tgt'])
        feats = None
        print('Using pseudo labels from dataloader; K-sweep needs features on disk.')
    init_m = evaluate_pseudo_labels(y_true, y_pseudo_init, shr_threshold=args.shr_nc)
    print('Init pseudo metrics:', init_m)
    k_sens_df, k_misspec_df = None, None
    if feats is not None:
        k_min = args.k_sens_min if args.k_sens_min is not None else max(2, args.tgt_nc - 7)
        k_max = args.k_sens_max if args.k_sens_max is not None else args.tgt_nc + 7
        k_values = list(range(k_min, k_max + 1, args.k_sens_step))
        print('K-sensitivity:', k_values)
        k_sens_df = run_k_sensitivity(feats, y_true, k_values, shr_nc=args.shr_nc, tgt_nc=args.tgt_nc)
        k_sens_df.to_csv(os.path.join(out_dir, 'k_sensitivity.csv'), index=False)
        plot_k_sensitivity(k_sens_df, os.path.join(out_dir, 'k_sensitivity.png'))
        k_misspec_df = run_k_misspecification(
            feats, y_true, true_k=args.tgt_nc, shr_nc=args.shr_nc, tgt_nc=args.tgt_nc,
            under_delta=args.k_misspec_under, over_deltas=args.k_misspec_over,
        )
        k_misspec_df.to_csv(os.path.join(out_dir, 'k_misspecification.csv'), index=False)
        plot_k_misspec(k_misspec_df, os.path.join(out_dir, 'k_misspecification.png'))
        print(k_misspec_df[['spec', 'K_used', 'cluster_acc', 'boundary_acc']].to_string(index=False))
    report = write_summary_report(out_dir, init_m, k_sens_df, k_misspec_df)
    print('Report:', report)


if __name__ == '__main__':
    main()


