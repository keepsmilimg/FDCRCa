import argparse
import os
import torch

def _default_n2awa():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'N2AwA') + os.sep


def _default_i2awa():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'I2AwA') + os.sep


def opts():
    parser = argparse.ArgumentParser(description='Semantic Recovery for OpenSet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='I2AwA', choices=['N2AwA', 'I2AwA'])
    _i2 = _default_i2awa()
    parser.add_argument('--data_path_source', type=str, default=_i2, help='root of source training set')
    parser.add_argument('--data_path_target', type=str, default=_i2, help='root of target training set')
    parser.add_argument('--src', type=str, default='3D2', help='source training set')
    parser.add_argument('--tgt', type=str, default='AwA2', help='target training set')
    parser.add_argument('--src_nc', type=int, default=10, help='src class number')
    parser.add_argument('--tgt_nc', type=int, default=17, help='tar class number')
    parser.add_argument('--shr_nc', type=int, default=10, help='src class number')
    parser.add_argument('--unk_nc', type=int, default=7, help='tar class number')

    # data specification
    parser.add_argument('--init_prot_type', type=str, default='sample', help='Type of init prototype classifcation, [center | sample]')
    parser.add_argument('--binary', type=bool, default=True, help='Binary the predicted or reconstructed attributes or not')
    parser.add_argument('--att_type', type=str, default='binary', help='binary or continuous attributes')
    parser.add_argument('--src_soft_select', action='store_true', help='whether to softly select source instances')
    parser.add_argument('--src_hard_select', action='store_true', help='whether to hardly select source instances')
    parser.add_argument('--src_mix_weight', action='store_true', help='whether to mix 1 and soft weight')
    parser.add_argument('--tao_param', type=float, default=0.5, help='threshold parameter of cosine similarity')
    # general optimization options
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--no_da', action='store_true', help='whether to not use data augmentation')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_plan', type=str, default='dao', help='learning rate decay plan of step or dao')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    # specific optimization options
    parser.add_argument('--ao', action='store_true', help='whether to use alternative optimization')
    parser.add_argument('--cluster_method', type=str, default='kmeans', help='clustering method of kmeans or spherical_kmeans or kernel_kmeans to choose')
    parser.add_argument('--cluster_iter', type=int, default=5, help='number of iterations of K-means')
    parser.add_argument('--cluster_kernel', type=str, default='rbf', help='kernel to choose when using kernel K-means')
    parser.add_argument('--gamma', type=float, default=None, help='bandwidth for rbf or polynomial kernel when using kernel K-means')
    parser.add_argument('--sample_weight', action='store_true', help='whether to adapt sample weight when using kernel K-means')
    parser.add_argument('--initial_cluster', type=int, default=1, help='target or source class centroids for initialization of K-means')
    parser.add_argument('--init_cen_on_st', action='store_true', help='whether to initialize learnable cluster centers on both source and target instances')
    parser.add_argument('--src_cen_first', action='store_true', help='whether to use source class centroids as initial target cluster centers at the first epoch')
    parser.add_argument('--src_cls', action='store_true', help='whether to classify source instances when clustering target instances')
    parser.add_argument('--src_fit', action='store_true', help='whether to use convex combination of true label vector and predicted label vector as training guide')
    parser.add_argument('--src_pretr_first', action='store_true', help='whether to perform clustering over features extracted by source pre-trained model at the first epoch')
    parser.add_argument('--learn_embed', action='store_true', help='whether to apply embedding clustering')
    parser.add_argument('--no_second_embed', action='store_true', help='whether to not apply embedding clustering on output features of the first FC layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='degrees of freedom of Student\'s t-distribution')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of auxiliary target distribution or assigned cluster labels')
    parser.add_argument('--embed_softmax', action='store_true', help='whether to use softmax to normalize soft cluster assignments for embedding clustering')
    parser.add_argument('--div', type=str, default='kl', help='measure of prediction divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--gray_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and augmented images on the target domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian for data augmentation operation of blurring')
    # checkpoints
    parser.add_argument('--resume', type=str, default='', help='checkpoints path to resume')
    parser.add_argument('--log', type=str, default='./checkpoints/N2AwA/', help='log folder')
    parser.add_argument('--stop_epoch', type=int, default=200, metavar='N', help='stop epoch for early stop (default: 200)')
    # architecture
    parser.add_argument('--GenA_type', type=str, default='FC', help='GenA type:[FC, MulDis, GCN]')
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--num_neurons', type=int, default=128, help='number of neurons of fc1')
    parser.add_argument('--step1', type=str, default='train', help='whether to train step 1') # 'train' | 'load' | 'resume'
    parser.add_argument('--step2', type=str, default='train', help='whether to train step 2')
    parser.add_argument('--step3', type=str, default='train', help='whether to train step 3')
    parser.add_argument('--combine_za', type=bool, default=True, help='Whether or not combine z and att to Clf')
    # Fig2 two-stage method (feature decoupling + semantic recovery)
    parser.add_argument('--method', type=str, default='fig2', choices=['srosda', 'fig2'],
                        help='srosda=original Step3; fig2=Gc/Gd/D/C + phi/psi/W')
    parser.add_argument('--paper_mode', action='store_true', default=True,
                        help='Fig2: paper losses (C_t clf, ArcFace W, eq.19)')
    parser.add_argument('--no_paper_mode', action='store_true',
                        help='legacy 41-class open-bucket head')
    parser.add_argument('--no_eq5_pseudo', action='store_true',
                        help='disable paper eq.5 adaptive-threshold pseudo-labels')
    parser.add_argument('--log_domain_d_acc', action='store_true',
                        help='log D accuracy on G_c each Stage-2 epoch (Figure domain-invariance)')
    parser.add_argument('--stage2_only', action='store_true',
                        help='skip Stage-1 (e.g. load weights and run Stage-2 only)')
    parser.add_argument('--domain_d_acc_csv', type=str, default='',
                        help='CSV path for domain-D accuracy log (auto if empty)')
    parser.add_argument('--pseudo_update', action='store_true',
                        help='iteratively refresh pseudo-labels (eq.5/W); default: fixed K-means CSV only')
    parser.add_argument('--lambda_r', type=float, default=0.1, help='eq.19 L_R weight')
    parser.add_argument('--lambda_au', type=float, default=0.5, help='eq.19 L_Au weight')
    parser.add_argument('--w_clfsu', type=float, default=1.5,
                        help='ClfSU shared/unknown gate loss (paper Stage2)')
    parser.add_argument('--w_unk_repulse', type=float, default=0.5,
                        help='push unknown z_c away from known prototypes')
    parser.add_argument('--unk_repulse_margin', type=float, default=0.3,
                        help='min L2 distance margin for unk_repulsion')
    parser.add_argument('--arc_unk_sample_weight', type=float, default=3.0,
                        help='ArcFace CE sample weight on pseudo unknown classes')
    parser.add_argument('--paper_use_clfsu', action='store_true', default=True,
                        help='train/eval ClfSU gate in paper mode (not W>=C_s only)')
    parser.add_argument('--no_paper_clfsu', action='store_true',
                        help='disable ClfSU; use W>=C_s open gate only')
    parser.add_argument('--open_gate', type=str, default='clfsu',
                        choices=['clfsu', 'w', 'hybrid'],
                        help='open-set gate: ClfSU, W threshold, or hybrid')
    parser.add_argument('--no_os_ou_cluster', action='store_true',
                        help='OS/OU/S/U: use per-class diagonal; default is Hungarian cluster ACC (40 known / 10 unknown)')
    parser.add_argument('--lambda_arc', type=float, default=1.0, help='eq.19 L_Arc-reg weight')
    parser.add_argument('--lambda_cc', type=float, default=0.1, help='eq.19 L_cc weight')
    parser.add_argument('--arc_s', type=float, default=30.0, help='ArcFace scale s')
    parser.add_argument('--arc_m', type=float, default=0.5, help='ArcFace margin m')
    parser.add_argument('--stage1_epochs', type=int, default=15,
                        help='Fig2 stage-1 epochs (feature decoupling)')
    parser.add_argument('--stage2_epochs', type=int, default=50,
                        help='Fig2 stage-2 epochs (semantic recovery)')
    parser.add_argument('--w_clf', type=float, default=2.0,
                        help='weight for open-set classifier W (stage2)')
    parser.add_argument('--w_phi', type=float, default=2.0,
                        help='weight for source attribute recovery loss_phi_s (stage2)')
    parser.add_argument('--open_class_weight', type=float, default=0.25,
                        help='CE class weight for open bucket index C_s (41-way head)')
    parser.add_argument('--phi_warmup_epochs', type=int, default=5,
                        help='Stage2 epochs: train Gc,phi,psi only before joint W')
    parser.add_argument('--stage1_phi_align_epochs', type=int, default=3,
                        help='after Stage1: extra epochs aligning Phi on source GT attributes')
    parser.add_argument('--stage2_w_dom', type=float, default=0.05,
                        help='domain adversarial on Gc during Stage2 joint training')
    parser.add_argument('--w_dom', type=float, default=0.5, help='domain adversarial weight')
    parser.add_argument('--w_decouple', type=float, default=0.3, help='Gd vs C_dec weight')
    parser.add_argument('--w_att', type=float, default=0.1, help='target attribute recovery weight')
    parser.add_argument('--w_psi', type=float, default=0.2, help='psi relational loss weight')
    parser.add_argument('--w_center_rel', type=float, default=0.05, help='mu^u vs mu^s center loss')
    parser.add_argument('--unfreeze_gd_stage2', action='store_true',
                        help='train G_d in Stage 2 (default: frozen per Alg.1 output)')
    parser.add_argument('--pseudo_update_stage2', action='store_true',
                        help='also refresh pseudo-labels each Stage-2 epoch (off = Alg.1)')
    parser.add_argument('--eval_freq', type=int, default=5, help='evaluate every N epochs per stage')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='disable early stopping on Stage2 (default: early stop on)')
    parser.add_argument('--early_stop_patience', type=int, default=2,
                        help='stop after N evals without improvement (eval every eval_freq epochs)')
    parser.add_argument('--early_stop_metric', type=str, default='H2',
                        choices=['H2', 'H1', 'OS', 'OU', 'paper'],
                        help='metric for early stop / best checkpoint')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.002,
                        help='minimum improvement to reset patience (metric in [0,1])')
    parser.add_argument('--early_stop_max_loss', type=float, default=350.0,
                        help='abort Stage2-joint if epoch loss exceeds this')
    parser.add_argument('--fixed_pseudo', action='store_true',
                        help='(default ON) keep round-0 K-means pseudo labels, no update during training')
    # pseudo-label analysis (Step 3 / paper experiments)
    parser.add_argument('--pseudo_analysis', action='store_true',
                        help='track pseudo-label ACC each epoch during Step 3 training')
    parser.add_argument('--pseudo_analysis_only', action='store_true',
                        help='run K-sensitivity & K-misspec analysis then exit (no Step 3 training)')
    parser.add_argument('--pseudo_analysis_dir', type=str, default='./results/pseudo_analysis/',
                        help='output directory for pseudo-label analysis')
    parser.add_argument('--k_sens_min', type=int, default=None, help='K-sensitivity min K (default: tgt_nc-7)')
    parser.add_argument('--k_sens_max', type=int, default=None, help='K-sensitivity max K (default: tgt_nc+7)')
    parser.add_argument('--k_sens_step', type=int, default=1, help='K-sensitivity step')
    parser.add_argument('--k_misspec_under', type=int, default=3, help='K_under = tgt_nc - this')
    parser.add_argument('--k_misspec_over', type=int, nargs='+', default=[3, 7],
                        help='K_over = tgt_nc + each value')
    parser.add_argument('--tgt_clu_override', type=str, default='',
                        help='override target pseudo-label CSV for misspec experiments')
    parser.add_argument('--unk_k', type=int, default=None,
                        help='I2AwA: preset unknown-class count K (unk_nc=K, tgt_nc=C_s+K, tag 3D22AwA2_KK)')
    # i/o
    parser.add_argument('--print_freq', type=int, default=10, metavar='N', help='print frequency (default: 10)')

    parser.set_defaults(fixed_pseudo=True, no_eq5_pseudo=True)
    args = parser.parse_args()
    if getattr(args, "no_paper_clfsu", False):
        args.paper_use_clfsu = False
    if getattr(args, "pseudo_update", False):
        args.fixed_pseudo = False
        args.no_eq5_pseudo = False
    args.pretrained = True

    if args.dataset == 'I2AwA':
        args.src_nc = 40
        args.shr_nc = 40
        if args.unk_k is not None:
            args.unk_nc = int(args.unk_k)
            args.tgt_nc = args.shr_nc + args.unk_nc
        else:
            args.tgt_nc = 50
            args.unk_nc = 10
    else:
        args.src_nc = args.src_nc
        args.tgt_nc = args.tgt_nc
        args.shr_nc = args.shr_nc
        args.unk_nc = args.unk_nc

    args.src_cls = True
    args.src_cen_first = True
    args.learn_embed = True
    args.embed_softmax = True
    args.log = args.log + '_' + args.src + '2' + args.tgt + '_bs' + str(args.batch_size) + '_lr' + str(args.lr)
    if not getattr(args, "domain_d_acc_csv", ""):
        from dataset_paths import fig2_run_tag
        args.domain_d_acc_csv = os.path.join(
            "./results",
            args.att_type,
            "fig2",
            args.dataset,
            fig2_run_tag(args),
            "stage2_domain_d_acc.csv",
        )

    return args
