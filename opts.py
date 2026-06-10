'''
* @name: opts.py
* @description: Hyperparameter configuration. Note: For hyperparameter settings, please refer to the appendix of the paper.
'''


import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',        
                 type=str,
                 default='mosi',
                 help='mosi, mosei or sims'),
            dict(name='--dataPath',
                 default="/mosi/aligned_50.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',     
                 default=[50, 50, 50],
                 type=list,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
           dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
            dict(name='--test_checkpoint',
                 default="",
                 type=str,
                 help=' '),
        ],
        'network': [
            dict(name='--CUDA_VISIBLE_DEVICES',        
                 default='0',
                 type=str),
            dict(name='--fusion_layer_depth',
                 default=1,
                 type=int),
            dict(name='--AHL_depth',
                 default=3,
                 type=int),
            dict(name='--single_modality_depth',
                 default=4,
                 type=int),
            dict(name='--mamba_type',
                 default='mamba', # or 'mamba2'
                 type=str),
            dict(name='--cross_modal_fusion',
                 default='text_guided_fusion',
                 # default=None,
                 type=str),
            dict(name='--fusion_depth',
                 default=1,
                 type=int),
            dict(name='--sub_loss',
                 default=True,
                 type=int),
            dict(name='--sub_loss_lambda',
                 default=0.5,
                 type=float),
            # Phase 1: Soft Ordinal Regression (SORD) head.
            # When enabled, the main prediction is decoded as the expectation
            # over the 7 ordinal levels {-3..+3} of the cls7 head, and the cls7
            # head is trained with a soft-label KL loss instead of hard CE.
            dict(name='--sord',
                 default=0,
                 type=int),
            dict(name='--sord_lambda',
                 default=1.0,
                 type=float),
            dict(name='--use_mlp',
                 default=False,
                 type=bool),
            dict(name='--use_con_loss',
                 default=False,
                 type=bool),
            dict(name='--con_loss_lambda',
                 default=0.5,
                 type=bool),
            dict(name='--use_roberta',
                 default=False,
                 type=bool),
        ],

        'common': [
            dict(name='--project_name',
                 default='MSAmba',
                 type=str
                 ),
            dict(name='--sm_block_type',
                 default='Block_GLCE',
                 type=str
                 ),
           dict(name='--is_test',    
                 default=1,
                 type=int
                 ),
            dict(name='--seed',  # try different seeds
                 default=1234,
                 type=int
                 ),
            dict(name='--models_save_root',
                 default='./checkpoint',
                 type=str
                 ),
            dict(name='--batch_size',
                 default=128,
                 type=int,
                 help=' '),
            dict(
                name='--n_threads',
                default=4,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(name='--lr',
                 type=float,
                 default=1e-5),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(
                name='--n_epochs',
                default=100,
                type=int,
                help='Number of total epochs to run',
            ),
            dict(
                name='--patience',
                default=20,
                type=int,
                help='Early stopping patience (epochs without valid improvement). 0 disables early stopping.',
            ),
            dict(
                name='--select_metric',
                default='MAE',
                type=str,
                help='Validation metric used for checkpoint selection / early stopping (e.g. MAE, Mult_acc_7, Non0_acc_2).',
            )
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args