import yaml

sweep_config = {'method': 'grid'}
metric = {'goal' : 'minimize' , 
          'name' : 'best_val_loss'}
parameters = {
    'n_experiments': {
        'values' : [1]
    },
    'split': {
        'values' : ['PublicSplit']
    },
    'dataset': {
        'values' : ['Cora']
    },
    'epochs': {
        'values' : [20, 50, 100]
    },
    'n_layers': {
        'values' : [1, 2]
    },
    'channels': {
        'values' : [128, 256, 512, 1024]
    },
    'tau': {
        'values' : [0.5]
    },
    'lr1': {
        'values' : [5e-4, 1e-3, 5e-3, 1e-2]
    },
    'lr2': {
        'values' : [1e-3, 5e-3, 1e-2]
    },
    'wd1': {
        'values' : [1e-4, 1e-3, 1e-2]
    },
    'wd2': {
        'values' : [1e-4, 1e-3, 1e-2]
    },
    'edr': {
        'values' : [0,0.1,0.2,0.3,0.4,0.5]
    },
    'fmr': {
        'values' : [0,0.1,0.2,0.3,0.4,0.5]
    },
}         
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters

with open(r'config.yaml', 'w') as file:
    documents = yaml.dump(sweep_config, file)
