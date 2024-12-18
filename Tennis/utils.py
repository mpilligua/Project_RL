import torch


def get_schedulers(config, optimizer):
    
    
    if config['type_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'], eta_min=config['eta_min'])
        
    elif config['type_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        
    elif config['type_scheduler'] == 'multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    else:
        raise ValueError("Scheduler not recognized")
    return scheduler

