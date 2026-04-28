import torch


def setup_device(args) -> torch.device:
    """Detect best available device, set args.device, and seed appropriately."""
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available() and not args.disable_cuda:
        args.device = torch.device('mps')
        torch.mps.manual_seed(args.seed)
    else:
        args.device = torch.device('cpu')
    return args.device
