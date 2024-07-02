import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 2  # 1
    training.n_iters = 400000
    training.snapshot_freq = 8000  # 10000#10000 #50000
    training.log_freq = 100
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = False
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 1
    evaluate.batch_size = 1
    evaluate.enable_sampling = False
    evaluate.num_samples = 10  # 1000 #50000
    evaluate.enable_loss = False
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "LSUN"
    data.image_size = 512
    data.random_flip = False  # True
    data.uniform_dequantization = False
    data.centered = False  ### inverse scale / get_data_scaler
    data.num_channels = 1
    data.num_channels_unet_input = 1
    # model
    ## train 378     test:0.6
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 0.6  ##300
    model.sigma_min = 0.01
    model.num_scales = 1500  # 1000 #2000####1000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.0
    model.embedding_type = "fourier"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 42
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
