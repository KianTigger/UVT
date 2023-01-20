import os
import Network
import argparse

# train = False
# test = True
# predict = False

# model_path="./output/UVT_M_CLA"
# model_path="./output/UVT_R_CLA"
# model_path="./output/UVT_M_REG"
# model_path="./output/UVT_R_REG"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, type=bool, help="Whether to train the model.")
    parser.add_argument("--test", default=False, type=bool, help="Whether to test the model.")
    parser.add_argument("--predict", default=False, type=bool, help="Whether to predict using the model.")
    parser.add_argument("--model_path", default="./output/UVT_R_REG", type=str, 
    help="The path to the model. options are './output/UVT_M_CLA', './output/UVT_R_CLA', './output/UVT_M_REG', './output/UVT_R_REG', or './output/UVT_repeat_reg' if you Train the model yourself.")
    parser.add_argument("--dataset_path", default="../../Datasets/Echonet-Dynamic", type=str, 
    help="The path to the dataset folder containing the 'Videos' foldes and 'FileList.csv' file.")
    args = parser.parse_args()

    # if train, test, or predict is not specified, then quit with message
    if not args.train and not args.test and not args.predict:
        print("Please specify whether to train, test, or predict")
        quit()

    if args.train:
        Network.train(  dataset_path=args.dataset_path,  # path to the dataset folder containing the "Videos" foldes and "FileList.csv" file
                    num_epochs=5,               # number of epoch to train
                    device=[0],                 # "cpu" or gpu ids, ex [0] or [0,1] or [2] etc
                    batch_size=2,               # batch size
                    seed=0,                     # random seed for reproducibility
                    run_test=False,             # run test loop after each epoch
                    lr = 1e-5,                  # learning rate
                    modelname="UVT_repeat_reg", # name of the folder where weight files will be stored
                    latent_dim=1024,            # embedding dimension
                    lr_step_period=3,           # number of epoch before dividing the learning rate by 10
                    ds_max_length = 128,        # maximum number of frame during training
                    ds_min_spacing = 10,        # minimum number of frame during training
                    DTmode = 'repeat',          # data preprocessing method: 'repeat' (mirroring) / 'full' (entire video) / 'sample' (single heartbeat with random amounf of additional frames)
                    SDmode = 'reg',             # SD branch network type: reg (regression) or cla (classification)
                    num_hidden_layers = 16,     # Number of Transformers
                    intermediate_size = 8192,   # size of the main MLP inside of the Transformers
                    rm_branch = None,           # select branch to not train: None, 'SD', 'EF'
                    use_conv = False,           # use convolutions instead of MLP for the regressors - worse results
                    attention_heads = 16        # number of attention heads in each Transformer
                    )
    
    # add a variable for SDmode depending on the model_path
    if args.model_path == "./output/UVT_M_CLA" or args.model_path == "./output/UVT_R_CLA":
        SDmode = "cla"
    elif args.model_path == "./output/UVT_M_REG" or args.model_path == "./output/UVT_R_REG" or args.model_path == "./output/UVT_repeat_reg":
        SDmode = "reg"
    else:
        print("Please specify a valid model_path")
        quit()

    if args.test:
        # Parameters must match train-time parameters, or the weight files wont load
        Network.test(   dataset_path=args.dataset_path,  # Path to the dataset folder containing the "Videos" foldes and "FileList.csv" file
                    SDmode=SDmode,               # SD branch network type: reg (regression) or cla (classification)
                    use_full_videos=True,       # Use full video (no preprocessing other than intensity scaling)
                    latent_dim=1024,            # embedding dimension
                    num_hidden_layers=16,       # Number of Transformers
                    intermediate_size=8192,     # Size of the main MLP inside of the Transformers
                    # model_path="./output/UVT_repeat_reg",# path of trained weight
                    model_path=args.model_path,      # path of trained weight
                    device=[0]
                    )

    if args.predict:
        # Parameters must match train-time parameters, or the weight files wont load
        Network.predictEDES(   
                    dataset_path=args.dataset_path,  # Path to the dataset folder containing the "Videos" foldes and "FileList.csv" file
                    SDmode=SDmode,               # SD branch network type: reg (regression) or cla (classification)
                    use_full_videos=True,       # Use full video (no preprocessing other than intensity scaling)
                    latent_dim=1024,            # embedding dimension
                    num_hidden_layers=16,       # Number of Transformers
                    intermediate_size=8192,     # Size of the main MLP inside of the Transformers
                    model_path=args.model_path,      # path of trained weight
                    device=[0]
                    )
