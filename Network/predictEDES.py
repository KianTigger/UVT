from matplotlib.legend import Legend
import os
import numpy as np
import math
import torch
from torch._C import dtype
import tqdm
import scipy.signal
from scipy.interpolate import interp1d
import pandas as pd
import sklearn.metrics
from PIL import Image


import matplotlib.pyplot as plt

from Network.model import get_model
from Network.dataloader import EchoSet

import torch
import tqdm

def predictEDES(dataset_path,
         SDmode='reg',
         DTmode='repeat',
         use_full_videos=False,
         latent_dim=1024,
         fixed_length=128,
         num_hidden_layers=16,
         intermediate_size=8192,
         rm_branch=None,
         use_conv=False,
         attention_heads=16,
         model_path=None,
         device=[0]
         ):

    print("Predicting on", SDmode, DTmode, use_full_videos)

    np.random.seed(0)
    torch.manual_seed(0)

    # SDmode = 'reg' #cla, reg
    # Dtmode = 'repeat' #repeat, full
    if use_full_videos:
        dsdtmode = 'full'
    else:
        dsdtmode = DTmode

    if not os.path.exists(dataset_path):
        raise ValueError(dataset_path+" does not exist.")

    destination_folder = model_path

    os.makedirs(os.path.join(destination_folder, "figs"), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(device) == type(list()):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in device)
        device = "cuda"
    device = torch.device(device)
    print("Using device:", device)

    # Load model
    best = torch.load(os.path.join(model_path, "best.pt"))
    model = get_model(latent_dim, img_per_video=1, SDmode=SDmode, num_hidden_layers=num_hidden_layers,
                      intermediate_size=intermediate_size, rm_branch=rm_branch, use_conv=use_conv, attention_heads=attention_heads)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(model.__class__.__name__, "contains",
          pytorch_total_params, "parameters.")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(best['state_dict'])
    model.cuda()
    model.eval()

    # Load data
    dataset = EchoSet(dataset_path, split="all", min_spacing=10, max_length=fixed_length,
                      fixed_length=fixed_length, pad=8, random_clip=False, 
                      dataset_mode=dsdtmode, SDmode=SDmode, train=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    device = torch.device('cuda')

    mean_dist_coef = 0.3

    limit = -10
    limiter = 0
    # plot_graph = False
    # save_graph = False
    mirror_vid = False  # worsen results

    broken = 0

    phase_predictions = []
    count = 0

    with torch.no_grad():
        with tqdm.tqdm(total=len(loader)) as pbar:
            for (filename, video) in loader:
                count += 1
            
                nB, nF, nC, nH, nW = video.shape

                # Merge batch and frames dimension
                if mirror_vid:
                    fps = 0
                    for i in range(video.size(0)):
                        fps += video[i].shape[0]
                    fps = fps / video.size(0)
                    offset = fps
                    video = video[0]  # squeeze batch dimension
                    video = torch.cat((video.flip(0)[-offset:-1, :, :, :], video, video.flip(0)[
                                      1:offset, :, :, :]), dim=0)  # mirror start and end
                    nF, nC, nH, nW = video.shape  # update nF
                if not(mirror_vid):
                    video = torch.cat(
                        ([video[i] for i in range(video.size(0))]), dim=0)
                video = video.to(device, dtype=torch.float)

                # AE.encode -> Transformer +-> class_vec
                #                          L-> ef_pred
                class_vec, ef_pred = model(video, nB, nF)
                if rm_branch != 'SD':
                    class_vec = class_vec.squeeze().cpu()
                if rm_branch != 'EF':
                    ef_pred = ef_pred.cpu().item()
                # fps = fps.item()

                if SDmode == 'reg' and dsdtmode == 'full':

                    if rm_branch != 'SD':
                        smooth_vec = smooth(class_vec, window=5, rep=3)

                        zero_crossing = ((smooth_vec.sign().roll(
                            1) - smooth_vec.sign()) != 0).to(dtype=torch.long)
                        zero_crossing[0] = 0

                        # Find heart phases and clean noise

                        peak_indices = torch.where(zero_crossing == 1)[0]
                        if peak_indices.shape[0] < 3:
                            zero_crossing[0] = 0
                            zero_crossing[-1] = 1
                        else:
                            peak_dist = (peak_indices-peak_indices.roll(1))[1:]
                            mean_dist = peak_dist.to(torch.float).mean()
                            while ((peak_dist[peak_dist < mean_dist*mean_dist_coef] > 0).sum() > 0).item():
                                bad_peaks = torch.where(
                                    peak_dist < mean_dist*mean_dist_coef)[0][0].item()
                                bp1 = peak_indices[bad_peaks].item()
                                bp2 = peak_indices[bad_peaks+1].item()
                                new_peak = int((bp1+bp2)/2)

                                peak_indices = torch.cat((peak_indices[:bad_peaks], torch.tensor(
                                    [new_peak]), peak_indices[bad_peaks+2:]), axis=0)
                                peak_dist = (
                                    peak_indices-peak_indices.roll(1))[1:]

                            peak_class = []
                            peak_intensity = []
                            peak_index = []
                            for i in range(peak_indices.shape[0]-1):
                                peak_class.append(
                                    np.sign(class_vec[peak_indices[i]:peak_indices[i+1]].numpy().mean()))
                                peak_index.append(peak_indices[i] + np.argmax(
                                    class_vec[peak_indices[i]:peak_indices[i+1]].numpy() * peak_class[-1]))
                                peak_intensity.append(
                                    peak_class[-1] * class_vec[peak_index[-1]])

                            if mirror_vid:
                                video = video[offset-1:-offset+1]
                                class_vec = class_vec[offset-1:-offset+1]

                                peak_index = torch.tensor(peak_index)
                                peak_class = torch.tensor(peak_class)
                                peak_intensity = torch.tensor(peak_intensity)

                                peak_class = peak_class[(peak_index > (
                                    offset-1)).logical_and(peak_index < (nF-offset+1))].numpy()
                                peak_intensity = peak_intensity[(peak_index > (
                                    offset-1)).logical_and(peak_index < (nF-offset+1))].numpy()
                                peak_index = peak_index[(peak_index > (
                                    offset-1)).logical_and(peak_index < (nF-offset+1))].numpy()
                                peak_index = peak_index-(offset-1)

                            # reg mode requites i.item() to convert from tensor to int
                            ED_predictions = [i.item() for c, i in zip(
                                peak_class, peak_index) if c == -1]
                            ES_predictions = [i.item() for c, i in zip(
                                peak_class, peak_index) if c == 1]
                            # phase_predictions.append((count, filename[0], "ED", ED_predictions))
                            # phase_predictions.append((count, filename[0], "ES", ES_predictions))
                            phase_predictions.append([filename[0], ED_predictions, ES_predictions])


                    # if plot_graph or save_graph:
                    #     plt.plot(-class_vec.numpy().reshape(-1),
                    #              label='Network pred', color='b')
                    #     es_label = True
                    #     ed_label = True
                    #     for i in range(len(peak_class)):
                    #         if peak_class[i] == 1 and ed_label:
                    #             plt.axvline(x=large_label, color='m',
                    #                         label='ES label', linewidth=3)
                    #             plt.axvline(x=peak_index[i], color='g' if peak_class[i] == 1 else 'r',
                    #                         label='ES preds' if peak_class[i] == 1 else 'ED preds')
                    #             ed_label = False
                    #         elif peak_class[i] == -1 and es_label:
                    #             plt.axvline(x=small_label, color='y',
                    #                         label='ED label', linewidth=3)
                    #             plt.axvline(x=peak_index[i], color='g' if peak_class[i] == 1 else 'r',
                    #                         label='ES preds' if peak_class[i] == 1 else 'ED preds')
                    #             es_label = False
                    #         else:
                    #             plt.axvline(
                    #                 x=peak_index[i], color='g' if peak_class[i] == 1 else 'r', )
                    #     plt.legend()
                    #     plt.title(filename[0])
                    #     plt.xlabel('Time axis (frames)')
                    #     plt.ylabel('Network prediction')
                    #     if plot_graph:
                    #         plt.show()
                    #     if save_graph:
                    #         plt.savefig(os.path.join(
                    #             destination_folder, "figs", filename[0]+'.pdf'), format='pdf')
                    #         plt.clf()

                elif SDmode == 'reg' and dsdtmode == 'repeat':

                    if rm_branch != 'SD':

                        # label = label[0]

                        smooth_vec = smooth(class_vec, window=5, rep=1)

                        zero_crossing = ((smooth_vec.sign().roll(
                            1) - smooth_vec.sign()) != 0).to(dtype=torch.long)
                        zero_crossing[0] = 0

                        # Find heart phases and clean noise
                        peak_indices = torch.where(zero_crossing == 1)[0]
                        
                        

                        peak_class = []
                        peak_intensity = []
                        peak_index = []
                        for i in range(peak_indices.shape[0]-1):
                            peak_class.append(
                                int(np.sign(class_vec[peak_indices[i]:peak_indices[i+1]].numpy().mean())))
                            peak_index.append(
                                int((
                                    np.arange(peak_indices[i], peak_indices[i+1]) *
                                    (class_vec[peak_indices[i]:peak_indices[i+1]].abs() /
                                     class_vec[peak_indices[i]:peak_indices[i+1]].abs().sum()).numpy()).sum().round())
                            )
                            peak_intensity.append(
                                peak_class[-1] * class_vec[peak_index[-1]])
                        
                        # reg mode requites i.item() to convert from tensor to int
                        ED_predictions = [i.item() for c, i in zip(
                            peak_class, peak_index) if c == -1]
                        ES_predictions = [i.item() for c, i in zip(
                            peak_class, peak_index) if c == 1]
                        # phase_predictions.append((count, filename[0], "ED", ED_predictions))
                        # phase_predictions.append((count, filename[0], "ES", ES_predictions))
                        phase_predictions.append([filename[0], ED_predictions, ES_predictions])



                    # if plot_graph:
                    #     plt.plot(class_vec.numpy().reshape(-1),
                    #              label='class pred')
                    #     plt.plot(label.numpy().reshape(-1),
                    #              label='class label')
                    #     for i in range(len(peak_class)):
                    #         plt.axvline(
                    #             x=peak_index[i], color='g' if peak_class[i] == 1 else 'b')
                    #     plt.legend()
                    #     plt.show()

                elif SDmode == 'cla' and dsdtmode == 'full':

                    # label = label.squeeze()  # [128,]
                    # Prepare ground truth
                    # small_label = torch.where(label == 1)[0][0].item()
                    # large_label = torch.where(label == 2)[0][0].item()

                    class_diff = class_vec[:, 1]-class_vec[:, 2]
                    zero_crossing = ((class_diff.sign().roll(
                        1) - class_diff.sign()) != 0).to(dtype=torch.long)
                    zero_crossing[0] = 0

                    # Find heart phases and clean noise
                    peak_indices = torch.where(zero_crossing == 1)[0]
                    if peak_indices.shape[0] >= 3:
                        peak_dist = (peak_indices-peak_indices.roll(1))[1:]
                        mean_dist = peak_dist.to(torch.float).mean()
                        while ((peak_dist[peak_dist < mean_dist*mean_dist_coef] > 0).sum() > 0).item():
                            bad_peaks = torch.where(
                                peak_dist < mean_dist*mean_dist_coef)[0][0].item()
                            bp1 = peak_indices[bad_peaks].item()
                            bp2 = peak_indices[bad_peaks+1].item()
                            new_peak = int((bp1+bp2)/2)

                            peak_indices = torch.cat((peak_indices[:bad_peaks], torch.tensor(
                                [new_peak]), peak_indices[bad_peaks+2:]), axis=0)
                            peak_dist = (peak_indices-peak_indices.roll(1))[1:]

                        peak_class = []
                        peak_intensity = []
                        peak_index = []
                        for i in range(peak_indices.shape[0]-1):
                            peak_class.append(1 if (class_vec[peak_indices[i]:peak_indices[i+1], 1].mean(
                            ) > class_vec[peak_indices[i]:peak_indices[i+1], 2].mean()) else 2)
                            peak_index.append(
                                int((
                                    np.arange(peak_indices[i], peak_indices[i+1]) *
                                    (class_vec[peak_indices[i]:peak_indices[i+1], peak_class[-1]].abs() /
                                     class_vec[peak_indices[i]:peak_indices[i+1], peak_class[-1]].abs().sum()).numpy()).sum().round())
                            )
                            peak_intensity.append(class_vec[peak_index[-1]])
                        
                        ED_predictions = [i for c, i in zip(
                            peak_class, peak_index) if c == 1]
                        ES_predictions = [i for c, i in zip(
                            peak_class, peak_index) if c == 2]
                        # phase_predictions.append((count, filename[0], "ED", ED_predictions))
                        # phase_predictions.append((count, filename[0], "ES", ES_predictions))
                        phase_predictions.append([filename[0], ED_predictions, ES_predictions])


                elif SDmode == 'cla' and dsdtmode == 'repeat':

                    # class_vec [128, 3]
                    # label = label.squeeze()  # [128,]
                    class_diff = class_vec[:, 1]-class_vec[:, 2]
                    zero_crossing = ((class_diff.sign().roll(
                        1) - class_diff.sign()) != 0).to(dtype=torch.long)
                    zero_crossing[0] = 0

                    peak_indices = torch.where(zero_crossing == 1)[0]
                    
                    # mean_dist = peak_dist.to(torch.float).mean()

                    # label[:peak_indices[0]] = 0
                    # label[peak_indices[-1]+1:] = 0

                    peak_class = []
                    peak_intensity = []
                    peak_index = []

                    for i in range(peak_indices.shape[0]-1):
                        peak_class.append(1 if (class_vec[peak_indices[i]:peak_indices[i+1], 1].mean(
                        ) > class_vec[peak_indices[i]:peak_indices[i+1], 2].mean()) else 2)
                        peak_index.append(
                            int((
                                np.arange(peak_indices[i], peak_indices[i+1]) *
                                (class_vec[peak_indices[i]:peak_indices[i+1], peak_class[-1]].abs() /
                                 class_vec[peak_indices[i]:peak_indices[i+1], peak_class[-1]].abs().sum()).numpy()).sum().round())
                        )
                        peak_intensity.append(class_vec[peak_index[-1]])

                    ED_predictions = [i for c, i in zip(
                        peak_class, peak_index) if c == 1]
                    ES_predictions = [i for c, i in zip(
                        peak_class, peak_index) if c == 2]
                    # phase_predictions.append((count, filename[0], "ED", ED_predictions))
                    # phase_predictions.append((count, filename[0], "ES", ES_predictions))
                    phase_predictions.append([filename[0], ED_predictions, ES_predictions])


                else:
                    broken += 1
                    print("Rejected", filename[0])

                limiter += 1
                if limiter == limit:
                    break

                pbar.update()

    # create a dataframe from final_predictions
    df = pd.DataFrame(phase_predictions, columns=["FileName", "ED Prediction", "ES Prediction"])

    # save the dataframe to a CSV file
    df.to_csv("PhasesPredictionsList.csv", index=False)

    n = 1
    while os.path.exists(os.path.join(destination_folder, f'phase_detection{n}.csv')):
        n += 1
    summary = pd.DataFrame(data=phase_predictions)
    summary.to_csv(os.path.join(destination_folder, f'phase_detection{n}.csv'), header=False, index=False)
    print('Saved to', os.path.join(destination_folder, f'phase_detection{n}.csv'))
    
    print("Rejected:", broken)

    return phase_predictions
    

def smooth(vec, window=5, rep=1):
    weight = torch.ones((1, 1, window))/window
    for _ in range(rep):
        pad = int((window-1)/2)
        vec = vec.unsqueeze(0).unsqueeze(0)
        vec = torch.nn.functional.conv1d(
            vec, weight, bias=None, stride=1, padding=pad, dilation=1, groups=1).squeeze()
    return vec

