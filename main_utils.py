import os
import random
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy as m

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
from PIL import Image
from io import BytesIO
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.utils.track4d_utils import (
    filter_moving_boxes_det,
    get_bbx_param,
    get_gt_flow_new,
    obj_centre,
    filter_object_points,
    map_gt_objects,
)
from PIL import Image
from losses import *
from models.track4d import Affinity
from vod.configuration.file_locations import VodTrackLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.transformations import FrameTransformMatrix
from csv import DictWriter
import logging


def save_json_list_to_csv(
    json_list: list[dict], filename: str, mode: str = "a"
) -> None:
    """Save data to a CSV file.

    Args:
        json_list (list): List of dictionaries containing data to be saved.
        filename (str): Name of the CSV file to save.
        mode (str): Mode to open the CSV file. Default is 'a' (append).
    """
    if not json_list:
        logging.warning("No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fieldnames = set().union(*(json_dict.keys() for json_dict in json_list))

    if not os.path.exists(filename):
        mode = "w"

    with open(filename, mode=mode, newline="", encoding="utf-8") as file:
        dict_writer = DictWriter(file, fieldnames=fieldnames)
        if mode == "w":
            dict_writer.writeheader()
        dict_writer.writerows(json_list)


def train_one_epoch(args, net, train_loader, opt, mode, ep):
    if mode == "train":
        net.train()
        aff_net = Affinity().cuda()
        aff_net.train()
    elif mode == "val":
        net.eval()
        aff_net = Affinity().cuda()
        aff_net.eval()

    num_examples, total_loss, loss_items, trk_met, seg_met, flow_met = epoch(
        args, net, train_loader, opt=opt, mode=mode, ep_num=ep
    )

    total_loss = total_loss * 1.0 / num_examples

    for l in loss_items:
        for i in range(len(loss_items[l])):
            loss_items[l][i] = loss_items[l][i]
        loss_items[l] = np.mean(np.array(loss_items[l]))
    for key in seg_met.keys():
        seg_met[key] = seg_met[key] / num_examples
    print("segmentation: ", seg_met)
    folder_results = f"./artifacts/metrics/"

    # TODO:NT: add a timestamp key to seg_met and flow_met
    seg_met["timestamp"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    flow_met["timestamp"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_json_list_to_csv(
        [seg_met], os.path.join(folder_results, "train-segmentation-metrics.csv")
    )
    for key in flow_met.keys():
        flow_met[key] = flow_met[key] / num_examples
    print("scene flow: ", flow_met)
    save_json_list_to_csv(
        [flow_met], os.path.join(folder_results, "train-scene-flow-metrics.csv")
    )

    return total_loss, loss_items


def epoch(
    args, net, train_loader, ep_num=None, opt=None, mode="train", display_images=True
):
    timestamp_folder = datetime.now().strftime("%Y-%m-%d--%H-%M")
    folder_results = f"./artifacts/{timestamp_folder}/results/"
    folder_results_vis = f"./artifacts/{timestamp_folder}/results_vis/"
    num_examples = 0
    total_loss = 0

    if args.model == "track4d_radar":
        loss_items = {
            "Loss": [],
            "SceneFlowLoss": [],
            "SegLoss": [],
            "TrackingLoss": [],
        }
    else:
        raise NotImplementedError

    objects_prev = dict()
    mappings_prev = dict()
    h = None

    trk_met = dict()
    seg_met = dict()
    flow_met = dict()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    results = {}

    for i, data in pbar:
        if args.model == "track4d_radar":
            (
                pc1,
                pc2,
                ft1,
                ft2,
                pc1_compensated,
                index,
                seq,
                ego_motion,
                pc_last_lidar,
                pc0_lidar,
                pc1_lidar,
                is_new_seq,
            ) = data

            if is_new_seq:
                objects_prev = dict()
                lives_obj_prev = dict()
                mappings_prev = dict()
                h = None

            pc1 = pc1.permute(0, 2, 1)[:, :3, :].cuda()
            pc2 = pc2.permute(0, 2, 1)[:, :3, :].cuda()
            ft1 = ft1.permute(0, 2, 1)[:, 0:2, :].cuda()
            ft2 = ft2.permute(0, 2, 1)[:, 0:2, :].cuda()
            pc1_compensated = pc1_compensated.permute(0, 2, 1)[:, :3, :].cuda()
            index = int(index)

            kitti_locations = VodTrackLocations(
                root_dir=args.dataset_path,
                output_dir=args.dataset_path,
                frame_set_path="",
                pred_dir="",
            )

            frame_data_0 = FrameDataLoader(
                kitti_locations=kitti_locations, frame_number=str(index + 1).zfill(5)
            )
            frame_data_1 = FrameDataLoader(
                kitti_locations=kitti_locations, frame_number=str(index).zfill(5)
            )

            try:
                import dataset_classes.track_vod_3d as vod_data

                labels1 = vod_data.load_labels(
                    frame_data_0.raw_tracking_labels, index + 1
                )
                labels2 = vod_data.load_labels(frame_data_1.raw_tracking_labels, index)

                transforms1 = FrameTransformMatrix(frame_data_0)
                transforms2 = FrameTransformMatrix(frame_data_1)

                lbl1 = labels1.data[index + 1]
                lbl2 = labels2.data[index]

                lbl1_mov = filter_moving_boxes_det(
                    frame_data_0.raw_detection_labels, lbl1
                )
                lbl2_mov = filter_moving_boxes_det(
                    frame_data_1.raw_detection_labels, lbl2
                )
            except:
                continue

            lbl1 = lbl1_mov
            lbl2 = lbl2_mov

            batch_size = pc1.size(0)
            num_examples += batch_size

        if args.model == "track4d_radar":
            (
                gt_mov_pts1,
                gt_cls1,
                gt_objs1,
                objs_idx1,
                objs_centre1,
                cls_obj_id1,
                boxes1,
                objs_combined1,
                objs_idx_combined1,
                objs_centre_combined1,
            ) = filter_object_points(args, lbl1, pc1, transforms1)
            (
                gt_mov_pts2,
                gt_cls2,
                gt_objs2,
                objs_idx2,
                objs_centre2,
                cls_obj_id2,
                boxes2,
                objs_combined2,
                objs_idx_combined2,
                objs_centre_combined2,
            ) = filter_object_points(args, lbl2, pc2, transforms2)

            gt_flow = get_gt_flow_new(
                objs_centre1,
                objs_centre2,
                gt_cls1,
                cls_obj_id1,
                cls_obj_id2,
                pc1,
                pc1_compensated,
                boxes1,
                boxes2,
            )
            gt_objs1 = objs_combined1
            objs_idx1 = objs_idx_combined1
            objs_centre1 = objs_centre_combined1
            (
                h,
                pc1_wrap,
                cls,
                aff_list,
                aff_mat,
                assig,
                confs,
                objects,
                timeout_obj_curr,
                objects_curr,
            ) = net(pc1, pc2, ft1, ft2, h, objects_prev)

            cls_mask = torch.where(cls.squeeze(0).squeeze(0) > 0.50, 1, 0)

            mappings_curr, mappings_inv = map_gt_objects(
                objs_centre1, gt_objs1, objects
            )

            seg_met_f = eval_motion_seg(cls_mask.float(), gt_cls1.float())
            for key in seg_met_f.keys():
                if key not in seg_met:
                    seg_met[key] = 0
                seg_met[key] += seg_met_f[key]

            flow_met_f = eval_scene_flow(pc1, pc1_wrap, gt_flow, cls)
            for key in flow_met_f.keys():
                if key not in flow_met:
                    flow_met[key] = 0
                flow_met[key] += flow_met_f[key]

            pretrain = True if ep_num < args.pretrain_epochs else False
            # pretrain = True if ep_num // 2 == 0 else False
            # pretrain = True if ep_num // 1 == 0 else False

            loss, items = track_4d_loss(
                objects_prev,
                objects,
                mappings_prev,
                mappings_curr,
                mappings_inv,
                lbl1,
                lbl2,
                pc1,
                pc2,
                pc1_wrap,
                cls,
                gt_flow,
                aff_list,
                gt_mov_pts1,
                gt_cls1,
                gt_objs1,
                objs_idx1,
                objs_centre1,
                pretrain=pretrain,
            )
            pbar.set_postfix({key: value.item() for key, value in items.items()})
            objects_prev = dict()
            for key, obj in objects.items():
                objects_prev[key] = obj.clone().detach()
            mappings_prev = mappings_curr
            if h is None:
                h = h.detach()

            if mode == "eval":
                Path(folder_results + seq[0]).mkdir(parents=True, exist_ok=True)

                file = open(
                    os.path.join(folder_results, seq[0], f"{index:05}.txt"), "w+"
                )
                idx = -1
                for obj_id, obj in objects.items():
                    idx += 1
                    result_str = "NA"
                    result_str += " 1"
                    result_str += " -1"
                    result_str += " -1"
                    result_str += " " + str(float(confs[idx]))
                    result_str += " " + str(obj_id)
                    for i in range(obj.size(2)):
                        result_str += " " + str(float(obj[0, 3, i]))
                        result_str += " " + str(float(obj[0, 4, i]))
                        result_str += " " + str(float(obj[0, 5, i]))
                    result_str += "\n"
                    file.writelines(result_str)

                pc1 = pc1.cpu()
                pc2 = pc2.cpu()
                ft1 = ft1.cpu()
                ft2 = ft2.cpu()
                pc1_compensated = pc1_compensated.cpu()

                mov_mask = (cls > 0.5).squeeze(0).cpu()
                pc1_mov = pc1[:, :, mov_mask]

                bbxs1 = []
                cors1 = []
                for idx, obj in lbl1.items():
                    bbox = get_bbx_param(
                        [obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.ry],
                        transforms1,
                        "radar",
                    )
                    bbxs1.append(bbox)
                    cor = np.asarray(bbox.get_box_points())
                    cors1.append(cor)
                bbxs2 = []
                cors2 = []
                cor_lbl2 = []
                for idx, obj in lbl1_mov.items():
                    bbox = get_bbx_param(
                        [obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.ry],
                        transforms1,
                        "radar",
                    )
                    bbxs2.append(bbox)
                    cor = np.asarray(bbox.get_box_points())
                    cors2.append(cor)
                    cor_lbl2.append(obj)
                # TODO: NT: consider saving cor2 in a binary log

                save_graph(cors2, folder_results_vis, index, objects, pc1, pc1_mov)

                """If display_images is True, display the point cloud and image frame side by side."""
                if display_images:
                    display_point_with_image_frame(
                        folder_results_vis, index, frame_data_0
                    )

                result = collect_data(cors2, objects, pc1, pc1_mov)
                results[index] = result

        if mode == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        batch_size = 1
        total_loss += loss.item() * batch_size

        for l in loss_items:
            loss_items[l].append(items[l].detach().cpu().numpy())

    # Save the results to msgpack file
    m.patch()
    print(len(results))  # TODO: NT: check if results is empty
    with open(
        f"./artifacts/{timestamp_folder}/results/msgpack.msgpack", "wb"
    ) as outfile:
        outfile.write(msgpack.packb(results))
    results = {}
    cv2.destroyAllWindows()

    return num_examples, total_loss, loss_items, trk_met, seg_met, flow_met


import os
from pathlib import Path
import matplotlib.pyplot as plt


def save_graph(cors2, folder_results_vis, index, objects, pc1, pc1_mov):
    fig = plt.figure()
    ax = fig.add_subplot()
    # Swapping the axes for the scatter plots
    ax.scatter(pc1[:, 1], pc1[:, 0], s=5, c="grey", marker=".", edgecolors="none")
    ax.scatter(
        pc1_mov[:, 1],
        pc1_mov[:, 0],
        s=5,
        c="black",
        marker=".",
        edgecolors="none",
    )
    cmap = get_cmap(len(objects))
    for i in range(len(objects)):
        key = list(objects.keys())[i]
        # Swapping the axes for object scatter plots
        ax.scatter(
            objects[key].detach().cpu().numpy()[:, 1 + 3],
            objects[key].detach().cpu().numpy()[:, 0 + 3],
            s=5,
            color=cmap(i),
            marker=".",
            edgecolors="none",
        )
        # Calculating the new centre after swapping axes
        centre = obj_centre(objects[key][:, :3, :])[0]
        ax.text(centre[1], centre[0], str(key), alpha=0.7, size=8)

    for cor in cors2:
        cor = cor[[True, True, True, False, False, False, False, True], :]
        # Swapping the axes for line plots
        ax.plot(
            [cor[0, 1], cor[1, 1]],
            [cor[0, 0], cor[1, 0]],
            "bo-",
            linewidth=0.3,
            markersize=0,
        )
        ax.plot(
            [cor[0, 1], cor[2, 1]],
            [cor[0, 0], cor[2, 0]],
            "bo-",
            linewidth=0.3,
            markersize=0,
        )
        ax.plot(
            [cor[1, 1], cor[3, 1]],
            [cor[1, 0], cor[3, 0]],
            "bo-",
            linewidth=0.3,
            markersize=0,
        )
        ax.plot(
            [cor[2, 1], cor[3, 1]],
            [cor[2, 0], cor[3, 0]],
            "bo-",
            linewidth=0.3,
            markersize=0,
        )

    # Updating the axis limits after swapping
    plt.xlim([30, -30])
    plt.ylim([-10, 50])
    plt.show()
    plt.axis("off")
    Path(folder_results_vis).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(folder_results_vis, f"seq{index}.png"), dpi=200)
    plt.close()


def get_cmap(n, name="hsv"):
    """Return a colormap with n different colors."""
    return plt.cm.get_cmap(name, n)


def obj_centre(obj):
    """Dummy function to calculate the center of the object."""
    return obj.mean(dim=1)


def display_point_with_image_frame(folder_results_vis, index, frame_data):

    cv_image = frame_data.get_image()

    # Load the point cloud image file
    pc_image_path = os.path.join(folder_results_vis, f"seq{index}.png")
    pc_image = Image.open(pc_image_path)  # Open the image file
    pc_image_array = np.array(pc_image)  # Convert the image to an array

    # Flip the point cloud image left to right
    # pc_image_array = np.fliplr(pc_image_array)

    # Combine and display images
    combined_image = combine_images(pc_image_array, cv_image)
    cv2.imshow("Point Cloud and Image", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # Wait for 1 ms to allow GUI to refresh and handle window events

    if cv2.getWindowProperty("Point Cloud and Image", cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()  # Close the window if it has been closed by the user


"""
This function combines two images side by side with a border in between.
The images are resized to match in height and adjusted in width to not exceed a maximum width.
The images are expected to be in the form of numpy arrays with shape (height, width, channels).
The border color, border width, and maximum width can be customized.

Args:
    img1 (np.ndarray): First image as a numpy array.
    img2 (np.ndarray): Second image as a numpy array.
    border_color (tuple): RGB color of the border. Default is white.
    border_width (int): Width of the border in pixels. Default is 10.
    max_width (int): Maximum width of the combined image. Default is 1920.
    
Returns:
    np.ndarray: Combined image as a numpy array.
"""


def combine_images(
    img1, img2, border_color=(255, 255, 255), border_width=10, max_width=1920
):
    # Check if images have 4 channels and convert to 3 if necessary
    if img1.shape[2] == 4:
        img1 = img1[:, :, :3]
    if img2.shape[2] == 4:
        img2 = img2[:, :, :3]

    # Resize images to match in height and adjust width to not exceed max_width
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate scale to fit both images and border within max_width
    scale_factor = min((max_width - border_width) / (w1 + w2), 1)

    # Resize both images with the same scale factor to maintain aspect ratio
    img1 = cv2.resize(img1, (int(w1 * scale_factor), int(h1 * scale_factor)))
    img2 = cv2.resize(img2, (int(w2 * scale_factor), int(h2 * scale_factor)))

    # Adjust height of the images to match by adding a black border to the smaller image
    if img1.shape[0] > img2.shape[0]:
        delta_h = img1.shape[0] - img2.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        img2 = cv2.copyMakeBorder(
            img2, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    elif img2.shape[0] > img1.shape[0]:
        delta_h = img2.shape[0] - img1.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        img1 = cv2.copyMakeBorder(
            img1, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # Create a vertical border with 3 channels
    border = np.full((img1.shape[0], border_width, 3), border_color, dtype=np.uint8)

    # Concatenate images with a border in between
    combined_image = np.hstack((img1, border, img2))
    return combined_image


def collect_data(cors2, objects, pc1, pc1_mov):
    # Ensure pc1 and pc1_mov are converted to numpy arrays at the beginning
    data = {
        "pc1": pc1.detach().cpu().numpy(),  # Detach and convert pc1 to numpy
        "pc1_mov": pc1_mov.detach()
        .cpu()
        .numpy(),  # Detach and convert pc1_mov to numpy
        "objects": {},
        "cors2": [],
    }

    # Collect object data
    for key, obj in objects.items():
        obj_data = obj.detach().cpu().numpy()  # Convert object tensor to numpy array
        center = (
            obj_centre(obj[:, :3, :])[0].cpu().numpy()
        )  # Ensure center is also a numpy array
        data["objects"][key] = {
            "points": obj_data[:, 3:6],  # Assuming the relevant columns are 3, 4, 5
            "center": center,
        }

    # Collect cors2 data
    # Ensure cors2 items are numpy arrays or compatible formats
    for cor in cors2:
        cor_filtered = cor[[True, True, True, False, False, False, False, True], :]
        data["cors2"].append(
            cor_filtered
        )  # Assume cor is already a numpy array or does not require conversion

    return data


def plot_loss_epoch(train_items_iter, args, epoch):
    plt.clf()
    plt.plot(np.array(train_items_iter["Loss"]).T, "b")
    plt.plot(np.array(train_items_iter["SceneFlowLoss"]).T, "r")
    plt.plot(np.array(train_items_iter["SegLoss"]).T, "g")
    plt.legend(["Total", "SceneFlowLoss", "SegLoss"], loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("checkpoints/%s/loss_train_%s.png" % (args.exp_name, epoch), dpi=500)


def get_carterian_res(pc, sensor):
    # measure resolution for r/theta/phi
    if sensor == "radar":  # LRR30
        r_res = 0.2  # m
        theta_res = 1 * np.pi / 180  # radian
        phi_res = 1.6 * np.pi / 180  # radian

    if sensor == "lidar":  # HDL-64E
        r_res = 0.04  # m
        theta_res = 0.4 * np.pi / 180  # radian
        phi_res = 0.08 * np.pi / 180  # radian

    res = np.array([r_res, theta_res, phi_res])
    # x y z
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # from xyz to r/theta/phi (range/elevation/azimuth)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(z / r)
    phi = np.arctan2(y, x)

    # compute xyz's gradient about r/theta/phi
    grad_x = np.stack(
        (
            np.cos(phi) * np.cos(theta),
            -r * np.sin(theta) * np.cos(phi),
            -r * np.cos(theta) * np.sin(phi),
        ),
        axis=2,
    )
    grad_y = np.stack(
        (
            np.sin(phi) * np.cos(theta),
            -r * np.sin(phi) * np.sin(theta),
            r * np.cos(theta) * np.cos(phi),
        ),
        axis=2,
    )
    grad_z = np.stack(
        (np.sin(theta), r * np.cos(theta), np.zeros((np.size(x, 0), np.size(x, 1)))),
        axis=2,
    )

    # measure resolution for xyz (different positions have different resolution)
    x_res = np.sum(abs(grad_x) * res, axis=2)
    y_res = np.sum(abs(grad_y) * res, axis=2)
    z_res = np.sum(abs(grad_z) * res, axis=2)

    xyz_res = np.stack((x_res, y_res, z_res), axis=2)

    return xyz_res


def vis_temp(
    bbb, gt_mov_pts, i, pc1, pc1_wrap, pc2, thres_cls, objects_curr, objects, args
):
    ############################################
    # for debug use
    if gt_mov_pts and i >= 0:
        mov_pts = pc1.permute(0, 2, 1)[:, :, thres_cls]
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(
            pc2.detach().cpu().numpy()[0, :, 0],
            pc2.detach().cpu().numpy()[0, :, 2],
            s=1,
            c="green",
            marker=".",
            edgecolors="none",
            label="pc2",
        )
        ax.scatter(
            pc1.detach().cpu().numpy()[0, :, 0],
            pc1.detach().cpu().numpy()[0, :, 2],
            s=1,
            c="blue",
            marker=".",
            edgecolors="none",
            label="pc1",
        )
        for key, obj in objects.items():
            ax.scatter(
                obj[0, 0, :].detach().cpu().numpy(),
                obj[0, 2, :].detach().cpu().numpy(),
                marker=".",
                s=1,
                c="red",
                edgecolors="none",
                label="pred_object_pt" + str(key),
            )
            centre = obj_centre(obj[:, :3, :])[0]
            ax.text(centre[0], centre[2], str(key), alpha=0.7, size=8)

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.legend(loc=1, prop={"size": 4}, markerscale=5)

        plt.xlim([-20, 20])
        plt.ylim([0, 35])

        plt.show()
        plt.savefig("checkpoints/%s/models/" % args.exp_name + str(i) + ".png", dpi=500)
    ##########################################################################


def eval_scene_flow(pc, pred, labels, mask):
    mask = mask.squeeze(0)
    pc = pc.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    mask = mask.detach().cpu().numpy()
    error = np.sqrt(np.sum((pred - labels) ** 2, 1) + 1e-20)
    epe = np.mean(error)
    gtflow_len = np.sqrt(np.sum(labels * labels, 1) + 1e-20)

    # obtain x y z measure resolution for each point (radar lidar)
    xyz_res_r = get_carterian_res(pc, "radar")
    res_r = np.sqrt(np.sum(xyz_res_r, 2) + 1e-20)
    xyz_res_l = get_carterian_res(pc, "lidar")
    res_l = np.sqrt(np.sum(xyz_res_l, 2) + 1e-20)

    # calcualte Resolution-Normalized Error
    rn_error = error / (res_r / res_l)
    rne = np.mean(rn_error)
    mov_rne = np.sum(rn_error[:, mask == 0]) / (np.sum(mask == 0) + 1e-6)
    stat_rne = np.mean(rn_error[:, mask == 1])
    avg_rne = (mov_rne + stat_rne) / 2

    # calculate Strict/Relaxed Accuracy Score
    sas = np.sum(np.logical_or((rn_error <= 0.10), (rn_error / gtflow_len <= 0.10))) / (
        np.size(pred, 0) * np.size(pred, 2)
    )
    ras = np.sum(np.logical_or((rn_error <= 0.20), (rn_error / gtflow_len <= 0.20))) / (
        np.size(pred, 0) * np.size(pred, 2)
    )

    sf_metric = {
        "rne": rne,
        "50-50 rne": avg_rne,
        "mov_rne": mov_rne,
        "stat_rne": stat_rne,
        "sas": sas,
        "ras": ras,
        "epe": epe,
    }

    return sf_metric


def eval_motion_seg(pre, gt):
    pre = pre.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    tp = np.logical_and((pre == 1), (gt == 1)).sum() + 1e-20
    tn = np.logical_and((pre == 0), (gt == 0)).sum() + 1e-20
    fp = np.logical_and((pre == 1), (gt == 0)).sum() + 1e-20
    fn = np.logical_and((pre == 0), (gt == 1)).sum() + 1e-20
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    miou = 0.5 * (tp / (tp + fp + fn + 1e-4) + tn / (tn + fp + fn + 1e-4))
    seg_metric = {"acc": acc, "miou": miou, "sen": sen}

    return seg_metric


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
