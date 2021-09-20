import os
import numpy as np
import nevergrad as ng
from cv2 import cv2
from concurrent import futures
from itertools import combinations
from camera_calibration.util_files.config import Config
from camera_calibration.util_files.process import normalize, softmax
from camera_calibration.util_files.data_structure import TreeNode
from camera_calibration.util_files.io import load_image_points, load_object_points
import argparse
import time


class CameraCalibrator_CMA(object):
    def __init__(self, config):
        self.camera_code = config.camera_code
        self.data_rootpath = config.data_rootpath
        self.object_points_path = config.object_points_path
        self.image_points_index_path = config.image_points_index_path
        self.file_prefix_template = config.file_prefix_template
        self.montage_height = int(config.montage_height)
        self.montage_width = int(config.montage_width)
        self.image_height = int(config.default_image_height)
        self.image_width = int(config.default_image_width)
        self.focal_length = config.default_focal_length

        self.save_rootpath = config.save_rootpath
        self.finetune_template = config.finetune_template
        self.param_name_template = config.param_name_template_CMA
        self.montage_name_template = config.montage_name_template_CMA
        self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix[0, 0] = config.default_focal_length
        self.intrinsic_matrix[1, 1] = config.default_focal_length
        self.intrinsic_matrix[0, 2] = config.default_image_center_x
        self.intrinsic_matrix[1, 2] = config.default_image_center_y
        self.alpha = config.default_CMA_loss_alpha
        self.beta = config.default_CMA_loss_beta
        self.budget = config.default_CMA_budget
        self.softmax_temperature = config.default_CMA_softmax_temperature
        self.discoeffs = config.default_discoeffs
        self.max_label_nums = int(config.max_label_nums)

        self.method = "CMA"
        self.pre_method = "EPNP"
        self.pre_method_suffix = None
        self.object_points = None
        self.image_points_index = None
        self.namelist = None
        self.cartypes = None
        self.all_leaf_nodes = []

        self.fix_index = -1
        self.holdout_index = -1

    def init_tree_leafnode(self):
        leaf_nodes = []
        if self.pre_method_suffix is None:
            pre_path = os.path.join(self.save_rootpath, self.camera_code, self.pre_method)
        else:
            pre_path = os.path.join(self.save_rootpath, self.camera_code,
                                    self.pre_method, self.pre_method_suffix)
        for i_name, each in enumerate(self.namelist):
            pre_data_path = os.path.join(self.data_rootpath, self.camera_code)
            if self.holdout_index != -1 and i_name >= self.holdout_index:
                pre_param_path = os.path.join(pre_path,
                                              self.param_name_template % (
                                              self.camera_code, i_name + 1, self.pre_method))
            else:
                pre_param_path = os.path.join(pre_path,
                                              self.param_name_template % (
                                                  self.camera_code, i_name, self.pre_method))
            temp_image_points, labels = load_image_points(os.path.join(pre_data_path, each + ".json"))
            temp_image_params = np.load(pre_param_path)
            temp_object_points, missing_label = self.select_match_points(labels)
            leaf_node = TreeNode()
            leaf_node.set_name(each)
            leaf_node.set_value({"rvec": temp_image_params["best_rvec"],
                                 "tvec": temp_image_params["best_tvec"],
                                 "f": temp_image_params["best_f"],
                                 "object_points": temp_object_points,
                                 "image_points": temp_image_points,
                                 "missing_label": missing_label,
                                 "cartype": temp_image_params["cartype"]})
            leaf_node.set_parent(self.root_node)
            leaf_node.set_children(None)
            leaf_nodes.append(leaf_node)
        return leaf_nodes

    def init_tree_rootnode(self):
        print("The name list in current directory is:\n", "\n".join(self.namelist))
        root_node_children = self.init_tree_leafnode()
        self.root_node.set_children(root_node_children)
        self.root_node.set_value({"fl": self.focal_length, "rvec": None})
        self.root_node.set_name("rotation_matrix_and_focal_length")

    def select_match_points(self, labels):
        temp_object_points = {}
        if len(labels) == self.max_label_nums:
            return self.object_points, -1

        if len(labels) <= 4:
            raise IndexError("labels are not sufficient.")

        missing_label = None
        for each in range(1, self.max_label_nums + 1):
            if each not in labels:
                missing_label = each
                break
        for key, value in self.object_points.items():
            temp_value = np.array(value)
            temp_value = np.delete(temp_value, missing_label - 1, 0)
            temp_object_points[key] = temp_value

        return temp_object_points, missing_label

    def get_world_reprojection(self, image_point, rvec, tvec, f, Z):
        rmat = cv2.Rodrigues(rvec)[0]
        u = (image_point[0] - self.image_width // 2) / f
        v = (image_point[1] - self.image_height // 2) / f

        coeffcient_x_1 = rmat[0, 0] - u * rmat[2, 0]
        coeffcient_x_2 = rmat[1, 0] - v * rmat[2, 0]
        coeffcient_y_1 = rmat[0, 1] - u * rmat[2, 1]
        coeffcient_y_2 = rmat[1, 1] - v * rmat[2, 1]

        coeffcient_c_1 = 0 - ((tvec[0] - u * tvec[2]) + (rmat[0, 2] - u * rmat[2, 2]) * Z)
        coeffcient_c_2 = 0 - ((tvec[1] - v * tvec[2]) + (rmat[1, 2] - v * rmat[2, 2]) * Z)
        A = np.array([[coeffcient_x_1, coeffcient_y_1], [coeffcient_x_2, coeffcient_y_2]])
        b = np.array([[coeffcient_c_1], [coeffcient_c_2]])
        retval = np.linalg.solve(A, b)
        return retval[0, 0], retval[1, 0]

    def cal_3D_3D_loss(self, rvec, tvec, f, debug=False):
        losses = []
        reproj_centriods = []
        fix_object_points = self.root_node.get_children()[self.fix_index].get_value()["object_points"][
            self.cartypes[self.fix_index]]
        for i, leaf_node in enumerate(self.root_node.get_children()):
            X_Y_hats = []
            image_points = leaf_node.get_value()["image_points"]
            object_points = leaf_node.get_value()["object_points"][self.cartypes[i]]
            Zs = object_points[:, 2]
            for image_point, Z in zip(image_points, Zs):
                X_hat, Y_hat = self.get_world_reprojection(image_point, rvec, tvec, f, Z)
                X_Y_hats.append([X_hat, Y_hat])
            reproj_object_points = np.c_[np.array(X_Y_hats), Zs]
            losses.append(self.cal_internal_loss(reproj_object_points, object_points))
            reproj_centriods.append(np.mean(reproj_object_points, axis=0))
        fix_centriod = np.reshape(np.mean(fix_object_points, axis=0), [1, 3])

        weights = np.linalg.norm(fix_centriod - reproj_centriods, axis=1)
        weights = softmax(normalize(weights), self.softmax_temperature)
        losses = np.array(losses)

        if debug:
            print()
            print("Debugging..., fix index is %d and loss is %s" % (self.fix_index,
                                                                    " ".join(["%.2f" % each for each in weights])))
            return np.sum(losses[:, 0] * weights), np.mean(losses[:, 1]), np.mean(losses[:, 2])
        else:
            return np.sum(losses[:, 0] * weights)

    def cal_evaluated_3D_loss(self, rvec, tvec, f, cartype):
        losses = []
        reproj_centriods = []
        fix_object_points = self.root_node.get_children()[self.fix_index].get_value()["object_points"][cartype]
        for i, leaf_node in enumerate(self.root_node.get_children()):
            X_Y_hats = []
            image_points = leaf_node.get_value()["image_points"]
            object_points = leaf_node.get_value()["object_points"][cartype]
            Zs = object_points[:, 2]
            for image_point, Z in zip(image_points, Zs):
                X_hat, Y_hat = self.get_world_reprojection(image_point, rvec, tvec, f, Z)
                X_Y_hats.append([X_hat, Y_hat])
            reproj_object_points = np.c_[np.array(X_Y_hats), Zs]
            losses.append(self.cal_internal_loss(reproj_object_points, object_points))
            reproj_centriods.append(np.mean(reproj_object_points, axis=0))
        fix_centriod = np.reshape(np.mean(fix_object_points, axis=0), [1, 3])

        weights = np.linalg.norm(fix_centriod - reproj_centriods, axis=1)
        weights = softmax(normalize(weights), self.softmax_temperature)
        losses = np.array(losses)

        return np.sum(losses[:, 0] * weights), np.mean(losses[:, 1]), np.mean(losses[:, 2])

    def cal_internal_loss(self, reproj_object_points, object_points):
        pairs = sorted(list(combinations(list(range(1, len(object_points) + 1)), 2)))
        line_loss = 0
        angle_loss = 0
        total_loss = 0
        for each in pairs:
            pre = each[0] - 1
            pro = each[1] - 1
            reproj_object_vector = reproj_object_points[pre] - reproj_object_points[pro]
            object_vector = object_points[pre] - object_points[pro]
            length_origin = np.linalg.norm(object_vector)
            length_reproj = np.linalg.norm(reproj_object_vector)
            cos_theta = np.dot(object_vector, reproj_object_vector.T) / (
                    np.linalg.norm(object_vector) * np.linalg.norm(reproj_object_vector))
            temp_line_loss = np.square(length_origin - length_reproj)
            temp_angle_loss = np.sqrt(1 - min(np.square(cos_theta), 1))
            line_loss += temp_line_loss
            angle_loss += temp_angle_loss
            total_loss += temp_line_loss + self.alpha * angle_loss
        return total_loss / len(pairs), line_loss / len(pairs), angle_loss / len(pairs)

    def cal_2D_2D_loss(self, rvec, tvec, f, cartype=None):
        self.intrinsic_matrix[0, 0] = f
        self.intrinsic_matrix[1, 1] = f
        fix_leaf_node = self.root_node.get_children()[self.fix_index]
        if self.cartypes is None:
            object_points = fix_leaf_node.get_value()["object_points"][cartype]
        else:
            object_points = fix_leaf_node.get_value()["object_points"][self.cartypes[self.fix_index]]
        image_points = fix_leaf_node.get_value()["image_points"]
        projection_points = cv2.projectPoints(object_points, rvec, tvec, self.intrinsic_matrix, self.discoeffs)[0][:, 0,
                            :]
        return np.mean(np.linalg.norm(projection_points - image_points, axis=1))

    def prepare_for_CMA(self, cartype):
        fix_sub_node = self.root_node.get_children()[self.fix_index]
        self.fix_missing_label = self.root_node.get_children()[self.fix_index].get_value()["missing_label"]
        index = fix_sub_node.get_value()["cartype"].tolist().index(cartype)
        tvec = fix_sub_node.get_value()["tvec"][index]
        rvec = fix_sub_node.get_value()["rvec"][index]
        f = fix_sub_node.get_value()["f"][index]
        return rvec, tvec, f

    def params_loss_func(self, row, pitch, yaw, t1, t2, t3, f, debug=False):
        state = np.array([row, pitch, yaw, t1, t2, t3, f])
        return self.loss_func(state, debug)

    def params_2D_loss(self, row, pitch, yaw, t1, t2, t3, f, cartype):
        rvec = np.array([row, pitch, yaw])
        tvec = np.array([t1, t2, t3])
        return self.cal_2D_2D_loss(rvec, tvec, f, cartype)

    def params_3D_loss(self, row, pitch, yaw, t1, t2, t3, f, cartype=None):
        rvec = np.array([row, pitch, yaw])
        tvec = np.array([t1, t2, t3])
        # return self.cal_3D_3D_loss(rvec, tvec, f) + self.cal_2D_2D_loss(rvec, tvec, f, cartype)
        return self.cal_3D_3D_loss(rvec, tvec, f)
        # return self.cal_2D_2D_loss(rvec, tvec, f, cartype)

    def optimize_Nevergrad(self, name, rvec, tvec, f, loss_func, cartype=None):
        state = np.array(list(rvec) + list(tvec) + [f])
        roll = ng.p.Scalar(init=state[0],
                           lower=state[0] - 0.3 * np.abs(state[0]),
                           upper=state[0] + 0.3 * np.abs(state[0]))
        pitch = ng.p.Scalar(init=state[1],
                            lower=state[1] - 0.3 * np.abs(state[1]),
                            upper=state[1] + 0.3 * np.abs(state[1]))
        yaw = ng.p.Scalar(init=state[2],
                          lower=state[2] - 0.3 * np.abs(state[2]),
                          upper=state[2] + 0.3 * np.abs(state[2]))
        t1 = ng.p.Scalar(init=state[3],
                         lower=state[3] - 0.3 * np.abs(state[3]),
                         upper=state[3] + 0.3 * np.abs(state[3]))
        t2 = ng.p.Scalar(init=state[4],
                         lower=state[4] - 0.3 * np.abs(state[4]),
                         upper=state[4] + 0.3 * np.abs(state[4]))
        t3 = ng.p.Scalar(init=state[5],
                         lower=state[5] - 0.3 * np.abs(state[5]),
                         upper=state[5] + 0.3 * np.abs(state[5]))
        f = ng.p.Scalar(init=state[6],
                        lower=state[6] - 0.3 * np.abs(state[6]),
                        upper=state[6] + 0.3 * np.abs(state[6]))
        params = ng.p.Instrumentation(roll, pitch, yaw, t1, t2, t3, f, cartype)
        before_losses = loss_func(state[0], state[1], state[2],
                                  state[3], state[4], state[5],
                                  state[6], cartype)

        optimizer = ng.optimizers.registry[name](parametrization=params, budget=self.budget, num_workers=5)
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(loss_func, executor=executor, batch_mode=False)
        res = recommendation.args
        after_losses = loss_func(res[0], res[1], res[2],
                                 res[3], res[4], res[5],
                                 res[6], cartype)
        print("Before finetune is: ", before_losses)
        print("After finetune is: ", after_losses)
        best_rvec = np.array([res[0], res[1], res[2]])
        best_tvec = np.array([res[3], res[4], res[5]])
        best_f = res[6]
        return best_rvec, best_tvec, best_f, after_losses

    def initialize(self):
        print("Load the real coordinates from files")
        self.object_points = load_object_points(self.object_points_path)
        self.all_cartypes = list(self.object_points.keys())
        print("Initialize the tree.")
        self.root_node = TreeNode()
        self.init_tree_rootnode()
        print("Tree has been built.")
        print("Initialize other params.")

    def run_preliminary(self):
        self.namelist = [self.file_prefix_template % (i + 1) for i in range(int(self.montage_height))]
        self.initialize()
        total_rvecs = []
        total_tvecs = []
        total_fs = []
        total_errors = []
        total_axises = []
        total_cartype = []

        before_2d_losses = []
        after_2d_losses = []
        before_3d_losses = []
        after_3d_losses = []
        before_line_losses = []
        after_line_losses = []
        before_angle_losses = []
        after_angle_losses = []
        for i_name, each in enumerate(self.namelist):
            self.fix_index = i_name
            temp_rvecs = []
            temp_tvecs = []
            temp_fs = []
            temp_errors = []
            temp_cartype = []

            temp_before_2d_loss = []
            temp_after_2d_loss = []
            temp_before_3d_loss = []
            temp_after_3d_loss = []
            temp_before_line_loss = []
            temp_after_line_loss = []
            temp_before_angle_loss = []
            temp_after_angle_loss = []
            for j, cartype in enumerate(self.all_cartypes):
                print("Presenting the CMA algorithm. And Optimizing with {}".format(cartype))
                rvec, tvec, f = self.prepare_for_CMA(cartype)
                before_2d_loss = self.cal_2D_2D_loss(rvec, tvec, f, cartype)
                before_3d_loss, before_line_loss, before_angle_loss = self.cal_evaluated_3D_loss(rvec, tvec, f, cartype)
                rvec, tvec, f, errors = self.optimize_Nevergrad(self.method, rvec, tvec, f, self.params_2D_loss,
                                                                cartype)
                after_2d_loss = self.cal_2D_2D_loss(rvec, tvec, f, cartype)
                after_3d_loss, after_line_loss, after_angle_loss = self.cal_evaluated_3D_loss(rvec, tvec, f, cartype)
                temp_rvecs.append(rvec)
                temp_tvecs.append(tvec)
                temp_fs.append(f)
                temp_errors.append(errors)
                temp_cartype.append(cartype)

                temp_before_2d_loss.append(before_2d_loss)
                temp_before_3d_loss.append(before_3d_loss)
                temp_before_line_loss.append(before_line_loss)
                temp_before_angle_loss.append(before_angle_loss)
                temp_after_2d_loss.append(after_2d_loss)
                temp_after_3d_loss.append(after_3d_loss)
                temp_after_line_loss.append(after_line_loss)
                temp_after_angle_loss.append(after_angle_loss)

            min_index = np.argmin(np.array(temp_errors))
            total_rvecs.append(temp_rvecs[min_index])
            total_tvecs.append(temp_tvecs[min_index])
            total_fs.append(temp_fs[min_index])
            total_errors.append(temp_errors[min_index])
            total_cartype.append(temp_cartype[min_index])
            total_axises.append(temp_cartype[min_index] + "_index_" + str(min_index))

            before_2d_losses.append(temp_before_2d_loss)
            before_3d_losses.append(temp_before_3d_loss)
            before_line_losses.append(temp_before_line_loss)
            before_angle_losses.append(temp_before_angle_loss)
            after_2d_losses.append(temp_after_2d_loss)
            after_3d_losses.append(temp_after_3d_loss)
            after_line_losses.append(temp_after_line_loss)
            after_angle_losses.append(temp_after_angle_loss)

        self.cartypes = total_cartype
        print("the first part of CMA has finished.")
        np.savez(self.camera_code + "_warmup_loss.npz",
                 before_2d_losses=before_2d_losses, before_3d_losses=before_3d_losses,
                 before_line_losses=before_line_losses, before_angle_losses=before_angle_losses,
                 after_2d_losses=after_2d_losses, after_3d_losses=after_3d_losses,
                 after_line_losses=after_line_losses, after_angle_losses=after_angle_losses)

    def run_final(self):
        self.namelist = [self.file_prefix_template % (i + 1) for i in range(int(self.montage_height))]
        self.initialize()
        total_rvecs = []
        total_tvecs = []
        total_fs = []
        total_errors = []
        total_axises = []
        plot_2d_losses = []
        plot_3d_losses = []
        plot_line_losses = []
        plot_angle_losses = []
        plot_names = []

        self.budget *= 5
        for i_name, each in enumerate(self.namelist):
            self.fix_index = i_name
            rvec, tvec, f = self.prepare_for_CMA(self.cartypes[i_name])
            # rvec, tvec, f, errors = self.optimize_Nevergrad(self.method, rvec, tvec, f, self.params_3D_loss, self.cartypes[i_name])
            rvec, tvec, f, errors = self.optimize_Nevergrad(self.method, rvec, tvec, f, self.params_2D_loss,
                                                            self.cartypes[i_name])
            total_rvecs.append(rvec)
            total_tvecs.append(tvec)
            total_fs.append(f)
            total_errors.append(errors)
            total_axises.append(self.cartypes[i_name] + "_index_" + str(i_name))
            _2d_loss = self.cal_2D_2D_loss(rvec, tvec, f, self.cartypes[i_name])
            _3d_loss, line_loss, angle_loss = self.cal_evaluated_3D_loss(rvec, tvec, f, self.cartypes[i_name])
            plot_2d_losses.append(_2d_loss)
            plot_3d_losses.append(_3d_loss)
            plot_line_losses.append(line_loss)
            plot_angle_losses.append(angle_loss)
            plot_names.append(self.cartypes[i_name])

        self.save_final_results(total_rvecs, total_tvecs, total_fs, total_errors)
        print("the final part of CMA has finished.")
        np.savez(self.camera_code + "_final_loss.npz",
                 plot_2d_losses=plot_2d_losses, plot_3d_losses=plot_3d_losses,
                 plot_line_losses=plot_line_losses, plot_angle_losses=plot_angle_losses,
                 plot_names=plot_names)

    def save_final_results(self, rvecs, tvecs, fs, errors):
        sub_savepath = os.path.join(self.save_rootpath, self.camera_code)
        subsub_savepath = os.path.join(sub_savepath, self.method)
        subsubsub_savepath = os.path.join(subsub_savepath,
                                          self.finetune_template % (self.beta * self.alpha,
                                                                    self.beta,
                                                                    self.budget))
        if not os.path.exists(self.save_rootpath):
            os.mkdir(self.save_rootpath)
        if not os.path.exists(sub_savepath):
            os.mkdir(sub_savepath)
        if not os.path.exists(subsub_savepath):
            os.mkdir(subsub_savepath)
        if not os.path.exists(subsubsub_savepath):
            os.mkdir(subsubsub_savepath)
        savepath = os.path.join(subsubsub_savepath,
                                self.param_name_template % (self.camera_code, 0, self.method))

        print("Finish optimizing, saving result to {}".format(savepath))
        np.savez(savepath, best_rvec=np.array(rvecs), best_tvec=np.array(tvecs),
                 best_f=np.array(fs),
                 loss=np.array(errors), cartype=self.cartypes,
                 region=self.camera_code)

    def plot_match_point(self, projection_points, image, key="None"):
        for i in range(len(projection_points)):
            cv2.circle(image, tuple(projection_points[i][0]), 1, (0, 0, 255), -1)
        cv2.imwrite("calibration_result_%s.jpg" % key, image)

    def generate_projection_points(self, projection_points, image_points, image, key=None, alpha=0.75):
        for i in range(np.shape(image_points)[0]):
            cv2.circle(image, tuple(image_points[i]), 1, (255, 0, 0), -1)
        for i in range(len(projection_points)):
            cv2.circle(image, tuple(projection_points[i][0]), 1, (0, 0, 255), -1)
        cv2.rectangle(image, (0, 440), (720, 480), (50, 50, 50), -1)
        cv2.putText(image, key, (0, 470), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate a certain region.')
    parser.add_argument('-r', "--region", dest="region", type=str, default="K109F", help="Select a region")
    args = parser.parse_args()
    region = args.region
    time_start = time.time()
    results = []
    config = Config("util_files/camera_config.xml")
    for each in config.parse_xml_file():
        # if config.camera_code == region:
        if config.camera_code in ["Caltrans_I5"]:
            # if config.camera_code in ["K109F"]:
            print("Current camera code is: %s" % config.camera_code)
            calibrator = CameraCalibrator_CMA(config)
            calibrator.run_preliminary()
            calibrator.run_final()

    time_end = time.time()
    run_time = time_end - time_start
    print("Final time cost is: {} seconds".format(run_time))
    print("Final time cost is: {} minutes".format(run_time / 60))
    print("Final time cost is: {} hours".format(run_time / 3600))