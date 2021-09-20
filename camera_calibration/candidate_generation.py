import os
import json
import numpy as np
from cv2 import cv2
from itertools import combinations
from camera_calibration.util_files.io import load_image, load_image_points, load_object_points


class CandidateGeneration(object):
    def __init__(self, config):
        self.camera_code = config["camera_code"]
        self.image_height = config["image_height"]
        self.image_width = config["image_width"]
        self.image_center_x = self.image_width // 2
        self.image_center_y = self.image_height // 2
        self.focal_length = config["focal_length"]
        self.loss_alpha = config["alpha"]
        self.loss_t = config["temperature"]
        self.budget_1 = config["budget_1"]
        self.budget_2 = config["budget_2"]
        self.use_ransac = config["use_ransac"]

        self.object_points_path = config["object_points_path"]
        self.image_points_dir = config["image_points_dir"]
        self.images_dir = config["images_dir"]
        self.save_dir = config["save_dir"]
        # self.data_rootpath = config.data_rootpath
        # self.object_points_path = config.object_points_path
        # self.image_points_index_path = config.image_points_index_path
        # self.file_prefix_template = config.file_prefix_template
        # self.montage_height = int(config.montage_height)
        # self.montage_width = int(config.montage_width)

        # self.save_rootpath = config.save_rootpath
        # self.param_name_template = config.param_name_template
        # self.montage_name_template = config.montage_name_template
        self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix[0, 0] = self.focal_length
        self.intrinsic_matrix[1, 1] = self.focal_length
        self.intrinsic_matrix[0, 2] = self.image_center_x
        self.intrinsic_matrix[1, 2] = self.image_center_y
        self.discoeffs = np.zeros(5)

        # self.max_label_nums = int(config.max_label_nums)
        # self.object_points = None
        # self.image_points_index = None

    def prepare_keypoint_pairs(self, object_points, image_points_dict):
        selected_object_points = []
        selected_image_points = []
        for key, value in image_points_dict.items():
            selected_object_points.append(object_points[key - 1])
            selected_image_points.append(value)
        return np.array(selected_object_points), np.array(selected_image_points)

    def get_projection_error(self, object_points, image_points, rvec, tvec):
        """
        :param object_points: 3D key points in the real world.
        :param image_points: 2D key points in images.
        :param rvec: rotation vector.
        :param tvec: translation vector.
        :return: projection error.
        """
        projected_points = cv2.projectPoints(object_points, rvec, tvec,
                                              self.intrinsic_matrix, self.discoeffs)
        loss = np.sqrt(np.sum(np.square(projected_points[0][:, 0, :] - image_points)) / np.shape(image_points)[0])
        return loss

    def get_pairwise_combination_iter(self, objects_points, image_points):
        comb = combinations(range(len(objects_points)), 4)
        for each in comb:
            yield np.r_[[objects_points[i] for i in each]], np.r_[[image_points[i] for i in each]]

    def calibration(self, object_points, image_points):
        if self.use_ransac:
            retval, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                              self.intrinsic_matrix, self.discoeffs,
                                              flags=cv2.SOLVEPNP_EPNP)
            loss = self.get_projection_error(object_points, image_points, rvec, tvec)

        else:
            loss = 1e5
            rvec = None
            tvec = None
            for each in self.get_pairwise_combination_iter(object_points, image_points):
                retval, temp_rvec, temp_tvec = cv2.solvePnP(each[0], each[1],
                                                            self.intrinsic_matrix, self.discoeffs,
                                                            flags=cv2.SOLVEPNP_EPNP)
                temp_loss = self.get_projection_error(object_points, image_points, temp_rvec, temp_tvec)
                if temp_loss < loss:
                    rvec = temp_rvec
                    tvec = temp_tvec
                    loss = temp_loss

        return rvec, tvec, loss

    def get_back_projected_point(self, image_point, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        u = (image_point[0] - self.image_width // 2) / self.focal_length
        v = (image_point[1] - self.image_height // 2) / self.focal_length
        coeffcient_x_1 = rmat[0, 0] - u * rmat[2, 0]
        coeffcient_x_2 = rmat[1, 0] - v * rmat[2, 0]
        coeffcient_y_1 = rmat[0, 1] - u * rmat[2, 1]
        coeffcient_y_2 = rmat[1, 1] - v * rmat[2, 1]

        coeffcient_c_1 = 0 - (tvec[0, 0] - u * tvec[2, 0])
        coeffcient_c_2 = 0 - (tvec[1, 0] - v * tvec[2, 0])
        A = np.array([[coeffcient_x_1, coeffcient_y_1], [coeffcient_x_2, coeffcient_y_2]])
        b = np.array([[coeffcient_c_1], [coeffcient_c_2]])
        retval = np.linalg.solve(A, b)
        return retval[0, 0], retval[1, 0]

    def generate_projection_points(self, projection_points, image, key="None"):
        for i in range(len(projection_points)):
            cv2.circle(image, tuple(projection_points[i][0]), 1, (0, 0, 255), -1)
        cv2.rectangle(image,
                      (0, int(0.9 * self.image_height)),
                      (self.image_width, self.image_height),
                      (50, 50, 50), -1)
        cv2.putText(image, key, (0, int(0.95 * self.image_height)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def visualization(self, rvec, tvec, vehicle_index, model_name):
        corner_1 = (0.25 * self.image_width, 0.25 * self.image_height)
        corner_2 = (0.25 * self.image_width, 0.75 * self.image_height)
        corner_3 = (0.75 * self.image_width, 0.25 * self.image_height)
        corner_4 = (0.75 * self.image_width, 0.75 * self.image_height)
        x1, y1 = self.get_back_projected_point(corner_1, rvec, tvec)
        x2, y2 = self.get_back_projected_point(corner_2, rvec, tvec)
        x3, y3 = self.get_back_projected_point(corner_3, rvec, tvec)
        x4, y4 = self.get_back_projected_point(corner_4, rvec, tvec)
        x = int(max(abs(x1), abs(x2), abs(x3), abs(x4)))
        y = int(max(abs(y1), abs(y2), abs(y3), abs(y4)))
        index = np.reshape(np.indices((x, y)).transpose(1, 2, 0), [-1, 2])
        index = np.c_[(index - np.array([x // 2, y // 2])), np.zeros([index.shape[0], 1])]
        projection_points = cv2.projectPoints(index, rvec, tvec, self.intrinsic_matrix, self.discoeffs)
        return self.generate_projection_points(projection_points[0].astype(np.int32),
                                               load_image(os.path.join(self.images_dir, sorted(os.listdir(self.images_dir))[vehicle_index])),
                                               "%s_%02d" % (model_name, vehicle_index))

    def create_montage(self, images, savepath, width=5, height=7, interval=10):
        interval_height_canvas = np.zeros([self.image_height, interval], dtype=np.uint8)
        interval_height_canvas = cv2.cvtColor(interval_height_canvas, cv2.COLOR_GRAY2BGR)
        width_interval_canvas = np.zeros([interval, self.image_width + interval], dtype=np.uint8)
        width_interval_canvas = cv2.cvtColor(width_interval_canvas, cv2.COLOR_GRAY2BGR)
        rol = []

        for i in range(height):
            col = []
            for j in range(width):
                temp = cv2.hconcat([images[i * width + j], interval_height_canvas])
                temp = cv2.vconcat([temp, width_interval_canvas])
                col.append(temp)
            col = cv2.hconcat(col)
            rol.append(col)
        rol = cv2.vconcat(rol)
        cv2.imwrite(savepath, rol[:-interval, :-interval, :])

    def run(self):
        print("Begin Calibration by EPNP.")
        object_points_dict = load_object_points(self.object_points_path)
        image_points_dict = load_image_points(self.image_points_dir)
        vehicle_num = len(image_points_dict)
        model_num = len(object_points_dict)

        images = []
        res = {}
        for i, each in enumerate(image_points_dict):
            temp_res = {}
            for key, value in object_points_dict.items():
                object_points, image_points = self.prepare_keypoint_pairs(value, each)
                rvec, tvec, loss = self.calibration(object_points, image_points)
                images.append(self.visualization(rvec, tvec, i, key))
                temp_res[key] = {}
                temp_res[key]["rvec"] = rvec.tolist()
                temp_res[key]["tvec"] = tvec.tolist()
                temp_res[key]["f"] = self.focal_length
                temp_res[key]["loss"] = loss
            res[i] = temp_res

        self.create_montage(images, os.path.join(self.save_dir, "cg_montage.jpg"), model_num, vehicle_num)
        with open(os.path.join(self.save_dir, "cg_results.json"), "w") as f:
            json.dump(res, f)



if __name__ == '__main__':
    config = {}
    config["camera_code"] = "K109F"
    config["image_height"] = 240
    config["image_width"] = 320
    config["focal_length"] = 350
    config["alpha"] = 1
    config["temperature"] = 0
    config["budget_1"] = 2000
    config["budget_2"] = 4000
    config["use_ransac"] = False
    config["object_points_path"] = "../examples/camera_calibration/K109F/objects/vehicle_3dkeypoints_head.json"
    config["image_points_dir"] = "../examples/camera_calibration/K109F/labels"
    config["images_dir"] = "../examples/camera_calibration/K109F/images"
    config["save_dir"] = "../examples/camera_calibration/K109F/outputs"
    cg = CandidateGeneration(config)
    cg.run()