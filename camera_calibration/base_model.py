import os
import numpy as np
from cv2 import cv2
from camera_calibration.util_files.io import load_image

class BaseModel(object):
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
        self.opt_name = config["opt_name"]

        self.object_points_path = config["object_points_path"]
        self.image_points_dir = config["image_points_dir"]
        self.images_dir = config["images_dir"]
        self.save_dir = config["save_dir"]

        self.verbose = config["verbose"]
        self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix[0, 0] = self.focal_length
        self.intrinsic_matrix[1, 1] = self.focal_length
        self.intrinsic_matrix[0, 2] = self.image_center_x
        self.intrinsic_matrix[1, 2] = self.image_center_y
        self.discoeffs = np.zeros(5)


    def prepare_keypoint_pairs(self, object_points, image_points_dict):
            selected_object_points = []
            selected_image_points = []
            for key, value in image_points_dict.items():
                selected_object_points.append(object_points[key - 1])
                selected_image_points.append(value)
            return np.array(selected_object_points), np.array(selected_image_points)

    def get_projection_error(self, object_points, image_points, rvec, tvec, f):
        """
        :param object_points: 3D key points in the real world.
        :param image_points: 2D key points in images.
        :param rvec: rotation vector.
        :param tvec: translation vector.
        :return: projection error.
        """
        self.intrinsic_matrix[0, 0] = f
        self.intrinsic_matrix[1, 1] = f
        projected_points = cv2.projectPoints(object_points, rvec, tvec, self.intrinsic_matrix, self.discoeffs)[0][:, 0, :]
        return np.mean(np.linalg.norm(projected_points - image_points, axis=1))

    def get_back_projected_point(self, image_point, rvec, tvec, f, Z=0):
        rmat = cv2.Rodrigues(rvec)[0]
        u = (image_point[0] - self.image_width // 2) / f
        v = (image_point[1] - self.image_height // 2) / f
        coeffcient_x_1 = rmat[0, 0] - u * rmat[2, 0]
        coeffcient_x_2 = rmat[1, 0] - v * rmat[2, 0]
        coeffcient_y_1 = rmat[0, 1] - u * rmat[2, 1]
        coeffcient_y_2 = rmat[1, 1] - v * rmat[2, 1]

        coeffcient_c_1 = 0 - ((tvec[0, 0] - u * tvec[2, 0]) + (rmat[0, 2] - u * rmat[2, 2]) * Z)
        coeffcient_c_2 = 0 - ((tvec[1, 0] - v * tvec[2, 0]) + (rmat[1, 2] - v * rmat[2, 2]) * Z)
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

    def visualization(self, rvec, tvec, f, vehicle_index, model_name):
        corner_1 = (0.25 * self.image_width, 0.25 * self.image_height)
        corner_2 = (0.25 * self.image_width, 0.75 * self.image_height)
        corner_3 = (0.75 * self.image_width, 0.25 * self.image_height)
        corner_4 = (0.75 * self.image_width, 0.75 * self.image_height)
        x1, y1 = self.get_back_projected_point(corner_1, rvec, tvec, f)
        x2, y2 = self.get_back_projected_point(corner_2, rvec, tvec, f)
        x3, y3 = self.get_back_projected_point(corner_3, rvec, tvec, f)
        x4, y4 = self.get_back_projected_point(corner_4, rvec, tvec, f)
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