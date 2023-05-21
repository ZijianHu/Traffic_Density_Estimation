import os
import json
import numpy as np
# from cv2 import cv2
import cv2
from base_model import BaseModel
from itertools import combinations
from camera_calibration.util_files.io import load_image_points, load_object_points


class CandidateGeneration(BaseModel):
    def __init__(self, config):
        super(CandidateGeneration, self).__init__(config)

    def get_pairwise_combination_iter(self, objects_points, image_points):
        comb = combinations(range(len(objects_points)), 4)
        for each in comb:
            yield np.r_[[objects_points[i] for i in each]], np.r_[[image_points[i] for i in each]]

    def calibration(self, object_points, image_points):
        if self.use_ransac:
            retval, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                              self.intrinsic_matrix, self.discoeffs,
                                              flags=cv2.SOLVEPNP_EPNP)
            loss = self.get_projection_error(object_points, image_points,
                                             rvec, tvec, self.focal_length)

        else:
            loss = 1e5
            rvec = None
            tvec = None
            for each in self.get_pairwise_combination_iter(object_points, image_points):
                retval, temp_rvec, temp_tvec = cv2.solvePnP(each[0], each[1],
                                                            self.intrinsic_matrix, self.discoeffs,
                                                            flags=cv2.SOLVEPNP_EPNP)
                temp_loss = self.get_projection_error(object_points, image_points,
                                                      temp_rvec, temp_tvec, self.focal_length)
                if temp_loss < loss:
                    rvec = temp_rvec
                    tvec = temp_tvec
                    loss = temp_loss

        return rvec, tvec, loss

    def run(self):
        print("Candidate Generation...")
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
                images.append(self.visualization(rvec, tvec, self.focal_length, i, key))
                temp_res[key] = {}
                temp_res[key]["rvec"] = rvec.tolist()
                temp_res[key]["tvec"] = tvec.tolist()
                temp_res[key]["f"] = self.focal_length
                temp_res[key]["loss"] = loss.tolist()
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
    config["opt_name"] = "CMA"

    config["object_points_path"] = "../examples/camera_calibration/K109F/objects/vehicle_3dkeypoints_head.json"
    config["image_points_dir"] = "../examples/camera_calibration/K109F/labels"
    config["images_dir"] = "../examples/camera_calibration/K109F/images"
    config["save_dir"] = "../examples/camera_calibration/K109F/outputs"

    config["verbose"] = True
    cg = CandidateGeneration(config)
    cg.run()