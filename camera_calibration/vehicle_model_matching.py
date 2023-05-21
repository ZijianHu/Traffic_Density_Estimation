import os
import json
import numpy as np
# from cv2 import cv2
import cv2
import nevergrad as ng
from base_model import BaseModel
from concurrent import futures
from camera_calibration.util_files.io import load_image_points, load_object_points

class VehicleModelMatching(BaseModel):
    def __init__(self, config):
        super(VehicleModelMatching, self).__init__(config)
        self.cma_object_points = None
        self.cma_image_points = None

    def load_cg_results(self):
        with open(os.path.join(self.save_dir, "cg_results.json")) as f:
            cg_results = json.load(f)
        return cg_results

    def get_projection_error_cma(self, theta1, theta2, theta3, t1, t2, t3, f):
        rvec = np.array([[theta1], [theta2], [theta3]])
        tvec = np.array([[t1], [t2], [t3]])
        return self.get_projection_error(self.cma_object_points, self.cma_image_points, rvec, tvec, f)

    def optimize(self, rvec, tvec, f):
        theta1 = ng.p.Scalar(init=rvec[0][0],
                             lower=rvec[0][0] - 0.3 * np.abs(rvec[0][0]),
                             upper=rvec[0][0] + 0.3 * np.abs(rvec[0][0]))
        theta2 = ng.p.Scalar(init=rvec[1][0],
                             lower=rvec[1][0] - 0.3 * np.abs(rvec[1][0]),
                             upper=rvec[1][0] + 0.3 * np.abs(rvec[1][0]))
        theta3 = ng.p.Scalar(init=rvec[2][0],
                             lower=rvec[2][0] - 0.3 * np.abs(rvec[2][0]),
                             upper=rvec[2][0] + 0.3 * np.abs(rvec[2][0]))
        t1 = ng.p.Scalar(init=tvec[0][0],
                         lower=tvec[0][0] - 0.3 * np.abs(tvec[0][0]),
                         upper=tvec[0][0] + 0.3 * np.abs(tvec[0][0]))
        t2 = ng.p.Scalar(init=tvec[1][0],
                         lower=tvec[1][0] - 0.3 * np.abs(tvec[1][0]),
                         upper=tvec[1][0] + 0.3 * np.abs(tvec[1][0]))
        t3 = ng.p.Scalar(init=tvec[2][0],
                         lower=tvec[2][0] - 0.3 * np.abs(tvec[2][0]),
                         upper=tvec[2][0] + 0.3 * np.abs(tvec[2][0]))
        f = ng.p.Scalar(init=f,
                        lower=f - 0.3 * f,
                        upper=f + 0.3 * f)
        params = ng.p.Instrumentation(theta1, theta2, theta3, t1, t2, t3, f)
        optimizer = ng.optimizers.registry[self.opt_name](parametrization=params,
                                                          budget=self.budget_1,
                                                          num_workers=5)
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(self.get_projection_error_cma,
                                                executor=executor, batch_mode=False)
        res = recommendation.args
        return np.array([[res[0]], [res[1]], [res[2]]]), np.array([[res[3]], [res[4]], [res[5]]]), res[6]

    def run(self):
        print("Vehicle Model Matching...")
        object_points_dict = load_object_points(self.object_points_path)
        image_points_dict = load_image_points(self.image_points_dir)
        cg_results = self.load_cg_results()
        vehicle_num = len(image_points_dict)
        model_num = len(object_points_dict)

        images = []
        res = {}
        for i, each in enumerate(image_points_dict):
            loss_old_min = 1e5; best_old_model = None
            loss_new_min = 1e5; best_new_model = None
            best_rvec = None; best_tvec = None; best_f = None
            res[i] = {}
            for key, value in object_points_dict.items():
                self.cma_object_points, self.cma_image_points = self.prepare_keypoint_pairs(value, each)
                rvec_old = np.array(cg_results[str(i)][key]["rvec"])
                tvec_old = np.array(cg_results[str(i)][key]["tvec"])
                f_old = np.array(cg_results[str(i)][key]["f"])
                loss_old = self.get_projection_error(self.cma_object_points, self.cma_image_points,
                                                     rvec_old, tvec_old, f_old)
                rvec_new, tvec_new, f_new = self.optimize(rvec_old, tvec_old, f_old)
                loss_new = self.get_projection_error(self.cma_object_points, self.cma_image_points,
                                                     rvec_new, tvec_new, f_new)
                if loss_old < loss_old_min:
                    loss_old_min = loss_old
                    best_old_model = key

                if loss_new < loss_new_min:
                    loss_new_min = loss_new
                    best_new_model = key
                    best_rvec = rvec_new
                    best_tvec = tvec_new
                    best_f = f_new

                if self.verbose:
                    print("Before optimizing:")
                    print("\t Vehicle %02d, vehicle model %s, projection loss %f" % (i, key, loss_old))
                    print("After optimizing:")
                    print("\t Vehicle %02d, vehicle model %s, projection loss %f" % (i, key, loss_new))

            if self.verbose:
                print("Before vehicle model matching:")
                print("\t Vehicle %02d, vehicle model %s, projection loss %f" % (i, best_old_model, loss_old_min))
                print("After vehicle model matching:")
                print("\t Vehicle %02d, vehicle model %s, projection loss %f" % (i, best_new_model, loss_new_min))

            images.append(self.visualization(best_rvec, best_tvec, best_f, 
                                             i, best_new_model))
            res[i]["rvec"] = best_rvec.tolist()
            res[i]["tvec"] = best_tvec.tolist()
            res[i]["f"] = best_f
            res[i]["loss"] = loss_new_min.tolist()
            res[i]["model"] = best_new_model
        self.create_montage(images, os.path.join(self.save_dir, "vmm_montage.jpg"), 1, vehicle_num)
        with open(os.path.join(self.save_dir, "vmm_results.json"), "w") as f:
            json.dump(res, f)

if __name__ == "__main__":
    config = {}
    config["camera_code"] = "K109F"
    config["image_height"] = 240
    config["image_width"] = 320
    config["focal_length"] = 350
    config["alpha"] = 6
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
    cg = VehicleModelMatching(config)
    cg.run()
